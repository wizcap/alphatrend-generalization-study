"""Monthly causal learning, prospective model screening and version-pinned exits."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .fast_accounting import audit_simulate
from .indicators_v11 import alpha_target, allowed_times
from .repair_v12 import (FRAMES, SYMBOLS, TRAIN_START, END, EXIT_FEATURES, context,
    signals, policy as fixed_policy, cut, fit_model, predict)

MONTHS = [str(t.date()) for t in pd.date_range('2024-01-01', '2026-08-01', freq='MS')]
ARMS = ['alpha', 'alpha_half', 'frozen', 'semiannual', 'monthly', 'weighted', 'guarded']
BOOST_PARAMS = dict(loss='squared_error', learning_rate=.05, max_iter=100, max_leaf_nodes=7,
    min_samples_leaf=20, l2_regularization=10., early_stopping=False, random_state=11)


def utc(value):
    return pd.Timestamp(value).tz_localize('UTC') if pd.Timestamp(value).tzinfo is None else pd.Timestamp(value).tz_convert('UTC')


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def version_id(tf, method, date):
    return f'{tf}_{method}_{utc(date):%Y%m}'


def model_digest(model):
    # Semantic prediction/fit fingerprint, stable across pickle reloads and array layout.
    value = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in model.items() if k not in ['estimator', 'scaler']}
    if 'scaler' in model:
        value['scaler'] = {k: getattr(model['scaler'], k).tolist() for k in ['mean_', 'scale_', 'var_']}
        est = model['estimator']
        value['params'] = est.get_params()
        value['baseline'] = est._baseline_prediction.tolist()
        value['trees'] = [[{k: t.nodes[k].tolist() for k in t.nodes.dtype.names} for t in trees] for trees in est._predictors]
    return digest(value)


def train_version(rows, tf, method, available):
    if tf not in FRAMES or any(r.get('timeframe', tf) != tf for r in rows):
        raise ValueError('Training timeframe mismatch')
    available = utc(available)
    cutoff = available-pd.Timedelta(days=7)
    if method == 'expanding':
        model = fit_model(rows, 'exit', 'boost', str(cutoff.tz_localize(None)))
        weights = np.ones(model['training_count'])
    elif method == 'weighted':
        use = [r for r in rows if r['head'] == 'exit' and not r['left_censored']
               and r['known'] is not None and utc(r['known']) < cutoff]
        if len(use) < 20:
            raise ValueError('Insufficient matured training labels')
        x, y = np.asarray([r['x'] for r in use]), np.asarray([r['y'] for r in use])
        if not np.isfinite(x).all() or not np.isfinite(y).all() or x.shape[1] != 18:
            raise ValueError('Invalid training data')
        ages = np.asarray([(cutoff-utc(r['decision'])).total_seconds()/86400 for r in use])
        weights = np.exp2(-ages/180.)
        weights *= len(weights)/weights.sum()
        scaler = StandardScaler().fit(x, sample_weight=weights)
        estimator = HistGradientBoostingRegressor(**BOOST_PARAMS)
        with threadpool_limits(limits=1):
            estimator.fit(scaler.transform(x), y, sample_weight=weights)
        model = {'kind': 'boost', 'head': 'exit', 'features': EXIT_FEATURES,
            'scaler': scaler, 'estimator': estimator, 'cutoff': str(cutoff),
            'training_ids': [r['id'] for r in use], 'training_count': len(use),
            'last_label_known': max(r['known'] for r in use), 'mean': float(np.average(y, weights=weights))}
        model['training_predictions'] = predict(model, x)
    else:
        raise ValueError('Unregistered learning method')
    if any(model['estimator'].get_params()[k] != v for k, v in BOOST_PARAMS.items()):
        raise ValueError('Fixed learner settings drifted from study012')
    model.update({'name': version_id(tf, method, available), 'method': method, 'timeframe': tf,
        'available': str(available), 'weights': weights,
        'effective_weight_count': float(weights.sum()**2/np.square(weights).sum())})
    return model


def feedback_catalog(bars, frames, atr):
    result = {tf: [] for tf in FRAMES}
    for tf in FRAMES:
        for symbol in SYMBOLS:
            ctx = context(symbol, bars[symbol], frames[tf][symbol], atr[tf][symbol], TRAIN_START, END)
            m = ctx['market']
            episodes, events = signals(ctx, tf)
            checkpoints = {v['episode']: v for v in events if v['head'] == 'exit'}
            p, a, _ = fixed_policy(ctx, tf, 'alpha')
            for i, ep in enumerate(episodes):
                start = ep['entry']+1
                end = ep['exit']+1 if ep['exit'] is not None else len(m.x)
                if not 0 <= start < end <= len(m.x):
                    raise ValueError('Shadow payoff execution bounds invalid')
                event = checkpoints.get(i)
                row = {'id': f'{symbol}_{m.times[ep["entry"]].isoformat()}', 'symbol': symbol,
                    'entry': str(m.times[ep['entry']]), 'checkpoint': str(m.times[event['decision']]) if event else None,
                    'known': str(m.times[end]) if ep['exit'] is not None else None,
                    'left_censored': ep['left_censored'], 'x': event['x'].tolist() if event else None,
                    'base': {}, 'early': {}}
                for name, cost in [('base', 7.), ('double_cost', 14.)]:
                    row['base'][name] = float(audit_simulate(cut(m, start, end), p[start:end], a[start:end], cost)['return'])
                    close = event['decision']+1 if event else end
                    row['early'][name] = float(audit_simulate(cut(m, start, close), p[start:close], a[start:close], cost)['return'])
                if any(not np.isfinite(v) or v <= -1 for side in ['base', 'early'] for v in row[side].values()):
                    raise ValueError('Invalid or insolvent shadow payoff')
                result[tf].append(row)
    return result


def matured(records, boundary, start):
    cutoff = utc(boundary)-pd.Timedelta(days=7)
    first = max(utc(start), utc(boundary)-pd.Timedelta(days=180))
    return [r for r in records if not r['left_censored'] and utc(r['entry']) >= first
            and r['known'] is not None and utc(r['known']) < cutoff]


def screen(records, model, incumbent=None):
    """Only resolved shadow payoffs. This is not an intratrade drawdown test."""
    checkpoints = [r for r in records if r['x'] is not None]
    for fitted in [model, incumbent]:
        if fitted is not None:
            if any(utc(r['entry']) < utc(fitted['available']) for r in records):
                raise ValueError('A model cannot score pre-creation validation episodes')
    forecasts = predict(model, [r['x'] for r in checkpoints]) if checkpoints else []
    exits = {r['id'] for r, f in zip(checkpoints, forecasts) if f < 0}
    old_predictions = predict(incumbent, [r['x'] for r in checkpoints]) if incumbent is not None and checkpoints else []
    old_exits = {r['id'] for r, f in zip(checkpoints, old_predictions) if f < 0}
    economic = {}
    for cost in ['base', 'double_cost']:
        base = np.asarray([np.log1p(r['base'][cost]) for r in records])
        candidate = np.asarray([np.log1p(r['early' if r['id'] in exits else 'base'][cost]) for r in records])
        reference = np.asarray([np.log1p(r['early' if r['id'] in old_exits else 'base'][cost]) for r in records])
        losing = base < 0
        lost, won = -base[losing].sum(), base[~losing].sum()
        economic[cost] = {'alpha_excess': float((candidate-base).sum()),
            'incumbent_excess': float((candidate-reference).sum()),
            'repair': float((candidate-base)[losing].sum()/lost) if lost > 0 else None,
            'retention': float(candidate[~losing].sum()/won) if won > 0 else None,
            'losers': int(losing.sum()), 'winners': int((~losing).sum())}
    return {'ids': [r['id'] for r in records], 'checkpoints': len(checkpoints),
        'by_symbol': {s: sum(r['symbol'] == s for r in checkpoints) for s in SYMBOLS},
        'early_exits': len(exits), 'economics': economic}


def promotion_checks(value):
    a, d = value['economics']['base'], value['economics']['double_cost']
    return {'sample': value['checkpoints'] >= 20 and min(value['by_symbol'].values()) >= 4,
        'outcomes': a['losers'] >= 3 and a['winners'] >= 3 and value['early_exits'] >= 2,
        'alpha_base': a['alpha_excess'] > 0, 'alpha_double': d['alpha_excess'] > 0,
        'incumbent_base': a['incumbent_excess'] > 0, 'incumbent_double': d['incumbent_excess'] >= 0,
        'repair': a['repair'] is not None and a['repair'] >= .20,
        'retention': a['retention'] is not None and a['retention'] >= .90}


class Controller:
    def __init__(self, tf, protocol_hash, state=None):
        self.tf, self.protocol_hash = tf, protocol_hash
        self.state = deepcopy(state) if state else {'schema': 1, 'timeframe': tf, 'protocol_hash': protocol_hash,
            'active': None, 'activated': None, 'challenger': None, 'queued': None, 'incumbent': None,
            'last_boundary': None, 'last_input': None, 'last_versions': [], 'history': []}
        if (self.state['schema'], self.state['timeframe'], self.state['protocol_hash']) != (1, tf, protocol_hash):
            raise ValueError('Controller state fingerprint mismatch')

    def step(self, boundary, records, models):
        now = utc(boundary)
        if now != now.normalize() or now.day != 1:
            raise ValueError('Updates require a UTC month boundary')
        s = deepcopy(self.state)
        current = version_id(self.tf, 'weighted', now)
        if current not in models or utc(models[current]['available']) != now:
            raise ValueError('Current monthly candidate unavailable')
        known = sorted([r for r in records if r['known'] is not None and utc(r['known']) < now-pd.Timedelta(days=7)], key=lambda r: r['id'])
        required = s['last_versions'] if s['last_boundary'] is not None and now == utc(s['last_boundary']) else sorted(
            {v for v in [current, s['active'], s['challenger'], s['incumbent']] if v is not None})
        if any(v not in models or utc(models[v]['available']) > now for v in required):
            raise ValueError('A state-referenced model is missing or not yet available')
        incoming = digest({'known': known, 'versions': {v: model_digest(models[v]) for v in required}})
        if s['last_boundary'] is not None:
            last = utc(s['last_boundary'])
            if now == last:
                if incoming != s['last_input']:
                    raise ValueError('Changed input at an already processed boundary')
                return deepcopy(s['history'][-1])
            if now != last+pd.offsets.MonthBegin(1):
                raise ValueError('Out-of-order or skipped update boundary')
        record = {'boundary': str(now), 'before': s['active'], 'actions': [], 'health': None, 'validation': None,
                  'health_status': 'inactive', 'validation_status': 'newly_queued'}
        rolled_back = False
        if s['active'] is not None:
            health = screen(matured(known, now, s['activated']), models[s['active']])
            record['health'] = health
            enough = health['checkpoints'] >= 12 and min(health['by_symbol'].values()) >= 3
            record['health_status'] = 'healthy' if enough else 'insufficient_evidence'
            if enough and any(v['alpha_excess'] < 0 for v in health['economics'].values()):
                record['actions'].append('rollback')
                s.update(active=None, activated=None, challenger=None, queued=None, incumbent=None)
                rolled_back = True
                record.update(health_status='rollback', validation_status='cancelled_by_rollback')
        if s['challenger'] is not None and not rolled_back:
            value = screen(matured(known, now, s['queued']), models[s['challenger']], models.get(s['incumbent']))
            checks = promotion_checks(value)
            record['validation'] = {'version': s['challenger'], 'queued': s['queued'], 'incumbent': s['incumbent'], **value, 'checks': checks}
            record['validation_status'] = 'observing'
            if checks['sample'] and checks['outcomes']:
                if all(checks.values()):
                    s['active'], s['activated'] = s['challenger'], str(now)
                    record['actions'].append('promote')
                    record['validation_status'] = 'promoted'
                else:
                    record['actions'].append('reject')
                    record['validation_status'] = 'rejected'
                s.update(challenger=None, queued=None, incumbent=None)
            elif now-utc(s['queued']) >= pd.Timedelta(days=180):
                record['actions'].append('expire')
                record['validation_status'] = 'expired'
                s.update(challenger=None, queued=None, incumbent=None)
        if s['challenger'] is None:
            s.update(challenger=current, queued=str(now), incumbent=s['active'])
            record['actions'].append('queue')
        record.update({'active': s['active'], 'activated': s['activated'], 'challenger': s['challenger'], 'queued': s['queued']})
        s.update(last_boundary=str(now), last_input=incoming, last_versions=required)
        s['history'].append(record)
        self.state = s
        return deepcopy(record)

    def save(self, path):
        path = Path(path)
        payload = {'state': self.state, 'sha256': digest(self.state)}
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        tmp.replace(path)

    @classmethod
    def load(cls, path, tf, protocol_hash):
        payload = json.loads(Path(path).read_text())
        if digest(payload['state']) != payload['sha256']:
            raise ValueError('Corrupt controller state')
        return cls(tf, protocol_hash, payload['state'])


def selected_version(tf, arm, date, schedule):
    now = utc(date)
    if not utc(MONTHS[0]) <= now < utc('2026-09-01'):
        raise ValueError('Outside the registered study013 evaluation dates')
    if arm in ['alpha', 'alpha_half']:
        return None
    if arm == 'guarded':
        key = str(now.replace(day=1, hour=0, minute=0, second=0, microsecond=0))
        return schedule[key]
    if arm == 'frozen':
        now = utc('2024-01-01')
    elif arm == 'semiannual':
        now = utc(f'{min(now.year, 2026)}-{"07" if now.year < 2026 and now.month >= 7 else "01"}-01')
    elif arm not in ['monthly', 'weighted']:
        raise ValueError('Unregistered policy')
    return version_id(tf, 'weighted' if arm == 'weighted' else 'expanding', now)


def policy(ctx, tf, arm, models, schedule, delay=1):
    m = ctx['market']
    allowed = allowed_times(m, tf)
    episodes, events = signals(ctx, tf, delay)
    event_at = {v['decision']: v for v in events}
    if len(event_at) != len(events):
        raise ValueError('Colliding Alpha decision events')
    target, actions, version, blocked, active_episode = np.zeros(len(m.x)), [], None, False, None
    for t in np.flatnonzero(allowed):
        if t+delay >= len(m.x):
            raise ValueError('Decision lacks an execution before terminal boundary')
        base = float(alpha_target(m.x[t:t+1])[0])
        if base == 0:
            blocked = False
        event = event_at.get(int(t))
        if event is not None:
            protected = episodes[event['episode']]['entry'] == 0
            forecast, reason = None, ''
            if event['head'] == 'entry':
                blocked = False
                active_episode = event['episode']
                version = None if protected else selected_version(tf, arm, m.times[t], schedule)
                if version is not None:
                    if utc(models[version]['available']) > m.times[t]:
                        raise ValueError('Position manager created after entry')
                reason = 'boundary_keep' if protected else 'latch'
            else:
                if active_episode != event['episode']:
                    raise ValueError('Checkpoint has no matching latched entry')
                if version is not None and not blocked:
                    forecast = float(predict(models[version], event['x'])[0])
                    if forecast < 0:
                        blocked, reason = True, 'early_exit'
            actions.append({'episode': event['episode'], 'head': event['head'], 'decision': int(t),
                'time': str(m.times[t]), 'execution': str(m.times[t+delay]), 'version': version,
                'prediction': forecast, 'blocked': bool(blocked), 'reason': reason, 'boundary_protected': protected})
        target[t] = (0. if blocked else base)*(.5 if arm == 'alpha_half' else 1.)
    anchors = np.maximum.accumulate(np.where(allowed, np.arange(len(m.x)), 0))
    return np.r_[np.zeros(delay), target[anchors][:-delay]], np.r_[np.zeros(delay, bool), allowed[:-delay]], actions
