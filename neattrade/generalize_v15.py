"""Frozen generalization interventions; research only, no controller integration."""
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .adaptive_v13 import MONTHS, BOOST_PARAMS, utc
from .repair_v12 import FRAMES, SYMBOLS, EXIT_FEATURES, signals, predict
from .indicators_v11 import alpha_target, allowed_times

FEATURES = ['alpha_distance', 'alpha_slope', 'rsi14', 'atr14_relative',
            'taker_fraction', 'risk_vol', 'pnl_atr', 'giveback_atr']
INDICES = [EXIT_FEATURES.index(f) for f in FEATURES]
ARMS = ['alpha', 'current', 'compact8', 'robust8', 'temporal_mean',
        'temporal_guard', 'ridge8', 'transfer_current', 'transfer_compact', 'transfer_robust']
COMPACT_PARAMS = dict(loss='squared_error', learning_rate=.03, max_iter=100,
    max_leaf_nodes=3, max_depth=2, min_samples_leaf=20, l2_regularization=50.,
    early_stopping=False, random_state=11)


def key(tf, family, days, heldout, date):
    return f'{tf}_{family}_{days or "all"}_{heldout or "pooled"}_{utc(date):%Y%m}'


def specifications():
    return [('compact', None, None), ('robust', None, None), ('robust', 730, None),
            ('robust', 365, None), ('ridge', None, None)] + [
            (f, None, s) for s in SYMBOLS for f in ['current', 'compact', 'robust']]


def cohort(rows, tf, available, days=None, heldout=None):
    if tf not in FRAMES or heldout not in [None, *SYMBOLS] or days not in [None, 365, 730]:
        raise ValueError('Unregistered cohort')
    now = utc(available)
    if str(now.date()) not in MONTHS or now != now.normalize():
        raise ValueError('Unregistered monthly version')
    cutoff, use, seen = now-pd.Timedelta(days=7), [], set()
    for r in rows:
        symbol = r['id'].split('_')[0]
        if r['id'] in seen or symbol not in SYMBOLS or r.get('timeframe', tf) != tf:
            raise ValueError('Invalid row identity/timeframe')
        seen.add(r['id'])
        if r['head'] not in ['entry', 'exit'] or not r['id'].endswith('_'+r['head']):
            raise ValueError('Invalid head identity')
        decision = utc(r['decision'])
        known = utc(r['known']) if r['known'] is not None else None
        if pd.isna(decision) or (known is not None and (pd.isna(known) or known <= decision)):
            raise ValueError('Invalid label timing')
        if r['head'] != 'exit' or r['left_censored'] or known is None or known >= cutoff or symbol == heldout:
            continue
        if days is not None and decision < cutoff-pd.Timedelta(days=days):
            continue
        x = np.asarray(r['x'], float)
        if x.shape != (18,) or not np.isfinite(x).all() or r['y'] is None or not np.isfinite(r['y']):
            raise ValueError('Invalid eligible training payload')
        use.append(r)
    return use


def train(rows, tf, family, days, heldout, date):
    if (family, days, heldout) not in specifications():
        raise ValueError('Unregistered fit')
    use = cohort(rows, tf, date, days, heldout)
    robust = family in ['robust', 'ridge']
    indices = list(range(18)) if family == 'current' else INDICES
    y = np.asarray([r['y'] for r in use], float)
    groups = [r['id'].split('_')[0]+'_'+str(utc(r['decision']).year)+'H'+str(1+(utc(r['decision']).month>6)) for r in use]
    counts = Counter(groups)
    weights = np.asarray([1/np.sqrt(counts[g]) if robust else 1. for g in groups])
    if len(weights):
        weights /= weights.mean()
    limits = np.quantile(y, [.025, .975], method='linear').tolist() if robust and len(y) else None
    target = np.clip(y, *limits) if limits is not None else y.copy()
    m = dict(name=key(tf, family, days, heldout, date), family=family, days=days,
        heldout=heldout, timeframe=tf, available=str(utc(date)), cutoff=str(utc(date)-pd.Timedelta(days=7)),
        input_indices=indices, features=[EXIT_FEATURES[i] for i in indices], head='exit',
        training_ids=[r['id'] for r in use], training_count=len(use), groups=groups,
        group_counts=dict(counts), weights=weights, raw_y=y, fit_y=target, clip_limits=limits,
        last_label_known=max((r['known'] for r in use), default=None), fitted=len(use)>=20,
        mean=float(np.average(target, weights=weights)) if len(use) else 0.,
        effective_weight_count=float(weights.sum()**2/np.square(weights).sum()) if len(use) else 0.)
    if not m['fitted']:
        m.update(kind='mean', mean=0., training_predictions=np.zeros(len(use)))
        return m
    x = np.asarray([r['x'] for r in use])[:, indices]
    scaler = StandardScaler().fit(x, sample_weight=weights if robust else None)
    estimator = Ridge(alpha=50., solver='cholesky') if family == 'ridge' else HistGradientBoostingRegressor(
        **(BOOST_PARAMS if family == 'current' else COMPACT_PARAMS))
    with threadpool_limits(limits=1):
        estimator.fit(scaler.transform(x), target, sample_weight=weights)
    m.update(kind='ridge' if family == 'ridge' else 'boost', scaler=scaler, estimator=estimator)
    m['training_predictions'] = predict(m, x)
    return m


def components(tf, arm, date, symbol):
    if arm not in ARMS or symbol not in SYMBOLS or tf not in FRAMES:
        raise ValueError('Unregistered policy')
    if str(utc(date).replace(day=1).date()) not in MONTHS:
        raise ValueError('Outside frozen evaluation')
    if arm == 'alpha':
        return []
    if arm == 'current':
        return [f'{tf}_expanding_{utc(date):%Y%m}']
    if arm.startswith('transfer_'):
        return [key(tf, arm.removeprefix('transfer_'), None, symbol, date)]
    if arm.startswith('temporal_'):
        return [key(tf, 'robust', d, None, date) for d in [None, 730, 365]]
    return [key(tf, arm.removesuffix('8'), None, None, date)]


def forecast(models, versions, x, arm):
    if any(not models[v].get('fitted', True) for v in versions):
        return 0., [], 'insufficient'
    values = [float(predict(models[v], np.asarray(x)[models[v].get('input_indices', list(range(18)))])[0]) for v in versions]
    score = max(values) if arm == 'temporal_guard' else float(np.mean(values))
    reason = 'early_exit' if score < 0 else 'disagreement' if arm == 'temporal_guard' and min(values)<0 else 'keep'
    return score, values, reason


def policy(ctx, tf, arm, models, delay=1):
    m = ctx['market']
    allowed = allowed_times(m, tf)
    episodes, events = signals(ctx, tf, delay)
    event_at = {e['decision']: e for e in events}
    if len(event_at) != len(events):
        raise ValueError('Colliding decision events')
    target, actions, versions, blocked, active = np.zeros(len(m.x)), [], [], False, None
    for t in np.flatnonzero(allowed):
        if t+delay >= len(m.x):
            raise ValueError('Missing execution interval')
        base = float(alpha_target(m.x[t:t+1])[0])
        if base == 0:
            blocked = False
        event = event_at.get(int(t))
        if event is not None:
            ep = episodes[event['episode']]
            protected = ep['entry'] == 0
            score, values, reason = None, [], ''
            if event['head'] == 'entry':
                blocked, active = False, event['episode']
                versions = [] if protected else components(tf, arm, m.times[t], m.symbol)
                if any(utc(models[v]['available']) > m.times[t] for v in versions):
                    raise ValueError('Manager created after entry')
                reason = 'boundary_keep' if protected else 'latch'
            else:
                if active != event['episode']:
                    raise ValueError('Checkpoint lacks entry')
                if versions and not blocked:
                    score, values, reason = forecast(models, versions, event['x'], arm)
                    blocked = score < 0
            actions.append(dict(episode=event['episode'], head=event['head'], decision=int(t),
                entry=str(m.times[ep['entry']]), time=str(m.times[t]), execution=str(m.times[t+delay]),
                versions=list(versions), prediction=score, component_predictions=values,
                blocked=bool(blocked), reason=reason, boundary_protected=protected))
        target[t] = 0. if blocked else base
    anchors = np.maximum.accumulate(np.where(allowed, np.arange(len(m.x)), 0))
    return np.r_[np.zeros(delay), target[anchors][:-delay]], np.r_[np.zeros(delay, bool), allowed[:-delay]], actions
