"""Causal AlphaTrend entry and one-time continuation correction, without depth."""
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .alphatrend import alpha_indicator
from .fast_accounting import audit_simulate
from .indicators_v11 import (SYMBOLS, COLUMNS, build_frame, market, alpha_target, allowed_times)
from .sources_v8 import load
from .timeframes import aggregate

FRAMES = ['2h', '4h']
ENTRY_FEATURES = COLUMNS.copy()
EXIT_FEATURES = COLUMNS+['pnl_atr', 'mfe_atr', 'mae_atr', 'giveback_atr', 'settled_funding_10bps']
CHECK_BARS = 96  # Exactly 24 hours on the 15-minute clock.
KINDS = ['mean', 'ridge', 'boost']
MODES = ['entry', 'exit', 'both']
CONTROLS = ['alpha', 'alpha_half', 'cash', 'exit24h', 'loss1atr24h']
TRAIN_START, END = '2023-01-02', '2026-08-31 20:00'


def dataset():
    bars, frames, atr = {}, {tf: {} for tf in FRAMES}, {tf: {} for tf in FRAMES}
    for symbol in SYMBOLS:
        b, _ = load(symbol)
        bars[symbol] = b
        for tf in FRAMES:
            frames[tf][symbol] = build_frame(b, tf)
            a = alpha_indicator(aggregate(b, tf)).atr_sma.shift(1)
            atr[tf][symbol] = a.reindex(b.index, method='ffill')
    return bars, frames, atr


def context(symbol, bars, frame, atr, start, end):
    m = market(symbol, bars, frame, start, end)
    a = atr.reindex(m.times[:-1]).to_numpy(float)
    closes = bars.close.reindex(m.times[:-1]).to_numpy(float)
    if not np.isfinite(a).all() or np.any(a <= 0) or not np.isfinite(closes).all():
        raise ValueError('Invalid causal ATR/close context')
    return {'market': m, 'atr': a, 'closes': closes}


def cut(m, start, end):
    if not 0 <= start < end <= len(m.x):
        raise ValueError('Empty or invalid market cut')
    return replace(m, times=m.times[start:end+1], x=m.x[start:end], prices=m.prices[start:end+1],
        marks=m.marks[start:end+1], funding=m.funding[start:end+1], lows=m.lows[start:end], highs=m.highs[start:end])


def exit_features(ctx, entry, decision, delay):
    m, a = ctx['market'], entry+delay
    if not a < decision < len(m.x):
        raise ValueError('Checkpoint requires a previously executed entry')
    price, scale = m.prices[a], ctx['atr'][entry]
    last = ctx['closes'][decision-1]
    high = max(price, float(m.highs[a:decision].max()))
    low = min(price, float(m.lows[a:decision].min()))
    # Exclude settlement at the decision timestamp; no publication-time guess.
    paid = m.funding[a+1:decision].sum()*1000
    state = np.clip([(last-price)/scale, (high-price)/scale, (low-price)/scale,
                     (high-last)/scale, paid], -5, 5)
    return np.r_[m.x[decision], state]


def signals(ctx, tf, delay=1):
    """One forward scan; event eligibility uses current/past Alpha state only."""
    m = ctx['market']
    schedule = allowed_times(m, tf)
    episodes, events, active = [], [], None
    for t in np.flatnonzero(schedule):
        on = m.x[t, 0] > 0
        if on and active is None:
            active = len(episodes)
            episodes.append({'entry': int(t), 'exit': None, 'left_censored': bool(t == 0)})
            events.append({'head': 'entry', 'episode': active, 'decision': int(t), 'x': m.x[t].copy()})
        elif not on and active is not None:
            episodes[active]['exit'] = int(t)
            active = None
        if on and active is not None and t == episodes[active]['entry']+CHECK_BARS:
            events.append({'head': 'exit', 'episode': active, 'decision': int(t),
                           'x': exit_features(ctx, episodes[active]['entry'], int(t), delay)})
    return episodes, events


def predict(model, x):
    x = np.atleast_2d(np.asarray(x, float))
    if x.shape[1] != len(model['features']) or not np.isfinite(x).all():
        raise ValueError('Prediction input mismatch')
    if model['kind'] == 'mean':
        return np.full(len(x), model['mean'])
    with threadpool_limits(limits=1):
        return model['estimator'].predict(model['scaler'].transform(x))


def fit_model(rows, head, kind, cutoff):
    cutoff = pd.Timestamp(cutoff, tz='UTC')
    usable = [r for r in rows if r['head'] == head and not r['left_censored']
              and r['known'] is not None and pd.Timestamp(r['known']) < cutoff]
    if len(usable) < 20:
        raise ValueError(f'Insufficient matured {head} labels: {len(usable)}')
    x = np.stack([r['x'] for r in usable])
    y = np.asarray([r['y'] for r in usable], float)
    features = ENTRY_FEATURES if head == 'entry' else EXIT_FEATURES
    if x.shape[1] != len(features) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid training matrix')
    result = {'kind': kind, 'head': head, 'features': features, 'cutoff': str(cutoff),
        'training_ids': [r['id'] for r in usable], 'training_count': len(usable),
        'negative_labels': int((y < 0).sum()), 'mean': float(y.mean()),
        'last_label_known': max(r['known'] for r in usable)}
    if kind != 'mean':
        scaler = StandardScaler().fit(x)
        if kind == 'ridge':
            estimator = Ridge(alpha=10., solver='cholesky')
        elif kind == 'boost':
            estimator = HistGradientBoostingRegressor(loss='squared_error', learning_rate=.05,
                max_iter=100, max_leaf_nodes=7, min_samples_leaf=20, l2_regularization=10.,
                early_stopping=False, random_state=11)
        else:
            raise ValueError('Unknown repair model')
        with threadpool_limits(limits=1):
            estimator.fit(scaler.transform(x), y)
        result.update({'scaler': scaler, 'estimator': estimator})
    result['training_predictions'] = predict(result, x)
    result['training_prediction_std'] = float(np.std(result['training_predictions']))
    result['training_negative_prediction_fraction'] = float(np.mean(result['training_predictions'] < 0))
    durations = [(pd.Timestamp(r['known'])-pd.Timestamp(r['decision'])).total_seconds()/3600 for r in usable]
    result['training_label_delay_hours'] = {'min': float(min(durations)), 'median': float(np.median(durations)), 'max': float(max(durations))}
    result['unresolved_or_not_mature_count'] = sum(r['head'] == head and pd.Timestamp(r['decision']) < cutoff
        and not r['left_censored'] and (r['known'] is None or pd.Timestamp(r['known']) >= cutoff) for r in rows)
    return result


def policy(ctx, tf, candidate, delay=1):
    """Only suppress baseline positions; no additional entries or loss chasing."""
    if delay not in [1, 2]:
        raise ValueError('Unregistered execution latency')
    if not (isinstance(candidate, dict) and candidate.get('mode') in MODES) and candidate not in CONTROLS:
        raise ValueError('Unregistered policy')
    m = ctx['market']
    schedule = allowed_times(m, tf)
    episodes, events = signals(ctx, tf, delay)
    event_at = {e['decision']: e for e in events}
    values, actions, blocked = np.zeros(len(m.x)), [], False
    for t in np.flatnonzero(schedule):
        if t+delay >= len(m.x):
            raise ValueError('Native decision lacks execution before terminal boundary')
        base = float(alpha_target(m.x[t:t+1])[0])
        assert (base > 0) == (m.x[t, 0] > 0)
        event = event_at.get(int(t))
        reason, forecast = '', None
        if base == 0:
            blocked = False
        elif event is not None:
            head = event['head']
            if head == 'entry':
                blocked = False
            if isinstance(candidate, dict):
                use = candidate['mode'] in [head, 'both']
                protected = episodes[event['episode']]['entry'] == 0
                if use and protected:
                    reason = 'boundary_keep'
                if use and not blocked and not protected:
                    forecast = float(predict(candidate[head], event['x'])[0])
                    if forecast < 0:
                        blocked, reason = True, 'veto' if head == 'entry' else 'early_exit'
            elif head == 'exit' and not blocked:
                if candidate == 'exit24h' or (candidate == 'loss1atr24h' and event['x'][len(COLUMNS)] < -1):
                    blocked, reason = True, 'rule_exit'
            actions.append({'episode': event['episode'], 'head': head, 'decision': int(t),
                'time': str(m.times[t]), 'execution': str(m.times[min(t+delay, len(m.x))]),
                'prediction': forecast, 'blocked': bool(blocked), 'reason': reason,
                'boundary_protected': bool(isinstance(candidate, dict) and episodes[event['episode']]['entry'] == 0)})
        values[t] = 0. if blocked or candidate == 'cash' else base*(.5 if candidate == 'alpha_half' else 1.)
    anchor = np.maximum.accumulate(np.where(schedule, np.arange(len(m.x)), 0))
    held = values[anchor]
    positions = np.r_[np.zeros(delay), held[:-delay]]
    allowed = np.r_[np.zeros(delay, bool), schedule[:-delay]]
    return positions, allowed, actions


def labelled_events(ctx, tf):
    """Labels mature at the natural Alpha exit execution, never at a truncation."""
    m = ctx['market']
    episodes, events = signals(ctx, tf)
    p, allowed, _ = policy(ctx, tf, 'alpha')
    labels, diagnostics = [], []
    for i, ep in enumerate(episodes):
        a = ep['entry']+1
        natural = ep['exit'] is not None and ep['exit']+1 <= len(m.x)
        z = ep['exit']+1 if natural else len(m.x)
        if a >= z:
            raise ValueError('Registered native schedule produced unexecutable training entry')
        r = audit_simulate(cut(m, a, z), p[a:z], allowed[a:z])
        known = str(m.times[z]) if natural else None
        diagnostics.append({'episode': i, 'start': str(m.times[a]), 'end': str(m.times[z]),
            'left_censored': ep['left_censored'], 'right_censored': not natural,
            'net_return': r['return'], 'price_pnl': r['return']+r['cost_paid_initial_equity']+r['funding_paid_initial_equity'],
            'fees': r['cost_paid_initial_equity'], 'funding': r['funding_paid_initial_equity'],
            'hold_hours': (z-a)/4})
        for event in (v for v in events if v['episode'] == i):
            if event['head'] == 'entry':
                y = r['return']
            else:
                c = event['decision']+1
                early = audit_simulate(cut(m, a, c), p[a:c], allowed[a:c])
                y = r['return']-early['return']
            labels.append({'id': f'{m.symbol}_{m.times[ep["entry"]].isoformat()}_{event["head"]}',
                'head': event['head'], 'decision': str(m.times[event['decision']]), 'known': known,
                'left_censored': ep['left_censored'], 'x': event['x'].tolist(), 'y': float(y) if natural else None})
    return labels, diagnostics


def attribution(ctx, tf, baseline, candidate, delay=1):
    """Exact additive per-asset log-wealth change, grouped by baseline outcome."""
    m = ctx['market']
    episodes, _ = signals(ctx, tf, delay)
    records = []
    for i, ep in enumerate(episodes):
        a = ep['entry']+delay
        z = min(ep['exit']+delay, len(m.x)) if ep['exit'] is not None else len(m.x)
        if a >= z:
            continue
        # Equity[z] excludes a normal order at z; charge its exit fee to this episode.
        def closed_log(r):
            if z == len(m.x):
                finish = r['equity'][z]  # Terminal close already charged.
            else:
                # Both curves remain flat from the exit until the next Alpha entry.
                finish = r['equity'][z+1]
            return float(np.log(finish/r['equity'][a]))
        b, c = closed_log(baseline), closed_log(candidate)
        records.append({'episode': i, 'start': str(m.times[a]), 'end': str(m.times[z]),
            'left_censored': ep['left_censored'], 'right_censored': ep['exit'] is None or ep['exit']+delay > len(m.x),
            'baseline_log': b, 'candidate_log': c, 'delta_log': c-b,
            'baseline_loss': bool(b < 0)})
    np.testing.assert_allclose(sum(r['baseline_log'] for r in records), np.log1p(baseline['return']), atol=2e-10)
    np.testing.assert_allclose(sum(r['candidate_log'] for r in records), np.log1p(candidate['return']), atol=2e-10)
    return records
