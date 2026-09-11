"""Independent feedback, model, version-controller, action and account replay."""
from hashlib import sha256
import json
from pathlib import Path
import pickle
import re
import subprocess
import sys

import numpy as np
import pandas as pd

from .adaptive_v13 import (FRAMES, SYMBOLS, MONTHS, ARMS, TRAIN_START, END, utc, digest,
    feedback_catalog, train_version, version_id, Controller, policy)
from .repair_v12 import dataset, context, predict
from .verify_v12 import reference_policy as fixed_reference, reference_state, reference_predict, spans, reference_repair
from .market import simulate
from .fast_accounting import audit_simulate
from .research_v13 import OUT, WINDOWS, SCENARIOS, contrasts, interval, INTERVAL_COUNT
from .research_v11 import verify_raw_sources, grouped
from .research_v6 import write_json
from .timeframes import TIMEFRAMES, metrics, portfolio, daily_returns


def equal(a, b, tolerance=2e-9):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (a.keys(), b.keys())
        for k in a:
            equal(a[k], b[k], tolerance)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for u, v in zip(a, b):
            equal(u, v, tolerance)
    elif isinstance(a, (float, int, np.number)) and not isinstance(a, bool):
        np.testing.assert_allclose(a, b, atol=tolerance, rtol=0)
    else:
        assert a == b, (a, b)


def reference_feedback(ctx, tf):
    m = ctx['market']
    p, allowed, _ = fixed_reference(ctx, tf, 'alpha', 1)
    rows = []
    for e, z in spans(ctx, tf):
        c = e+96
        checkpoint = c < z and c < len(m.x)
        rows.append({'id': f'{m.symbol}_{m.times[e].isoformat()}', 'symbol': m.symbol,
            'entry': str(m.times[e]), 'checkpoint': str(m.times[c]) if checkpoint else None,
            'known': str(m.times[z+1]) if z < len(m.x) else None, 'left_censored': e == 0,
            'x': reference_state(ctx, e, c, 1).tolist() if checkpoint else None,
            'base': {}, 'early': {}, '_entry': e+1, '_exit': min(z+1, len(m.x)), '_checkpoint': c+1 if checkpoint else None})
    for scenario, cost in [('base', 7.), ('double_cost', 14.)]:
        r = simulate(m, p, cost, 0., trade_allowed=allowed, rebalance_band=.05, trace=True)
        for v in rows:
            first, end, c = v['_entry'], v['_exit'], v['_checkpoint']
            initial = r['equity'][first]
            finish = r['equity'][min(end+1, len(m.x))]
            v['base'][scenario] = float(finish/initial-1)
            if c is None:
                v['early'][scenario] = v['base'][scenario]
            else:
                closed = r['equity'][c]-abs(r['ledger'][c-1]['units'])*m.prices[c]*cost/10000
                v['early'][scenario] = float(closed/initial-1)
    return [{k: v for k, v in r.items() if not k.startswith('_')} for r in rows]


def reference_score(records, model, old):
    selected, incumbent = set(), set()
    counts = {s: 0 for s in SYMBOLS}
    for row in records:
        if row['x'] is None:
            continue
        counts[row['symbol']] += 1
        if reference_predict(model, row['x'])[0] < 0:
            selected.add(row['id'])
        if old is not None and reference_predict(old, row['x'])[0] < 0:
            incumbent.add(row['id'])
    economics = {}
    for cost in ['base', 'double_cost']:
        loss = gain = saved = kept = alpha_delta = old_delta = 0.
        losers = winners = 0
        for r in records:
            b = np.log1p(r['base'][cost])
            v = np.log1p(r['early' if r['id'] in selected else 'base'][cost])
            old_v = np.log1p(r['early' if r['id'] in incumbent else 'base'][cost])
            alpha_delta += v-b; old_delta += v-old_v
            if b < 0:
                loss -= b; saved += v-b; losers += 1
            else:
                gain += b; kept += v; winners += 1
        economics[cost] = {'alpha_excess': float(alpha_delta), 'incumbent_excess': float(old_delta),
            'repair': float(saved/loss) if loss > 0 else None, 'retention': float(kept/gain) if gain > 0 else None,
            'losers': losers, 'winners': winners}
    return {'ids': [r['id'] for r in records], 'checkpoints': sum(counts.values()), 'by_symbol': counts,
            'early_exits': len(selected), 'economics': economics}


def reference_controller(records, models, tf):
    active = activated = pending = queued = opponent = None
    history = []
    for date in MONTHS:
        now = utc(date)
        eligible = lambda start: [r for r in records if not r['left_censored'] and utc(r['entry']) >= max(utc(start), now-pd.Timedelta(days=180))
            and r['known'] is not None and utc(r['known']) < now-pd.Timedelta(days=7)]
        event = {'boundary': str(now), 'before': active, 'actions': [], 'health': None, 'validation': None,
            'health_status': 'inactive', 'validation_status': 'newly_queued'}
        if active is not None:
            v = reference_score(eligible(activated), models[active], None)
            event['health'] = v
            event['health_status'] = 'healthy' if v['checkpoints'] >= 12 and all(n >= 3 for n in v['by_symbol'].values()) else 'insufficient_evidence'
            if v['checkpoints'] >= 12 and all(n >= 3 for n in v['by_symbol'].values()) and (v['economics']['base']['alpha_excess'] < 0 or v['economics']['double_cost']['alpha_excess'] < 0):
                active = activated = pending = queued = opponent = None
                event['actions'].append('rollback')
                event.update(health_status='rollback', validation_status='cancelled_by_rollback')
        if pending is not None:
            v = reference_score(eligible(queued), models[pending], models.get(opponent))
            b, d = v['economics']['base'], v['economics']['double_cost']
            checks = {'sample': v['checkpoints'] >= 20 and all(n >= 4 for n in v['by_symbol'].values()),
                'outcomes': b['losers'] >= 3 and b['winners'] >= 3 and v['early_exits'] >= 2,
                'alpha_base': b['alpha_excess'] > 0, 'alpha_double': d['alpha_excess'] > 0,
                'incumbent_base': b['incumbent_excess'] > 0, 'incumbent_double': d['incumbent_excess'] >= 0,
                'repair': b['repair'] is not None and b['repair'] >= .2,
                'retention': b['retention'] is not None and b['retention'] >= .9}
            event['validation'] = {'version': pending, 'queued': queued, 'incumbent': opponent, **v, 'checks': checks}
            event['validation_status'] = 'observing'
            if checks['sample'] and checks['outcomes']:
                if all(checks.values()):
                    active, activated = pending, str(now)
                    event['actions'].append('promote')
                    event['validation_status'] = 'promoted'
                else:
                    event['actions'].append('reject')
                    event['validation_status'] = 'rejected'
                pending = queued = opponent = None
            elif (now-utc(queued)).days >= 180:
                pending = queued = opponent = None
                event['actions'].append('expire')
                event['validation_status'] = 'expired'
        if pending is None:
            pending, queued, opponent = f'{tf}_weighted_{now:%Y%m}', str(now), active
            event['actions'].append('queue')
        event.update(active=active, activated=activated, challenger=pending, queued=queued)
        history.append(event)
    return history


def reference_policy(ctx, tf, arm, models, schedule, delay):
    m, stride = ctx['market'], TIMEFRAMES[tf]//15
    targets, allowed, actions = np.zeros(len(m.x)), np.zeros(len(m.x), bool), []
    previous, blocked, entry, episode, version = False, False, -1, -1, None
    for t in range(0, len(m.x), stride):
        on = m.x[t, 0] > 0
        head = None
        if on and not previous:
            entry, episode, blocked, head = t, episode+1, False, 'entry'
        elif on and t-entry == 96:
            head = 'exit'
        elif not on:
            blocked = False
        if head:
            forecast, reason = None, ''
            if head == 'entry':
                date = m.times[t]
                stamp = date.strftime('%Y%m')
                if entry == 0 or arm in ['alpha', 'alpha_half']:
                    version = None
                elif arm == 'guarded':
                    version = schedule[str(pd.Timestamp(year=date.year, month=date.month, day=1, tz='UTC'))]
                else:
                    if arm == 'frozen':
                        stamp = '202401'
                    elif arm == 'semiannual':
                        stamp = '202601' if date.year >= 2026 else f'{date.year}{"07" if date.month >= 7 else "01"}'
                    version = f'{tf}_{"weighted" if arm == "weighted" else "expanding"}_{stamp}'
                reason = 'boundary_keep' if entry == 0 else 'latch'
            elif version is not None and not blocked:
                forecast = float(reference_predict(models[version], reference_state(ctx, entry, t, delay))[0])
                if forecast < 0:
                    blocked, reason = True, 'early_exit'
            actions.append({'episode': episode, 'head': head, 'decision': t, 'time': str(m.times[t]),
                'execution': str(m.times[t+delay]), 'version': version, 'prediction': forecast,
                'blocked': bool(blocked), 'reason': reason, 'boundary_protected': entry == 0})
        target = min(.5, .2/max(m.x[t, -1], .1)) if on and not blocked else 0.
        if arm == 'alpha_half':
            target /= 2
        targets[t+delay:min(t+delay+stride, len(targets))] = target
        allowed[t+delay] = True
        previous = on
    return targets, allowed, actions


def preflight():
    count = verify_raw_sources()
    bars, frames, atr = dataset()
    feedback = feedback_catalog(bars, frames, atr)
    labels = json.loads(Path('artifacts/experiment_012/labels.json').read_text())
    counts = {tf: [sum(r['head'] == 'exit' and not r['left_censored'] and r['known'] is not None and utc(r['known']) < utc(date)-pd.Timedelta(days=7)
                       for r in labels[tf]) for date in MONTHS] for tf in FRAMES}
    assert min(n for values in counts.values() for n in values) >= 20
    checked = 0
    for tf in FRAMES:
        prior = {r['id']: r for r in labels[tf]}
        for symbol in SYMBOLS:
            ctx = context(symbol, bars[symbol], frames[tf][symbol], atr[tf][symbol], TRAIN_START, END)
            independent = reference_feedback(ctx, tf)
            expected = [r for r in feedback[tf] if r['symbol'] == symbol]
            equal(independent, expected)
            for r in expected:
                if r['x'] is not None:
                    old = prior[r['id']+'_exit']
                    equal(r['x'], old['x'], 2e-12)
                    assert r['known'] == old['known']
                    if r['known'] is not None:
                        equal(r['base']['base']-r['early']['base'], old['y'], 2e-10)
                checked += 4
            print(f'Independent base/early shadow payoffs at both costs:{tf} {symbol}', flush=True)
    write_json(Path('artifacts/study013_preflight.json'), {'raw_archives': count,
        'feedback_sha256': digest(feedback), 'independent_payoff_branches': checked,
        'training_counts_by_month': counts,
        'study012_exit_labels_reproduced': True,
        'module_hashes': {str(f): sha256(f.read_bytes()).hexdigest() for f in [Path('neattrade/adaptive_v13.py'), Path('neattrade/verify_v13.py')]}})


def run():
    protocol = json.loads((OUT/'protocol.json').read_text())
    protocol_hash = sha256((OUT/'protocol.json').read_bytes()).hexdigest()
    for f, h in {**protocol['source_hashes'], **protocol['evidence_hashes']}.items():
        assert sha256(Path(f).read_bytes()).hexdigest() == h, f
    verify_raw_sources()
    pre = json.loads(Path('artifacts/study013_preflight.json').read_text())
    labels = json.loads(Path('artifacts/experiment_012/labels.json').read_text())
    feedback = json.loads((OUT/'feedback.json').read_text())
    assert digest(feedback) == pre['feedback_sha256']
    report = json.loads((OUT/'report.json').read_text())
    bars, frames, atr = dataset()
    models, schedules = {}, {}
    for tf in FRAMES:
        refs = []
        for symbol in SYMBOLS:
            refs += reference_feedback(context(symbol, bars[symbol], frames[tf][symbol], atr[tf][symbol], TRAIN_START, END), tf)
        equal(refs, feedback[tf])
        for date in MONTHS:
            for method in ['expanding', 'weighted']:
                name = version_id(tf, method, date); path = OUT/'models'/f'{name}.pkl'
                meta = json.loads(path.with_suffix('.json').read_text())
                assert meta['protocol_hash'] == protocol_hash and meta['feedback_hash'] == pre['feedback_sha256']
                assert sha256(path.read_bytes()).hexdigest() == meta['model_hash']
                m = pickle.loads(path.read_bytes())
                refit = train_version(labels[tf], tf, method, date)
                cut = utc(date)-pd.Timedelta(days=7)
                rows = [r for r in labels[tf] if r['head'] == 'exit' and not r['left_censored'] and r['known'] is not None and utc(r['known']) < cut]
                assert m['training_ids'] == refit['training_ids'] == [r['id'] for r in rows]
                x = np.asarray([r['x'] for r in rows])
                weights = np.ones(len(rows)) if method == 'expanding' else np.asarray([2**(-(cut-utc(r['decision'])).total_seconds()/86400/180) for r in rows])
                weights *= len(weights)/weights.sum()
                np.testing.assert_allclose(m['weights'], weights, atol=2e-12, rtol=0)
                np.testing.assert_allclose(m['scaler'].mean_, np.average(x, weights=weights, axis=0), atol=2e-12, rtol=0)
                np.testing.assert_allclose(reference_predict(m, x), refit['training_predictions'], atol=2e-12, rtol=0)
                models[name] = m
            first, second = models[version_id(tf, 'expanding', date)], models[version_id(tf, 'weighted', date)]
            assert first['training_ids'] == second['training_ids']
            assert first['estimator'].get_params() == second['estimator'].get_params()
        history = reference_controller(refs, models, tf)
        equal(history, report['controller_history'][tf])
        state = Controller.load(OUT/f'{tf}_controller.json', tf, protocol_hash)
        equal(state.state['history'], history)
        schedules[tf] = {r['boundary']: r['active'] for r in history}
        print(f'Independent64 model refits and32 controller transitions:{tf}', flush=True)
    rows = {r['name']: r for r in report['strategies']}
    actions = json.loads((OUT/'actions.json').read_text()); trades = json.loads((OUT/'trades.json').read_text())
    action_index, trade_index = {}, {}
    for values, index in [(actions, action_index), (trades, trade_index)]:
        for v in values:
            index.setdefault((v['name'], v['scenario'], v['fold'], v['symbol']), []).append(v)
    all_daily, accounts, events_count, trades_count = {}, 0, 0, 0
    for tf in FRAMES:
        seq, records, endpoints = {}, {}, []
        for fold, (start, end) in enumerate(WINDOWS):
            contexts = {s: context(s, bars[s], frames[tf][s], atr[tf][s], start, end) for s in SYMBOLS}
            endpoints.append(contexts[SYMBOLS[0]]['market'].times[1:])
            for scenario, opts in SCENARIOS:
                baseline = {}
                for arm in ARMS:
                    name, results = tf+'_'+arm, {}
                    saved = rows[name]['folds'][scenario][fold]
                    for symbol, ctx in contexts.items():
                        m = ctx['market']
                        p, a, ev = reference_policy(ctx, tf, arm, models, schedules[tf], opts['delay'])
                        standard, standard_a, _ = policy(ctx, tf, arm, models, schedules[tf], opts['delay'])
                        np.testing.assert_allclose(p, standard, atol=2e-12, rtol=0)
                        np.testing.assert_array_equal(a, standard_a)
                        old = [{k: v for k, v in e.items() if k not in ['name', 'scenario', 'fold', 'symbol']} for e in action_index[name, scenario, fold, symbol]]
                        equal(ev, old, 2e-12); events_count += len(ev)
                        r = audit_simulate(m, p, a, opts['cost'])
                        if arm == 'alpha':
                            baseline[symbol] = r
                        for k in ['return', 'max_drawdown', 'sharpe', 'cost_paid_initial_equity', 'funding_paid_initial_equity', 'order_count', 'turnover', 'mean_abs_exposure']:
                            equal(r[k], saved['by_symbol'][symbol][k])
                        ts = []
                        for i, (entry, exit_decision) in enumerate(spans(ctx, tf)):
                            first, end_i = entry+opts['delay'], min(exit_decision+opts['delay'], len(m.x))
                            finish = min(end_i+1, len(m.x))
                            b = float(np.log(baseline[symbol]['equity'][finish]/baseline[symbol]['equity'][first]))
                            c = float(np.log(r['equity'][finish]/r['equity'][first]))
                            ts.append({'episode': i, 'start': str(m.times[first]), 'end': str(m.times[end_i]),
                                'left_censored': entry == 0, 'right_censored': exit_decision == len(m.x),
                                'baseline_log': b, 'candidate_log': c, 'delta_log': c-b, 'baseline_loss': b < 0})
                        equal(ts, [{k: v for k, v in t.items() if k not in ['name', 'scenario', 'fold', 'symbol']} for t in trade_index[name, scenario, fold, symbol]])
                        trades_count += len(ts); results[symbol] = r; accounts += 1
                    port = portfolio(results)
                    for k in ['return', 'max_drawdown', 'sharpe']:
                        equal(port[k], saved['portfolio'][k])
                    seq.setdefault((name, scenario), []).append(port['returns'])
                    records.setdefault((name, scenario), []).append({'by_symbol': results, 'observations': len(port['returns'])})
            print(f'Independent actions and accounts:{tf} window{fold} ({accounts}/504)', flush=True)
        for (name, sc), parts in seq.items():
            if sc == 'base':
                all_daily[name] = [daily_returns(r, t).to_numpy() for r, t in zip(parts[:4], endpoints[:4])]
        for group, sl in [('development', slice(0, 4)), ('recent', slice(4, 6))]:
            dates = pd.DatetimeIndex(np.concatenate([v.to_numpy() for v in endpoints[sl]]))
            with np.load(OUT/f'{tf}_{group}_equity.npz') as curves:
                np.testing.assert_array_equal(curves['timestamps'][1:], dates.as_unit('ns').asi8)
                for (name, sc), parts in seq.items():
                    equal(grouped(parts[sl], records[name, sc][sl], dates), rows[name][group][sc])
                    np.testing.assert_allclose(curves[name+'__'+sc], metrics(np.concatenate(parts[sl]))['equity'], atol=2e-9, rtol=0)
                    use = [v for v in trades if v['name'] == name and v['scenario'] == sc and v['fold'] in range(sl.start, sl.stop)]
                    equal(reference_repair(use), rows[name][group+'_repair'][sc]['all'])
                    equal(reference_repair([v for v in use if not v['left_censored'] and not v['right_censored']]), rows[name][group+'_repair'][sc]['natural_only'])
    assert [(v['candidate'], v['reference']) for v in report['comparisons']] == contrasts()
    for c in report['comparisons']:
        equal(interval(all_daily[c['candidate']], all_daily[c['reference']], direct=True), {k: c[k] for k in ['log_wealth_excess', 'intervals']})
    old = {r['name']: r for r in json.loads(Path('artifacts/experiment_012/report.json').read_text())['strategies']}
    for tf in FRAMES:
        for new, prior in [('alpha', 'alpha'), ('semiannual', 'boost_exit')]:
            for group in ['development', 'recent']:
                for sc in ['base', 'double_cost', 'delay30m']:
                    equal(rows[tf+'_'+new][group][sc], old[tf+'_'+prior][group][sc])
    tests = subprocess.run([sys.executable, '-m', 'pytest', '-q'], text=True, capture_output=True, check=True)
    (OUT/'pytest-output.txt').write_text(tests.stdout, encoding='utf-8')
    result = {'tests_passed': int(re.search(r'(\d+) passed', tests.stdout)[1]), 'models_refitted': len(models),
        'controller_updates_independently_replayed': 64, 'independent_payoff_branches': pre['independent_payoff_branches'],
        'asset_windows': accounts, 'action_events': events_count, 'episode_attributions': trades_count,
        'intervals': INTERVAL_COUNT, 'study012_alpha_and_semiannual_reproduced': True,
        'curves_annual_asset_returns_and_repair_verified': True, 'frozen_hashes_verified': True, 'production_selection': None}
    assert accounts == 504 and len(models) == 128
    write_json(OUT/'verification.json', result); print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    preflight() if '--preflight' in sys.argv else run()
