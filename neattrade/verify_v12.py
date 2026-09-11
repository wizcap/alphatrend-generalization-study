"""Independent supervised-label, policy, accounting and inference replay."""
from hashlib import sha256
from copy import deepcopy
import json
from pathlib import Path
import pickle
import re
import subprocess
import sys

import numpy as np
import pandas as pd

from .fast_accounting import audit_simulate
from .indicators_v11 import COLUMNS
from .market import simulate
from .repair_v12 import (SYMBOLS, FRAMES, KINDS, MODES, CONTROLS, ENTRY_FEATURES, EXIT_FEATURES,
    TRAIN_START, END, dataset, context, fit_model, predict, policy, signals, cut)
from .research_v3 import FOLDS
from .research_v6 import write_json
from .research_v11 import verify_raw_sources, grouped
from .research_v12 import (OUT, WINDOWS, SCENARIOS, INTERVAL_COUNT, DRAWS, catalog,
    candidate_models, contrasts)
from .timeframes import TIMEFRAMES, portfolio, metrics, daily_returns


def reference_state(ctx, entry, t, delay):
    m = ctx['market']
    a = entry+delay
    initial, atr = m.prices[a], ctx['atr'][entry]
    last = ctx['closes'][t-1]
    peak = max([initial]+list(m.highs[a:t]))
    trough = min([initial]+list(m.lows[a:t]))
    state = [(last-initial)/atr, (peak-initial)/atr, (trough-initial)/atr,
             (peak-last)/atr, sum(m.funding[a+1:t])*1000]
    return np.r_[m.x[t], [min(5., max(-5., v)) for v in state]]


def reference_predict(model, x):
    x = np.atleast_2d(x)
    if model['kind'] == 'mean':
        value = np.full(len(x), model['mean'])
    else:
        scale = model['scaler']
        z = np.column_stack([(x[:, j]-scale.mean_[j])/scale.scale_[j] for j in range(x.shape[1])])
        est = model['estimator']
        if model['kind'] == 'ridge':
            value = np.full(len(x), float(est.intercept_))
            for j in range(x.shape[1]):
                value += z[:, j]*est.coef_[j]
        else:
            value = np.full(len(x), float(est._baseline_prediction[0, 0]))
            for trees in est._predictors:
                assert len(trees) == 1
                nodes = trees[0].nodes
                assert not np.any(nodes['is_categorical'])
                for i, row in enumerate(z):
                    node = nodes[0]
                    while not node['is_leaf']:
                        node = nodes[node['left'] if row[node['feature_idx']] <= node['num_threshold'] else node['right']]
                    value[i] += node['value']
    np.testing.assert_allclose(value, predict(model, x), atol=2e-12, rtol=0)
    return value


def reference_policy(ctx, tf, candidate, delay):
    m, stride = ctx['market'], TIMEFRAMES[tf]//15
    p, allowed, actions = np.zeros(len(m.x)), np.zeros(len(m.x), bool), []
    was_on, stopped, entry, episode = False, False, -1, -1
    for t in range(0, len(m.x), stride):
        on = m.x[t, 0] > 0
        head = None
        if on and not was_on:
            entry, stopped, episode, head = t, False, episode+1, 'entry'
        elif on and t-entry == 96:
            head = 'exit'
        elif not on:
            stopped = False
        reason, forecast = '', None
        if head:
            x = m.x[t] if head == 'entry' else reference_state(ctx, entry, t, delay)
            if isinstance(candidate, dict) and candidate['mode'] in [head, 'both'] and entry == 0:
                reason = 'boundary_keep'
            elif isinstance(candidate, dict) and candidate['mode'] in [head, 'both'] and not stopped:
                forecast = float(reference_predict(candidate[head], x)[0])
                if forecast < 0:
                    stopped, reason = True, 'veto' if head == 'entry' else 'early_exit'
            elif isinstance(candidate, str) and head == 'exit' and not stopped:
                if candidate == 'exit24h' or (candidate == 'loss1atr24h' and x[13] < -1):
                    stopped, reason = True, 'rule_exit'
            actions.append({'episode': episode, 'head': head, 'decision': t, 'time': str(m.times[t]),
                'execution': str(m.times[min(t+delay, len(m.x))]), 'prediction': forecast,
                'blocked': stopped, 'reason': reason,
                'boundary_protected': bool(isinstance(candidate, dict) and entry == 0)})
        target = min(.5, .20/max(m.x[t, -1], .10)) if on and not stopped else 0.
        if candidate == 'cash':
            target = 0.
        if candidate == 'alpha_half':
            target *= .5
        if t+delay < len(p):
            p[t+delay:min(t+delay+stride, len(p))] = target
            allowed[t+delay] = True
        was_on = on
    return p, allowed, actions


def spans(ctx, tf):
    m, step = ctx['market'], TIMEFRAMES[tf]//15
    ix = np.arange(0, len(m.x), step)
    on = m.x[ix, 0] > 0
    starts = ix[on & ~np.r_[False, on[:-1]]]
    ends = ix[~on & np.r_[False, on[:-1]]]
    return [(int(a), int(next((v for v in ends if v > a), len(m.x)))) for a in starts]


def reference_labels(ctx, tf):
    m = ctx['market']
    p, a, _ = reference_policy(ctx, tf, 'alpha', 1)
    # Only units from the legacy trace are used; its old feature label is irrelevant.
    full = simulate(m, p, trade_allowed=a, rebalance_band=.05, trace=True)
    rows = []
    for e, z_decision in spans(ctx, tf):
        start = e+1
        resolved = z_decision < len(m.x) and z_decision+1 <= len(m.x)
        end = z_decision+1 if resolved else len(m.x)
        initial = full['equity'][start]
        terminal = full['equity'][end+1] if resolved and end < len(m.x) else full['equity'][end]
        y = terminal/initial-1
        known = str(m.times[end]) if resolved else None
        common = {'known': known, 'left_censored': e == 0}
        rows.append({'id': f'{m.symbol}_{m.times[e].isoformat()}_entry', 'head': 'entry',
            'decision': str(m.times[e]), 'x': m.x[e].tolist(), 'y': float(y) if resolved else None, **common})
        c = e+96
        if c < z_decision and c < len(m.x):
            execution = c+1
            units = full['ledger'][execution-1]['units']
            early = full['equity'][execution]-abs(units)*m.prices[execution]*.0007
            rows.append({'id': f'{m.symbol}_{m.times[e].isoformat()}_exit', 'head': 'exit',
                'decision': str(m.times[c]), 'x': reference_state(ctx, e, c, 1).tolist(),
                'y': float((terminal-early)/initial) if resolved else None, **common})
    return rows


def compare_labels(a, b):
    assert len(a) == len(b)
    lookup = {r['id']: r for r in b}
    for row in a:
        ref = lookup[row['id']]
        for key in ['head', 'decision', 'known', 'left_censored']:
            assert row[key] == ref[key], (row['id'], key)
        np.testing.assert_allclose(row['x'], ref['x'], atol=2e-12, rtol=0)
        if row['y'] is None:
            assert ref['y'] is None
        else:
            np.testing.assert_allclose(row['y'], ref['y'], atol=2e-10, rtol=0)


def reference_repair(records):
    baseline = np.asarray([r['baseline_log'] for r in records])
    candidate = np.asarray([r['candidate_log'] for r in records])
    losing = baseline < 0
    delta = candidate-baseline
    loss, win = -baseline[losing].sum(), baseline[~losing].sum()
    return {'episodes': len(records), 'baseline_losers': int(losing.sum()),
        'baseline_loss_log': loss, 'baseline_win_log': win,
        'loss_saved_log': delta[losing].sum(), 'winner_change_log': delta[~losing].sum(),
        'total_delta_log': delta.sum(),
        'loss_repair_fraction': delta[losing].sum()/loss if loss > 0 else None,
        'winner_retention': candidate[~losing].sum()/win if win > 0 else None,
        'losers_improved': int(((delta > 1e-12) & losing).sum()),
        'winners_harmed': int(((delta < -1e-12) & ~losing).sum())}


def preflight():
    raw = verify_raw_sources()
    bars, frames, atr = dataset()
    labels, diag = catalog(bars, frames, atr)
    count, cases = 0, 0
    for tf in FRAMES:
        for s in SYMBOLS:
            ctx = context(s, bars[s], frames[tf][s], atr[tf][s], TRAIN_START, END)
            independent = reference_labels(ctx, tf)
            compare_labels([v for v in labels[tf] if v['id'].startswith(s+'_')], independent)
            count += len(independent)
            _, events = signals(ctx, tf)
            selected = [v for v in events if v['head'] == 'exit']
            for event in [selected[0], selected[len(selected)//2], selected[-1]]:
                t = event['decision']
                short = {'market': cut(ctx['market'], 0, t+1), 'atr': ctx['atr'][:t+1], 'closes': ctx['closes'][:t+1]}
                before = [v for v in events if v['decision'] <= t]
                after = signals(short, tf)[1]
                assert len(before) == len(after)
                for u, v in zip(before, after):
                    assert u['head'] == v['head'] and u['decision'] == v['decision']
                    np.testing.assert_array_equal(u['x'], v['x'])
                future = deepcopy(ctx)
                for key in ['prices', 'marks', 'highs', 'lows', 'funding']:
                    getattr(future['market'], key)[t:] *= 1.5
                future['closes'][t:] *= 1.5
                future['atr'][t:] *= 1.5
                future['market'].x[t+1:, 0] = 0
                altered = [v for v in signals(future, tf)[1] if v['decision'] <= t]
                assert len(altered) == len(before)
                for u, v in zip(before, altered):
                    np.testing.assert_array_equal(u['x'], v['x'])
                cases += 1
            print(f'Independent labels and causal event prefixes: {tf} {s}', flush=True)
    result = {'raw_archives': raw, 'independent_label_rows': count, 'causal_prefix_cases': cases, 'future_mutation_cases': cases,
        'labels_sha256': sha256(json.dumps(labels, sort_keys=True).encode()).hexdigest(),
        'module_hashes': {str(p): sha256(p.read_bytes()).hexdigest() for p in [Path('neattrade/repair_v12.py'), Path('neattrade/verify_v12.py')]},
        'first_fold_counts': {tf: {h: sum(v['head'] == h and not v['left_censored'] and v['known'] is not None and pd.Timestamp(v['known']) < pd.Timestamp('2023-12-25', tz='UTC') for v in labels[tf]) for h in ['entry', 'exit']} for tf in FRAMES}}
    write_json(Path('artifacts/study012_preflight.json'), result)


def direct_intervals(parts, refs):
    differences = [np.log1p(a)-np.log1p(b) for a, b in zip(parts, refs)]
    output = {}
    for block in [7, 28]:
        rng = np.random.default_rng(1012)
        sampled = np.zeros(DRAWS)
        for d in differences:
            for first in range(0, DRAWS, 2000):
                start = rng.integers(0, len(d), (2000, int(np.ceil(len(d)/block))))
                ix = ((start[:, :, None]+np.arange(block)) % len(d)).reshape(2000, -1)[:, :len(d)]
                sampled[first:first+2000] += d[ix].sum(axis=1)
        output[str(block)] = np.quantile(sampled, [.025/INTERVAL_COUNT, 1-.025/INTERVAL_COUNT]).tolist()
    return output


def run():
    protocol = json.loads((OUT/'protocol.json').read_text())
    fingerprint = sha256((OUT/'protocol.json').read_bytes()).hexdigest()
    for p, h in {**protocol['source_hashes'], **protocol['evidence_hashes']}.items():
        assert sha256(Path(p).read_bytes()).hexdigest() == h, p
    raw = verify_raw_sources()
    pre = json.loads(Path('artifacts/study012_preflight.json').read_text())
    for p, h in pre['module_hashes'].items():
        assert sha256(Path(p).read_bytes()).hexdigest() == h
    bars, frames, atr = dataset()
    labels, _ = catalog(bars, frames, atr)
    assert sha256(json.dumps(labels, sort_keys=True).encode()).hexdigest() == pre['labels_sha256']
    assert json.loads((OUT/'labels.json').read_text()) == labels
    models = {}
    for tf in FRAMES:
        refs = []
        for s in SYMBOLS:
            ctx = context(s, bars[s], frames[tf][s], atr[tf][s], TRAIN_START, END)
            refs += reference_labels(ctx, tf)
        compare_labels(labels[tf], refs)
        for fold, (start, _) in enumerate(WINDOWS[:5]):
            cutoff = str((pd.Timestamp(start)-pd.Timedelta(days=7)).date())
            for kind in KINDS:
                for head in ['entry', 'exit']:
                    name = f'{tf}_fold{fold}_{kind}_{head}'
                    path = OUT/'models'/f'{name}.pkl'
                    meta = json.loads(path.with_suffix('.json').read_text())
                    assert meta['protocol_sha256'] == fingerprint and meta['labels_sha256'] == pre['labels_sha256']
                    assert sha256(path.read_bytes()).hexdigest() == meta['model_sha256']
                    fitted = pickle.loads(path.read_bytes())
                    independent = fit_model(refs, head, kind, cutoff)
                    assert fitted['training_ids'] == independent['training_ids']
                    train = [next(r for r in refs if r['id'] == identity) for identity in fitted['training_ids']]
                    x, y = np.asarray([r['x'] for r in train]), np.asarray([r['y'] for r in train])
                    reference_predict(fitted, x)
                    # Tiny economically equivalent label roundoff can change tree split ties.
                    # Refit exact stored labels to verify determinism as a separate check.
                    exact = fit_model(labels[tf], head, kind, cutoff)
                    np.testing.assert_allclose(predict(exact, x), fitted['training_predictions'], atol=2e-12, rtol=0)
                    if kind == 'ridge':
                        z = (x-x.mean(axis=0))/np.where(x.std(axis=0) > 0, x.std(axis=0), 1.)
                        centered = z-z.mean(axis=0)
                        coef = np.linalg.solve(centered.T@centered+10*np.eye(x.shape[1]), centered.T@(y-y.mean()))
                        pred = centered@coef+y.mean()
                        np.testing.assert_allclose(pred, reference_predict(fitted, x), atol=2e-10, rtol=0)
                    models[name] = fitted
            print(f'Independent labels, model refits and predictions: {tf} fold{fold}', flush=True)
    report = json.loads((OUT/'report.json').read_text())
    rows = {r['name']: r for r in report['strategies']}
    actions = json.loads((OUT/'actions.json').read_text())
    trades = json.loads((OUT/'trades.json').read_text())
    action_lookup = {}
    trade_lookup = {}
    for v in actions:
        action_lookup.setdefault((v['name'], v['scenario'], v['fold'], v['symbol']), []).append(v)
    for v in trades:
        trade_lookup.setdefault((v['name'], v['scenario'], v['fold'], v['symbol']), []).append(v)
    all_daily, accounting, events_checked, trades_checked = {}, 0, 0, 0
    for tf in FRAMES:
        seq, rec, endpoints = {}, {}, []
        for fold, (start, end) in enumerate(WINDOWS):
            ctxs = {s: context(s, bars[s], frames[tf][s], atr[tf][s], start, end) for s in SYMBOLS}
            endpoints.append(ctxs[SYMBOLS[0]]['market'].times[1:])
            for scenario, opts in SCENARIOS:
                bases = {}
                for s, ctx in ctxs.items():
                    p, a, _ = reference_policy(ctx, tf, 'alpha', opts['delay'])
                    bases[s] = audit_simulate(ctx['market'], p, a, opts['cost'])
                for arm, candidate in candidate_models(models, tf, fold).items():
                    name, results = tf+'_'+arm, {}
                    saved = rows[name]['folds'][scenario][fold]
                    for s, ctx in ctxs.items():
                        m = ctx['market']
                        p, a, es = reference_policy(ctx, tf, candidate, opts['delay'])
                        standard, standard_a, _ = policy(ctx, tf, candidate, opts['delay'])
                        np.testing.assert_allclose(p, standard, atol=2e-12, rtol=0)
                        np.testing.assert_array_equal(a, standard_a)
                        recorded = action_lookup.get((name, scenario, fold, s), [])
                        assert len(es) == len(recorded)
                        for ev, old in zip(es, recorded):
                            for key, value in ev.items():
                                if key == 'prediction' and value is not None:
                                    np.testing.assert_allclose(value, old[key], atol=2e-12, rtol=0)
                                else:
                                    assert value == old[key], (name, key)
                            events_checked += 1
                        r = audit_simulate(m, p, a, opts['cost'])
                        for key in ['return', 'max_drawdown', 'sharpe', 'cost_paid_initial_equity',
                                    'funding_paid_initial_equity', 'turnover', 'order_count', 'mean_abs_exposure']:
                            np.testing.assert_allclose(r[key], saved['by_symbol'][s][key], atol=2e-9, rtol=0)
                        old_trades = trade_lookup[name, scenario, fold, s]
                        expected_spans = spans(ctx, tf)
                        assert len(old_trades) == len(expected_spans)
                        total = 0.
                        for saved_trade, (e, z) in zip(old_trades, expected_spans):
                            a_idx = e+opts['delay']
                            end_idx = min(z+opts['delay'], len(m.x))
                            closed_idx = min(end_idx+1, len(m.x))
                            b_log = np.log(bases[s]['equity'][closed_idx])-np.log(bases[s]['equity'][a_idx])
                            c_log = np.log(r['equity'][closed_idx])-np.log(r['equity'][a_idx])
                            assert saved_trade['baseline_loss'] == bool(b_log < 0)
                            assert saved_trade['left_censored'] == (e == 0)
                            assert saved_trade['right_censored'] == (z == len(m.x) or z+opts['delay'] > len(m.x))
                            assert saved_trade['start'] == str(m.times[a_idx])
                            assert saved_trade['end'] == str(m.times[end_idx])
                            for key, value in [('baseline_log', b_log), ('candidate_log', c_log), ('delta_log', c_log-b_log)]:
                                np.testing.assert_allclose(value, saved_trade[key], atol=2e-10, rtol=0)
                            total += c_log-b_log
                            trades_checked += 1
                        np.testing.assert_allclose(total, np.log1p(r['return'])-np.log1p(bases[s]['return']), atol=2e-10)
                        results[s] = r
                        accounting += 1
                    port = portfolio(results)
                    for key in ['return', 'max_drawdown', 'sharpe']:
                        np.testing.assert_allclose(port[key], saved['portfolio'][key], atol=2e-9, rtol=0)
                    seq.setdefault((name, scenario), []).append(port['returns'])
                    rec.setdefault((name, scenario), []).append({'by_symbol': results, 'observations': len(port['returns'])})
            print(f'Independent actions, accounting, attribution: {tf} window{fold}, {accounting}/1008 asset windows', flush=True)
        for (name, sc), parts in seq.items():
            if sc == 'base':
                all_daily[name] = [daily_returns(r, t).to_numpy() for r, t in zip(parts[:4], endpoints[:4])]
        for group, sl in [('development', slice(0, 4)), ('recent', slice(4, 6))]:
            dates = pd.DatetimeIndex(np.concatenate([v.to_numpy() for v in endpoints[sl]]))
            with np.load(OUT/f'{tf}_{group}_equity.npz') as curves:
                np.testing.assert_array_equal(curves['timestamps'][1:], dates.as_unit('ns').asi8)
                for (name, sc), parts in seq.items():
                    got = grouped(parts[sl], rec[name, sc][sl], dates)
                    saved = rows[name][group][sc]
                    for key in ['return', 'max_drawdown', 'sharpe', 'daily_sharpe', 'orders', 'mean_abs_exposure']:
                        np.testing.assert_allclose(got[key], saved[key], atol=2e-9, rtol=0)
                    for key in ['annual_returns', 'by_symbol_return']:
                        for k, v in got[key].items():
                            np.testing.assert_allclose(v, saved[key][k], atol=2e-9, rtol=0)
                    np.testing.assert_allclose(curves[name+'__'+sc], metrics(np.concatenate(parts[sl]))['equity'], atol=2e-9, rtol=0)
                    ts = [v for v in trades if v['name'] == name and v['scenario'] == sc and v['fold'] in range(sl.start, sl.stop)]
                    for sample, use in [('all', ts), ('natural_only', [v for v in ts if not v['left_censored'] and not v['right_censored']])]:
                        recomputed = reference_repair(use)
                        for k, v in recomputed.items():
                            expected = rows[name][group+'_repair'][sc][sample][k]
                            if v is None:
                                assert expected is None
                            else:
                                np.testing.assert_allclose(v, expected, atol=2e-10)
    assert accounting == 1008 and len(models) == 60
    assert [(c['candidate'], c['reference']) for c in report['comparisons']] == contrasts()
    for i, c in enumerate(report['comparisons']):
        a, b = all_daily[c['candidate']], all_daily[c['reference']]
        value = sum((np.log1p(x)-np.log1p(y)).sum() for x, y in zip(a, b))
        np.testing.assert_allclose(value, c['log_wealth_excess'], atol=2e-10)
        independent = direct_intervals(a, b)
        for block in ['7', '28']:
            np.testing.assert_allclose(independent[block], c['intervals'][block], atol=2e-10)
        if i % 8 == 0:
            print(f'Independent intervals: {2*(i+1)}/64', flush=True)
    assert len(report['gates']) == 12
    for gate in report['gates']:
        name = gate['name']
        baseline = rows[name.split('_')[0]+'_alpha']
        candidate = rows[name]
        repair = candidate['development_repair']['base']['all']
        contrast = next(c for c in report['comparisons'] if c['candidate'] == name and c['reference'] == baseline['name'])
        conditions = [candidate['development']['base']['return']-baseline['development']['base']['return'] > 0,
            baseline['development']['base']['max_drawdown']-candidate['development']['base']['max_drawdown'] >= 0,
            candidate['development']['double_cost']['return']-baseline['development']['double_cost']['return'] >= 0,
            candidate['recent']['base']['return']-baseline['recent']['base']['return'] >= 0,
            repair['loss_repair_fraction'] is not None and repair['loss_repair_fraction'] >= .20,
            repair['winner_retention'] is not None and repair['winner_retention'] >= .90,
            min(v[0] for v in contrast['intervals'].values()) > 0]
        keys = ['net_improvement', 'drawdown_no_worse', 'double_cost_no_worse', 'recent_no_worse',
                'repair_20pct', 'retain_90pct', 'both_intervals_positive']
        assert dict(zip(keys, conditions)) == gate['checks']
        assert all(conditions) == gate['all_pass']
    old = {r['name']: r for r in json.loads(Path('artifacts/experiment_011/report.json').read_text())['strategies']}
    for tf in FRAMES:
        for group in ['development', 'recent']:
            for key in ['return', 'max_drawdown']:
                np.testing.assert_allclose(rows[tf+'_alpha'][group]['base'][key], old[tf+'_alpha'][group]['delay15m'][key], atol=2e-9, rtol=0)
    tests = subprocess.run([sys.executable, '-m', 'pytest', '-q'], text=True, capture_output=True, check=True)
    (OUT/'pytest-output.txt').write_text(tests.stdout, encoding='utf-8')
    result = {'tests_passed': int(re.search(r'(\d+) passed', tests.stdout)[1]), 'models_refitted': len(models),
        'independent_labels': pre['independent_label_rows'], 'raw_archives': raw,
        'independent_asset_windows': accounting, 'independent_action_events': events_checked,
        'independent_episode_attributions': trades_checked, 'independent_intervals': INTERVAL_COUNT,
        'independent_repair_aggregates_and_gates_verified': True,
        'curves_annual_and_asset_returns_verified': True, 'study011_delayed_alpha_reproduced': True,
        'frozen_hashes_verified': True, 'selected': None}
    write_json(OUT/'verification.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    preflight() if '--preflight' in sys.argv else run()
