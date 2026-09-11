"""Separate fit arithmetic, policy clock, tree/Ridge and full account reconciliation."""
from hashlib import sha256
import json
from pathlib import Path
import pickle
import re
import subprocess
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .generalize_v15 import FRAMES, SYMBOLS, MONTHS, ARMS, policy, utc
from .repair_v12 import dataset, context
from .verify_v12 import reference_state, reference_predict, spans, reference_repair
from .verify_v13 import equal
from .fast_accounting import audit_simulate
from .research_v15 import OUT, WINDOWS, SCENARIOS, contrasts, freeze
from .research_v11 import verify_raw_sources, grouped
from .research_v6 import write_json
from .timeframes import TIMEFRAMES, metrics, portfolio, daily_returns


def audit_model(m, data):
    date, family, heldout, days = utc(m['available']), m['family'], m['heldout'], m['days']
    cutoff = date-pd.Timedelta(days=7)
    selected = []
    for r in data:
        if r['head']!='exit' or r['left_censored'] or r['known'] is None:
            continue
        if utc(r['known'])>=cutoff or r['id'].split('_')[0]==heldout:
            continue
        if days is not None and (cutoff-utc(r['decision'])).total_seconds()>days*86400:
            continue
        selected.append(r)
    assert m['training_ids']==[r['id'] for r in selected]
    n = len(selected); assert m['training_count']==n and m['fitted']==(n>=20)
    assert m['cutoff']==str(cutoff)
    assert m['last_label_known']==max((r['known'] for r in selected), default=None)
    indices = list(range(18)) if family=='current' else [1, 2, 4, 9, 11, 12, 13, 16]
    assert m['input_indices']==indices
    groups = [r['id'].split('_')[0]+'_'+utc(r['decision']).strftime('%Y')+('H1' if utc(r['decision']).month<=6 else 'H2') for r in selected]
    counts = {g: groups.count(g) for g in set(groups)}
    assert m['groups']==groups and m['group_counts']==counts
    robust = family in ['robust', 'ridge']
    w = np.array([1/counts[g]**.5 if robust else 1. for g in groups])
    if n: w *= n/sum(w)
    y = np.array([r['y'] for r in selected]); ordered = sorted(y)
    limits = None
    if robust and n:
        limits = []
        for q in [.025, .975]:
            j = (n-1)*q; lo, hi = int(np.floor(j)), int(np.ceil(j))
            limits.append(ordered[lo]+(ordered[hi]-ordered[lo])*(j-lo))
    target = np.array([min(limits[1], max(limits[0], v)) if limits else v for v in y])
    equal(m['clip_limits'], limits, 2e-12)
    for got, expected in [(m['weights'], w), (m['raw_y'], y), (m['fit_y'], target)]:
        np.testing.assert_allclose(got, expected, atol=2e-12, rtol=0)
    equal(m['effective_weight_count'], sum(w)**2/sum(w*w) if n else 0., 2e-10)
    if n<20:
        assert m['kind']=='mean' and m['mean']==0. and 'scaler' not in m
        return False
    x = np.array([r['x'] for r in selected])[:, indices]
    mu = np.average(x, weights=w, axis=0); var = np.average((x-mu)**2, weights=w, axis=0)
    np.testing.assert_allclose(m['scaler'].mean_, mu, atol=2e-12, rtol=0)
    np.testing.assert_allclose(m['scaler'].var_, var, atol=2e-12, rtol=0)
    np.testing.assert_allclose(m['scaler'].scale_, np.where(var<1e-24, 1., np.sqrt(var)), atol=2e-12, rtol=0)
    # Independent preprocessing construction and estimator refit, not the study train() helper.
    scaler = StandardScaler().fit(x, sample_weight=w if robust else None)
    z = scaler.transform(x)
    if family=='ridge':
        params = dict(alpha=50., solver='cholesky')
        estimator = Ridge(**params)
        zw, yw = np.average(z, weights=w, axis=0), np.average(target, weights=w)
        centered = z-zw
        coef = np.linalg.solve(centered.T@(w[:, None]*centered)+50*np.eye(8), centered.T@(w*(target-yw)))
        np.testing.assert_allclose(coef, m['estimator'].coef_, atol=2e-12, rtol=0)
        equal(yw-zw@coef, m['estimator'].intercept_, 2e-12)
    else:
        params = dict(loss='squared_error', learning_rate=.05 if family=='current' else .03,
            max_iter=100, max_leaf_nodes=7 if family=='current' else 3,
            min_samples_leaf=20, l2_regularization=10. if family=='current' else 50.,
            early_stopping=False, random_state=11)
        if family!='current': params['max_depth']=2
        estimator = HistGradientBoostingRegressor(**params)
    assert all(m['estimator'].get_params()[k]==v for k, v in params.items())
    with threadpool_limits(limits=1): estimator.fit(z, target, sample_weight=w)
    ref = {**m, 'scaler': scaler, 'estimator': estimator}
    np.testing.assert_allclose(reference_predict(m, x), reference_predict(ref, x), atol=2e-12, rtol=0)
    np.testing.assert_allclose(reference_predict(m, x), m['training_predictions'], atol=2e-12, rtol=0)
    return True


def reference_policy(ctx, tf, arm, models, delay):
    m, stride = ctx['market'], TIMEFRAMES[tf]//15
    targets, allowed, actions = np.zeros(len(m.x)), np.zeros(len(m.x), bool), []
    previous, blocked, entry, episode, versions = False, False, -1, -1, []
    for t in range(0, len(m.x), stride):
        on = m.x[t, 0]>0
        head = None
        if on and not previous: entry, episode, blocked, head = t, episode+1, False, 'entry'
        elif on and t-entry==96: head='exit'
        elif not on: blocked=False
        if head:
            score, values, reason = None, [], ''
            if head=='entry':
                stamp = m.times[t].strftime('%Y%m')
                if entry==0 or arm=='alpha': versions=[]
                elif arm=='current': versions=[f'{tf}_expanding_{stamp}']
                else:
                    family = ('robust' if arm.startswith('temporal_') else
                        arm.split('_')[1] if arm.startswith('transfer_') else arm[:-1])
                    source = m.symbol if arm.startswith('transfer_') else 'pooled'
                    windows = ['all', '730', '365'] if arm.startswith('temporal_') else ['all']
                    versions = [f'{tf}_{family}_{w}_{source}_{stamp}' for w in windows]
                assert all(utc(models[v]['available'])<=m.times[t] for v in versions)
                reason='boundary_keep' if entry==0 else 'latch'
            elif versions:
                if any(models[v].get('fitted') is False for v in versions): score, reason=0., 'insufficient'
                else:
                    x = reference_state(ctx, entry, t, delay)
                    for v in versions:
                        indices = list(range(18)) if '_expanding_' in v or '_current_' in v else [1, 2, 4, 9, 11, 12, 13, 16]
                        values.append(float(reference_predict(models[v], x[indices])[0]))
                    score = max(values) if arm=='temporal_guard' else sum(values)/len(values)
                    blocked = all(v<0 for v in values) if arm=='temporal_guard' else score<0
                    reason = 'early_exit' if blocked else 'disagreement' if arm=='temporal_guard' and any(v<0 for v in values) else 'keep'
            actions.append(dict(episode=episode, head=head, decision=t, entry=str(m.times[entry]),
                time=str(m.times[t]), execution=str(m.times[t+delay]), versions=list(versions),
                prediction=score, component_predictions=values, blocked=bool(blocked), reason=reason, boundary_protected=entry==0))
        target = min(.5, .2/max(m.x[t, -1], .1)) if on and not blocked else 0.
        targets[t+delay:min(t+delay+stride, len(targets))] = target
        allowed[t+delay] = True; previous=on
    return targets, allowed, actions


def direct_interval(parts, refs):
    differences = [np.log1p(x)-np.log1p(y) for x, y in zip(parts, refs)]
    result = {}
    for block in [7, 28]:
        rng, values = np.random.default_rng(1015), np.zeros(100000)
        for d in differences:
            count = (len(d)+block-1)//block
            for offset in range(0, 100000, 2000):
                starts = rng.integers(0, len(d), size=(2000, count))
                index = ((starts[:, :, None]+np.arange(block))%len(d)).reshape(2000, -1)[:, :len(d)]
                values[offset:offset+2000] += d[index].sum(axis=1)
        result[str(block)] = np.quantile(values, [.025/40, 1-.025/40]).tolist()
    return dict(log_wealth_excess=float(sum(sum(d) for d in differences)), intervals=result)


def check_diagnostics(actions, labels, saved):
    wanted = [a for a in actions if a['scenario']=='base' and a['head']=='exit']
    assert len(wanted)==len(saved)
    for a, d in zip(wanted, saved):
        assert all(d[k]==v for k, v in a.items())
        matches = [r for r in labels[a['name'].split('_')[0]] if r['head']=='exit' and
                   r['id']==f'{a["symbol"]}_{utc(a["entry"]).isoformat()}_exit']
        r = matches[0] if matches else None
        status = 'boundary' if a['boundary_protected'] else 'unmatched' if r is None else 'censored' if (
            r['left_censored'] or r['known'] is None or utc(r['known'])>utc(WINDOWS[a['fold']][1])) else 'resolved'
        assert d['status']==status
        y = r['y'] if status=='resolved' else None
        p = sum(a['component_predictions'])/len(a['component_predictions']) if a['component_predictions'] else None
        equal(d['raw_y'], y); equal(d['payoff_prediction'], p)
        equal(d['absolute_error'], abs(p-y) if p is not None and y is not None else None)
        equal(d['regret'], max(y if a['blocked'] else -y, 0.) if y is not None else None)
        equal(d['alpha_regret'], max(-y, 0.) if y is not None else None)
    return len(saved)


def run():
    raw = verify_raw_sources(); protocol_hash = freeze()
    report = json.loads((OUT/'report.json').read_text())
    labels = json.loads(Path('artifacts/experiment_012/labels.json').read_text())
    bars, frames, atr = dataset()
    models, fitted = {}, 0
    for path in sorted((OUT/'models').glob('*.pkl')):
        meta = json.loads(path.with_suffix('.json').read_text())
        assert meta['protocol_hash']==protocol_hash and meta['model_hash']==sha256(path.read_bytes()).hexdigest()
        m = pickle.loads(path.read_bytes()); assert m['name']==path.stem
        fitted += audit_model(m, labels[m['timeframe']]); models[m['name']]=m
        if len(models)%88==0: print(f'Independent fit audit: {len(models)}/704', flush=True)
    assert len(models)==704 and fitted==report['new_models_fitted']
    for path in Path('artifacts/experiment_013/models').glob('*_expanding_*.pkl'):
        m = pickle.loads(path.read_bytes()); models[m['name']]=m
    assert len(models)==768
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
            asset_curves = np.load(OUT/'accounts'/f'{tf}_{fold}.npz')
            for scenario, opts in SCENARIOS:
                baseline = {}
                for arm in ARMS:
                    name, results = tf+'_'+arm, {}
                    saved = rows[name]['folds'][scenario][fold]
                    for symbol, ctx in contexts.items():
                        m = ctx['market']
                        p, a, ev = reference_policy(ctx, tf, arm, models, opts['delay'])
                        standard, standard_a, _ = policy(ctx, tf, arm, models, opts['delay'])
                        np.testing.assert_allclose(p, standard, atol=2e-12, rtol=0)
                        np.testing.assert_array_equal(a, standard_a)
                        old = [{k: v for k, v in e.items() if k not in ['name', 'scenario', 'fold', 'symbol']} for e in action_index[name, scenario, fold, symbol]]
                        equal(ev, old, 2e-12); events_count += len(ev)
                        r = audit_simulate(m, p, a, opts['cost'])
                        prefix = f'{arm}__{scenario}__{symbol}__'
                        np.testing.assert_allclose(p, asset_curves[prefix+'positions'], atol=2e-12, rtol=0)
                        np.testing.assert_array_equal(a, asset_curves[prefix+'allowed'])
                        np.testing.assert_allclose(r['equity'], asset_curves[prefix+'equity'], atol=2e-9, rtol=0)
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
            asset_curves.close()
            print(f'Independent actions and accounts:{tf} window{fold} ({accounts}/720)', flush=True)
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
    assert [(v['candidate'], v['reference']) for v in report['comparisons']]==contrasts()
    for c in report['comparisons']:
        equal(direct_interval(all_daily[c['candidate']], all_daily[c['reference']]),
            {k: c[k] for k in ['log_wealth_excess', 'intervals']})
    old = {r['name']: r for r in json.loads(Path('artifacts/experiment_013/report.json').read_text())['strategies']}
    for tf in FRAMES:
        for new, prior in [('alpha', 'alpha'), ('current', 'monthly')]:
            for field in ['folds', 'development', 'recent', 'development_repair', 'recent_repair']:
                equal(rows[tf+'_'+new][field], old[tf+'_'+prior][field])
            for group in ['development', 'recent']:
                with np.load(OUT/f'{tf}_{group}_equity.npz') as current, np.load(Path('artifacts/experiment_013')/f'{tf}_{group}_equity.npz') as previous:
                    for sc, _ in SCENARIOS:
                        np.testing.assert_allclose(current[f'{tf}_{new}__{sc}'], previous[f'{tf}_{prior}__{sc}'], atol=2e-9, rtol=0)
    diagnostic_count = check_diagnostics(actions, labels, json.loads((OUT/'diagnostics.json').read_text()))
    tests = subprocess.run([sys.executable, '-m', 'pytest', '-q'], capture_output=True, text=True, check=True)
    (OUT/'pytest-output.txt').write_text(tests.stdout, encoding='utf-8')
    freeze()  # Includes all prior shadow state and original model bytes.
    assert accounts==720
    result = dict(tests_passed=int(re.search(r'(\d+) passed', tests.stdout)[1]), raw_archives=raw,
        new_fit_slots=704, models_independently_refitted=fitted, no_model_slots=704-fitted,
        cached_references_verified=64, asset_windows=accounts, action_events=events_count,
        episode_attributions=trades_count, diagnostic_rows=diagnostic_count, intervals=40,
        independent_membership_quantiles_weights_scalers=True, manual_tree_predictions_and_ridge_solutions=True,
        full_asset_and_portfolio_curves_verified=True, repair_annual_asset_and_cost_summaries_verified=True,
        study013_alpha_and_current_exactly_reproduced=True, frozen_hashes_and_shadow_state_unchanged=True,
        production_selection=None)
    write_json(OUT/'verification.json', result); print(json.dumps(result, indent=2), flush=True)


if __name__=='__main__':
    run()
