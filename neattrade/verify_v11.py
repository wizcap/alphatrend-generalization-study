"""Independent activation, scalar fitness, compiled accounting and resampling audit."""
from hashlib import sha256
import json
from pathlib import Path
import pickle
import re
import subprocess
import sys

import neat
import numpy as np
import pandas as pd

from .fast_accounting import audit_simulate
from .indicators_v11 import (SYMBOLS, FRAMES, PACKS, FEATURES, COLUMNS, dataset, build_frame,
    market, train_markets, inputs, allowed_times, alpha_target, learned_target)
from .market import simulate
from .overlay import incremental_objective
from .policies_v11 import dimensions
from .research_v3 import FOLDS, SEEDS
from .research_v5 import model_config
from .research_v6 import write_json
from .research_v11 import (OUT, SPECS, WINDOWS, SCENARIOS, POPULATION, GENERATIONS,
    INTERVAL_COUNT, DRAWS, raw_output, prepare, desired, candidates_for, grouped, verify_raw_sources, contrasts)
from .timeframes import TIMEFRAMES, PER_YEAR, portfolio, metrics, daily_returns


def preflight():
    raw_count = verify_raw_sources()
    bars, frames, quality = dataset()
    checked = 0
    for s, b in bars.items():
        for boundary in ['2024-01-01', '2025-07-01', '2026-07-01']:
            t = pd.Timestamp(boundary, tz='UTC')
            changed = b.copy()
            changed.loc[t:, ['open', 'high', 'low', 'close', 'volume', 'buy_volume', 'mark']] *= 1.07
            changed.loc[t:, 'funding'] += .001
            for tf in FRAMES:
                pd.testing.assert_frame_equal(frames[tf][s].loc[:t], build_frame(changed, tf).loc[:t])
                prefix = b.loc[:t+pd.Timedelta(hours=4)-pd.Timedelta(minutes=15)]
                pd.testing.assert_frame_equal(frames[tf][s].loc[:t], build_frame(prefix, tf).loc[:t])
                checked += 1
        print(f'Causal prefix / future mutation verified: {s}', flush=True)
    result = {'raw_archives_verified': raw_count, 'causal_cases': checked, 'quality': quality,
              'actual_input_counts': {p: len(n) for p, n in PACKS.items()},
              'source_hashes': {str(p): sha256(p.read_bytes()).hexdigest() for p in [Path('neattrade/indicators_v11.py'), Path('neattrade/policies_v11.py')]}}
    write_json(Path('artifacts/study011_preflight.json'), result)
    return result


def reference_raw(member, x):
    kind, n = member['kind'], x.shape[1]
    if kind == 'linear':
        w = member['weights']
        ref = np.tanh(sum(x[:, i]*w[i] for i in range(n))+w[-1])
    elif kind == 'mlp':
        w = member['weights']
        hidden = [np.tanh(sum(x[:, i]*w[4*i+j] for i in range(n))+w[4*n+j]) for j in range(4)]
        ref = np.tanh(sum(hidden[j]*w[4*n+4+j] for j in range(4))+w[-1])
    else:
        net = neat.nn.FeedForwardNetwork.create(member['genome'], model_config(POPULATION, n))
        ref = np.array([net.activate(row)[0] for row in x])
    np.testing.assert_allclose(ref, raw_output(member, x), atol=2e-10, rtol=0)
    return ref


def reference_targets(candidate, m, tf):
    stride = TIMEFRAMES[tf]//15
    anchors = np.arange(0, len(m.x), stride)
    x = m.x[anchors]
    cap = np.clip(.20/np.maximum(x[:, COLUMNS.index('risk_vol')], .10), 0, .5)
    if isinstance(candidate, list):
        full = sum(reference_targets(member, m, tf) for member in candidate)/len(candidate)
        return full
    if isinstance(candidate, dict):
        selected = x[:, [FEATURES.index(n) for n in candidate['input_features']]]
        z = reference_raw(candidate, selected)
        target = np.full(len(z), .5)
        target[z < -.35], target[z > .35] = 0., 1.
        target *= cap
    elif candidate in ['alpha', 'alpha_half']:
        target = np.maximum(x[:, 0], 0)*cap*(.5 if candidate == 'alpha_half' else 1.)
    elif candidate == 'ema':
        target = np.where(x[:, FEATURES.index('ema12_26')] > 0, cap, 0.)
    elif candidate == 'passive_vol':
        target = cap
    elif candidate == 'cash':
        target = np.zeros(len(x))
    else:
        raise ValueError('Unknown audit candidate')
    return target[np.arange(len(m.x))//stride]


def reference_interval(parts, refs):
    differences = [np.log1p(a)-np.log1p(b) for a, b in zip(parts, refs)]
    result = {}
    for block in [7, 28]:
        rng = np.random.default_rng(1011)
        draws = np.zeros(DRAWS)
        for d in differences:
            for i in range(0, DRAWS, 2000):
                starts = rng.integers(0, len(d), size=(2000, int(np.ceil(len(d)/block))))
                indices = ((starts[:, :, None]+np.arange(block)) % len(d)).reshape(2000, -1)[:, :len(d)]
                draws[i:i+2000] += d[indices].sum(axis=1)
        result[str(block)] = np.quantile(draws, [.025/INTERVAL_COUNT, 1-.025/INTERVAL_COUNT]).tolist()
    return result


def run():
    protocol = json.loads((OUT/'protocol.json').read_text())
    fingerprint = sha256((OUT/'protocol.json').read_bytes()).hexdigest()
    for path, digest in {**protocol['source_hashes'], **protocol['evidence_hashes']}.items():
        assert sha256(Path(path).read_bytes()).hexdigest() == digest, path
    raw_count = verify_raw_sources()
    pre = json.loads(Path('artifacts/study011_preflight.json').read_text())
    assert pre['causal_cases'] == 18
    for path, digest in pre['source_hashes'].items():
        assert sha256(Path(path).read_bytes()).hexdigest() == digest
    bars, frames, _ = dataset()
    report = json.loads((OUT/'report.json').read_text())
    rows = {r['name']: r for r in report['strategies']}
    models, neural_rows = {}, 0
    for tf in FRAMES:
        for fold, start in enumerate([a for a, _ in FOLDS]+['2026-01-01']):
            training = train_markets(bars, frames[tf], start)
            prepared = [prepare(m, tf) for m in training]
            benchmarks = []
            for m in training:
                r = simulate(m, reference_targets('alpha', m, tf), trade_allowed=allowed_times(m, tf), rebalance_band=.05)
                benchmarks.append({'return': np.array([r['return']])})
            for kind, pack in SPECS:
                for seed in SEEDS:
                    name = f'{tf}_fold{fold}_{kind}_{pack}_{seed}'
                    path = OUT/'jobs'/f'{name}.pkl'
                    meta = json.loads(path.with_suffix('.json').read_text())
                    assert meta['model_sha256'] == sha256(path.read_bytes()).hexdigest()
                    assert meta['history_sha256'] == sha256(path.with_suffix('.csv').read_bytes()).hexdigest()
                    assert meta['protocol_sha256'] == fingerprint
                    member = pickle.loads(path.read_bytes())
                    assert (member['name'], member['timeframe'], member['kind'], member['pack'], member['seed']) == (name, tf, kind, pack, seed)
                    assert member['input_features'] == PACKS[pack] and member['input_count'] == len(PACKS[pack])
                    assert pd.Timestamp(member['train_end']) == pd.Timestamp(start, tz='UTC')-pd.Timedelta(days=7)
                    history = pd.read_csv(path.with_suffix('.csv'))
                    assert len(history) == GENERATIONS
                    assert (history.candidate_evaluations == POPULATION).all()
                    assert history.candidate_evaluations.sum() == member['candidate_evaluations'] == POPULATION*GENERATIONS
                    if kind != 'neat':
                        assert np.asarray(member['weights']).shape == (dimensions(kind, len(PACKS[pack]))[0],)
                        assert np.max(np.abs(member['weights'])) <= 5+1e-10
                    results = []
                    for d in prepared:
                        p = reference_targets(member, d.market, tf)
                        neural_rows += len(d.x)
                        np.testing.assert_allclose(p, desired(member, d.market, tf)[0], atol=2e-12, rtol=0)
                        r = simulate(d.market, p, trade_allowed=allowed_times(d.market, tf), rebalance_band=.05)
                        values = {key: np.array([r[key]]) for key in ['return', 'max_drawdown']}
                        values['ruined'] = np.array([False])
                        values['years'] = (d.market.times[-1]-d.market.times[0]).total_seconds()/(365.25*86400)
                        np.testing.assert_allclose(values['years'], len(d.market.x)/PER_YEAR, atol=1e-14, rtol=0)
                        results.append(values)
                    connections = dimensions(kind, len(PACKS[pack]))[1] if kind != 'neat' else len(member['genome'].connections)
                    expected = member['fitness'] if kind != 'neat' else member['genome'].fitness
                    np.testing.assert_allclose(incremental_objective(results, benchmarks)[0]-.0001*connections, expected, atol=2e-9, rtol=0)
                    models[name] = member
            print(f'Reference activation and training fitness: {tf} fold{fold}, {len(models)}/540 models', flush=True)
    checks, all_daily = 0, {}
    for tf in FRAMES:
        sequences, records, endpoints = {}, {}, []
        for fold, (start, end) in enumerate(WINDOWS):
            markets = {s: market(s, bars[s], frames[tf][s], start, end) for s in SYMBOLS}
            endpoints.append(markets[SYMBOLS[0]].times[1:])
            for arm, candidate in candidates_for(models, tf, fold).items():
                name = tf+'_'+arm
                targets = {s: reference_targets(candidate, m, tf) for s, m in markets.items()}
                for s, m in markets.items():
                    np.testing.assert_allclose(targets[s], desired(candidate, m, tf)[0], atol=2e-12, rtol=0)
                    if isinstance(candidate, list):
                        p, a = targets[s], allowed_times(m, tf)
                        alpha = reference_targets('alpha', m, tf)
                        saved = next(v for v in report['decision_diagnostics'] if v['name'] == name and v['fold'] == fold and v['symbol'] == s)
                        assert saved['decisions'] == int(a.sum())
                        assert saved['target_differences_vs_alpha'] == int((np.abs(p[a]-alpha[a])>1e-12).sum())
                        assert saved['long_while_alpha_flat'] == int(((p[a]>0) & (alpha[a]==0)).sum())
                        np.testing.assert_allclose(saved['flat_fraction'], np.mean(p[a] == 0))
                for scenario, opts in SCENARIOS:
                    saved = rows[name]['folds'][scenario][fold]
                    result = {}
                    for s, m in markets.items():
                        p, allowed = targets[s], allowed_times(m, tf)
                        if opts.get('delay'):
                            p, allowed = np.r_[0., p[:-1]], np.r_[False, allowed[:-1]]
                        r = audit_simulate(m, p, allowed, 7*opts.get('multiplier', 1.))
                        for key in ['return', 'max_drawdown', 'sharpe', 'cost_paid_initial_equity',
                                    'funding_paid_initial_equity', 'turnover', 'order_count', 'mean_abs_exposure']:
                            np.testing.assert_allclose(r[key], saved['by_symbol'][s][key], atol=2e-9, rtol=0)
                        result[s] = r
                        checks += 1
                    port = portfolio(result)
                    for key in ['return', 'max_drawdown', 'sharpe']:
                        np.testing.assert_allclose(port[key], saved['portfolio'][key], atol=2e-9, rtol=0)
                    sequences.setdefault((name, scenario), []).append(port['returns'])
                    records.setdefault((name, scenario), []).append({'by_symbol': result, 'observations': len(port['returns'])})
            print(f'Independent accounting: {checks}/5724 asset windows', flush=True)
        for (name, scenario), parts in sequences.items():
            if scenario == 'base':
                all_daily[name] = [daily_returns(r, t).to_numpy() for r, t in zip(parts[:4], endpoints[:4])]
        for group, sl in [('development', slice(0, 4)), ('recent', slice(4, 6))]:
            dates = pd.DatetimeIndex(np.concatenate([v.to_numpy() for v in endpoints[sl]]))
            with np.load(OUT/f'{tf}_{group}_equity.npz') as curves:
                np.testing.assert_array_equal(curves['timestamps'][1:], dates.as_unit('ns').asi8)
                for (name, scenario), parts in sequences.items():
                    replay = grouped(parts[sl], records[name, scenario][sl], dates)
                    saved = rows[name][group][scenario]
                    for key in ['return', 'max_drawdown', 'sharpe', 'daily_sharpe', 'orders', 'mean_abs_exposure']:
                        np.testing.assert_allclose(replay[key], saved[key], atol=2e-9, rtol=0)
                    for key in ['annual_returns', 'by_symbol_return']:
                        for subkey, value in replay[key].items():
                            np.testing.assert_allclose(value, saved[key][subkey], atol=2e-9, rtol=0)
                    np.testing.assert_allclose(curves[name+'__'+scenario], metrics(np.concatenate(parts[sl]))['equity'], atol=2e-9, rtol=0)
    assert checks == 5724 and len(models) == 540
    for parts in all_daily.values():
        assert [len(x) for x in parts] == [(pd.Timestamp(z)-pd.Timestamp(a)).days for a, z in FOLDS]
    assert [(c['candidate'], c['reference']) for c in report['comparisons']] == contrasts()
    for number, comparison in enumerate(report['comparisons']):
        a, b = all_daily[comparison['candidate']], all_daily[comparison['reference']]
        excess = sum((np.log1p(x)-np.log1p(y)).sum() for x, y in zip(a, b))
        np.testing.assert_allclose(excess, comparison['log_wealth_excess'], atol=2e-9, rtol=0)
        intervals = reference_interval(a, b)
        for block in ['7', '28']:
            np.testing.assert_allclose(intervals[block], comparison['adjusted_intervals'][block], atol=2e-9, rtol=0)
        if number % 10 == 0:
            print(f'Independent paired bootstrap: {2*(number+1)}/126 intervals', flush=True)
    previous = {r['name']: r for r in json.loads(Path('artifacts/experiment_008/report.json').read_text())['strategies']}
    for tf in FRAMES:
        for group in ['development', 'recent']:
            for scenario in ['base', 'double_cost', 'delay15m']:
                for key in ['return', 'max_drawdown']:
                    np.testing.assert_allclose(rows[tf+'_alpha'][group][scenario][key], previous[tf+'_long'][group][scenario][key], atol=2e-9, rtol=0)
    tests = subprocess.run([sys.executable, '-m', 'pytest', '-q'], text=True, capture_output=True, check=True)
    (OUT/'pytest-output.txt').write_text(tests.stdout, encoding='utf-8')
    result = {'tests_passed': int(re.search(r'(\d+) passed', tests.stdout)[1]),
        'raw_archives_verified': raw_count, 'causal_prefix_future_cases': pre['causal_cases'],
        'selected_models_replayed': len(models), 'reference_training_activation_rows': neural_rows,
        'training_fitness_replayed': True, 'independent_asset_accounting_checks': checks,
        'evaluation_targets_independently_recomputed': True, 'decision_diagnostics_verified': True,
        'full_curves_annual_and_symbol_returns_verified': True, 'study008_alpha_controls_reproduced': True,
        'paired_intervals_independently_recomputed': INTERVAL_COUNT,
        'actual_candidate_evaluations': sum(m['candidate_evaluations'] for m in models.values()),
        'frozen_sources_verified': True, 'depth_data_required': False}
    write_json(OUT/'verification.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    preflight() if '--preflight' in sys.argv else run()
