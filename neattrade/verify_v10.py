"""Independent short-book integration, selected-model and financial replay."""
from hashlib import sha256
import json
from pathlib import Path
import pickle
import re
import subprocess
import sys

import neat
from numba import njit
import numpy as np
import pandas as pd

from .fast_accounting import audit_simulate
from .market import simulate
from .orderbook import ROOT as BOOK_ROOT, SYMBOLS
from .overlay import incremental_objective
from .policies import mlp, CONNECTIONS
from .research import batch_activate
from .research_v3 import FOLDS, SEEDS
from .research_v5 import model_config
from .research_v6 import write_json
from .research_v9 import train_markets, base_targets, network_inputs, targets
from .research_v10 import (OUT, SHORT, ROOT, SPECS, WINDOWS, SCENARIOS, POPULATION, dataset,
                          market, prepare, allowed_times, desired, candidates_for, grouped)
from .timeframes import TIMEFRAMES, PER_YEAR, portfolio, metrics, daily_returns


@njit(cache=True)
def direct_windows(available, values, ends, horizon):
    result = np.zeros((len(ends), 3))
    for k, end in enumerate(ends):
        stop = np.searchsorted(available, end, side='left')
        if stop == 0:
            continue
        first = max(0, np.searchsorted(available, end-horizon, side='right')-1)
        covered, weighted = 0., 0.
        for i in range(first, stop):
            next_time = available[i+1] if i+1 < stop else end
            duration = max(0, min(next_time, available[i]+120, end)-max(available[i], end-horizon))
            if np.isfinite(values[i]):
                covered += duration
                weighted += duration*values[i]
        valid = covered/horizon >= .95-1e-12 and np.isfinite(values[stop-1]) and end-available[stop-1] <= 120
        result[k, 0] = weighted/covered if valid and covered else 0.
        result[k, 1] = 1. if valid else 0.
        result[k, 2] = covered/horizon
    return result


def preflight():
    meta = json.loads((ROOT/'feature_manifest.json').read_text())
    original = json.loads((BOOK_ROOT/'feature_manifest.json').read_text())
    assert sha256((BOOK_ROOT/'feature_manifest.json').read_bytes()).hexdigest() == meta['source_feature_manifest_sha256']
    assert sha256(Path('neattrade/short_book.py').read_bytes()).hexdigest() == meta['module_sha256']
    checks, largest = 0, 0.
    for s in SYMBOLS:
        path = BOOK_ROOT/f'{s}_snapshots.npz'
        assert sha256(path.read_bytes()).hexdigest() == original['symbols'][s]['snapshot_sha256']
        with np.load(path) as d:
            available, values = d['times']+60, d['values']
        for tf in SHORT:
            path = ROOT/f'{s}_{tf}.csv'
            assert sha256(path.read_bytes()).hexdigest() == meta['files'][path.as_posix()]
            f = pd.read_csv(path, index_col=0, parse_dates=True)
            f.index = pd.to_datetime(f.index, utc=True)
            f = f.loc[(f.index.hour*60+f.index.minute) % TIMEFRAMES[tf] == 0]
            reference = direct_windows(available, values, f.index.as_unit('ns').asi8//10**9, TIMEFRAMES[tf]*60)
            stored = f[['book_imbalance', 'book_valid', 'book_coverage']].to_numpy(float)
            np.testing.assert_allclose(reference, stored, atol=2e-10, rtol=0)
            largest = max(largest, float(np.max(np.abs(reference-stored))))
            checks += len(f)
            print(f'Independent book windows verified: {s} {tf}, {len(f):,}', flush=True)
    result = {'book_window_checks': checks, 'max_feature_error': largest,
        'feature_manifest_sha256': sha256((ROOT/'feature_manifest.json').read_bytes()).hexdigest(),
        'quality': meta['quality']}
    write_json(Path('artifacts/study010_preflight.json'), result)
    return result


def reference_raw(member, x):
    if member['kind'] == 'cma':
        # Per-node sums, independent of the broadcast/einsum policy implementation.
        w = member['weights']
        hidden = np.column_stack([np.tanh(sum(x[:, i]*w[4*i+j] for i in range(5))+w[20+j]) for j in range(4)])
        ref = np.tanh(sum(hidden[:, j]*w[24+j] for j in range(4))+w[28])
        fast = mlp(w, x)[0]
    else:
        net = neat.nn.FeedForwardNetwork.create(member['genome'], model_config(POPULATION, 5))
        ref = np.array([net.activate(row)[0] for row in x])
        fast = batch_activate(net, x)
    np.testing.assert_allclose(ref, fast, atol=2e-10, rtol=0)
    return ref


def held_targets(values, anchors, length):
    return np.repeat(values, np.diff(np.r_[anchors, length]))


def reference_interval(parts, refs):
    """Same random experiment, independently summing circular blocks via prefixes."""
    differences = [np.log1p(a)-np.log1p(b) for a, b in zip(parts, refs)]
    intervals = {}
    for block in [7, 28]:
        rng = np.random.default_rng(1010)
        draws = np.zeros(100000)
        for d in differences:
            n = len(d)
            number = int(np.ceil(n/block))
            remainder = n-(number-1)*block
            tiled = np.tile(d, int(np.ceil(block/n))+2)
            cumulative = np.r_[0., np.cumsum(tiled)]
            for i in range(0, len(draws), 2000):
                starts = rng.integers(0, n, size=(2000, number))
                whole = cumulative[starts[:, :-1]+block]-cumulative[starts[:, :-1]]
                last = cumulative[starts[:, -1]+remainder]-cumulative[starts[:, -1]]
                draws[i:i+2000] += whole.sum(axis=1)+last
        intervals[str(block)] = np.quantile(draws, [.025/42, 1-.025/42]).tolist()
    return intervals


def run():
    protocol = json.loads((OUT/'protocol.json').read_text())
    for path, digest in protocol['source_hashes'].items():
        assert sha256(Path(path).read_bytes()).hexdigest() == digest, path
        assert sha256((OUT/'source'/path).read_bytes()).hexdigest() == digest, path
    for path, digest in protocol['evidence_hashes'].items():
        assert sha256(Path(path).read_bytes()).hexdigest() == digest, path
    raw = json.loads(Path('artifacts/experiment_009/manifest.json').read_text())
    raw_count = 0
    for item in raw['market']+raw['book']:
        if item.get('status') == 'missing_404':
            continue
        assert sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
        raw_count += 1
    quality = preflight()
    report = json.loads((OUT/'report.json').read_text())
    bars, frames, late, _ = dataset()
    rows = {r['name']: r for r in report['strategies']}
    models, neural_rows = {}, 0
    for tf in SHORT:
        for fold, start in enumerate([a for a, _ in FOLDS]+['2026-01-01']):
            training = train_markets(bars, frames[tf], start)
            prepared = [prepare(m, tf) for m in training]
            benchmarks = []
            for d in prepared:
                p = held_targets(base_targets(d.x), d.anchors, len(d.market.x))
                r = simulate(d.market, p, trade_allowed=allowed_times(d.market, tf), rebalance_band=.05)
                benchmarks.append({'return': np.array([r['return']])})
            for spec in SPECS:
                for seed in SEEDS:
                    name = f'{tf}_fold{fold}_{spec}_{seed}'
                    path = OUT/'jobs'/f'{name}.pkl'
                    meta = json.loads(path.with_suffix('.json').read_text())
                    assert meta['model_sha256'] == sha256(path.read_bytes()).hexdigest()
                    assert meta['history_sha256'] == sha256(path.with_suffix('.csv').read_bytes()).hexdigest()
                    assert meta['protocol_sha256'] == sha256((OUT/'protocol.json').read_bytes()).hexdigest()
                    member = pickle.loads(path.read_bytes())
                    assert member['name'] == name and member['timeframe'] == tf and member['spec'] == spec and member['seed'] == seed
                    assert pd.Timestamp(member['train_end']) == pd.Timestamp(start, tz='UTC')-pd.Timedelta(days=7)
                    history = pd.read_csv(path.with_suffix('.csv'))
                    assert len(history) == protocol['generations']
                    assert history.candidate_evaluations.sum() == member['candidate_evaluations'] == POPULATION*protocol['generations']
                    if member['kind'] == 'cma':
                        assert np.asarray(member['weights']).shape == (29,)
                        assert np.max(np.abs(member['weights'])) <= 5+1e-10
                        assert history.candidate_evaluations.sum() == POPULATION*protocol['generations']
                    results = []
                    for d in prepared:
                        x = network_inputs(d.x, spec)
                        ref = reference_raw(member, x)
                        neural_rows += len(x)
                        p = held_targets(targets(ref, d.x), d.anchors, len(d.market.x))
                        r = simulate(d.market, p, trade_allowed=allowed_times(d.market, tf), rebalance_band=.05)
                        r = {key: np.array([r[key]]) for key in ['return', 'max_drawdown']}
                        r['ruined'] = np.array([False])
                        r['years'] = (d.market.times[-1]-d.market.times[0]).total_seconds()/(365.25*86400)
                        np.testing.assert_allclose(r['years'], len(d.market.x)/PER_YEAR, atol=1e-14, rtol=0)
                        results.append(r)
                    connections = CONNECTIONS if member['kind'] == 'cma' else len(member['genome'].connections)
                    fitness = incremental_objective(results, benchmarks)[0]-.0001*connections
                    expected = member['fitness'] if member['kind'] == 'cma' else member['genome'].fitness
                    np.testing.assert_allclose(fitness, expected, atol=2e-9, rtol=0)
                    models[name] = member
            print(f'Reference model activation and training fitness: {tf} fold{fold}', flush=True)
    accounting_checks = 0
    all_daily = {}
    for tf in SHORT:
        sequences, records, endpoints = {}, {}, []
        for fold, (start, end) in enumerate(WINDOWS):
            normal = {s: market(s, bars[s], frames[tf][s], start, end) for s in SYMBOLS}
            lagged = {s: market(s, bars[s], late[tf][s], start, end) for s in SYMBOLS}
            endpoints.append(normal[SYMBOLS[0]].times[1:])
            for arm, candidate in candidates_for(models, tf, fold).items():
                name = tf+'_'+arm
                for scenario, options in SCENARIOS:
                    opts = dict(options)
                    use = lagged if opts.pop('lag_book', False) else normal
                    saved = rows[name]['folds'][scenario][fold]
                    result = {}
                    for s, m in use.items():
                        p, allowed = desired(candidate, m, tf, opts.get('zero_book', False))
                        if opts.get('delay'):
                            p = np.r_[0., p[:-1]]
                            allowed = np.r_[False, allowed[:-1]]
                        r = audit_simulate(m, p, allowed, 7*opts.get('multiplier', 1.))
                        for key in ['return', 'max_drawdown', 'sharpe', 'cost_paid_initial_equity',
                                    'funding_paid_initial_equity', 'turnover', 'order_count', 'mean_abs_exposure']:
                            np.testing.assert_allclose(r[key], saved['by_symbol'][s][key], atol=2e-9, rtol=0)
                        result[s] = r
                        accounting_checks += 1
                    port = portfolio(result)
                    for key in ['return', 'max_drawdown', 'sharpe']:
                        np.testing.assert_allclose(port[key], saved['portfolio'][key], atol=2e-9, rtol=0)
                    sequences.setdefault((name, scenario), []).append(port['returns'])
                    records.setdefault((name, scenario), []).append({'by_symbol': result, 'observations': len(port['returns'])})
            print(f'Compiled independent accounting: {accounting_checks}/3600', flush=True)
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
                    for year, value in replay['annual_returns'].items():
                        np.testing.assert_allclose(value, saved['annual_returns'][year], atol=2e-9, rtol=0)
                    np.testing.assert_allclose(curves[name+'__'+scenario], metrics(np.concatenate(parts[sl]))['equity'], atol=2e-9, rtol=0)
    old = pd.read_csv('artifacts/experiment_009/development_equity.csv', index_col=0)
    old.index = pd.to_datetime(old.index, utc=True)
    for arm in ['at_book', 'cma_book']:
        parts = []
        for start, end in FOLDS:
            eq = old.loc[pd.Timestamp(start, tz='UTC'):pd.Timestamp(end, tz='UTC'), arm+'__base']
            r = eq.to_numpy()[1:]/eq.to_numpy()[:-1]-1
            parts.append(daily_returns(r, eq.index[1:]).to_numpy())
        all_daily['4h_'+arm] = parts
    for parts in all_daily.values():
        assert [len(x) for x in parts] == [(pd.Timestamp(z)-pd.Timestamp(a)).days for a, z in FOLDS]
    assert len(report['comparisons']) == 21
    for comparison in report['comparisons']:
        a, b = all_daily[comparison['candidate']], all_daily[comparison['reference']]
        excess = sum((np.log1p(x)-np.log1p(y)).sum() for x, y in zip(a, b))
        np.testing.assert_allclose(excess, comparison['log_wealth_excess'], atol=2e-9, rtol=0)
        intervals = reference_interval(a, b)
        for block in ['7', '28']:
            np.testing.assert_allclose(intervals[block], comparison['adjusted_intervals'][block], atol=2e-9, rtol=0)
    print('Independent daily alignment, 4h reference and 42 bootstrap intervals verified.', flush=True)
    previous = {r['name']: r for r in json.loads(Path('artifacts/experiment_008/report.json').read_text())['strategies']}
    for tf in SHORT:
        for group in ['development', 'recent']:
            for scenario in ['base', 'double_cost', 'delay15m']:
                for key in ['return', 'max_drawdown']:
                    np.testing.assert_allclose(rows[tf+'_alpha'][group][scenario][key], previous[tf+'_long'][group][scenario][key], atol=2e-9, rtol=0)
    tests = subprocess.run([sys.executable, '-m', 'pytest', '-q'], text=True, capture_output=True, check=True)
    (OUT/'pytest-output.txt').write_text(tests.stdout, encoding='utf-8')
    result = {'tests_passed': int(re.search(r'(\d+) passed', tests.stdout)[1]),
        'raw_archives_verified': raw_count, 'independent_book_window_checks': quality['book_window_checks'],
        'selected_models_replayed': len(models), 'reference_neural_activation_rows': neural_rows,
        'training_fitness_replayed': True, 'independent_asset_accounting_checks': accounting_checks,
        'full_curves_and_annual_returns_verified': True, 'study008_alpha_controls_reproduced': True,
        'paired_intervals_independently_recomputed': 42,
        'actual_candidate_evaluations': sum(m['candidate_evaluations'] for m in models.values()),
        'study009_context_hashes_verified': True, 'frozen_sources_verified': True}
    write_json(OUT/'verification.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    if '--preflight' in sys.argv:
        preflight()
    else:
        run()
