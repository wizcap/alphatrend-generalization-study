"""Fixed AlphaTrend timeframe comparison on one verified15m execution grid."""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from .data import load as load4h, read_archive
from .market import features as features4h, summary
from .research_v3 import FOLDS, snapshot
from .research_v6 import write_json
from .sources_v8 import SYMBOLS, load
from .timeframes import (TIMEFRAMES, aggregate, inputs, make_market, evaluate, portfolio,
                         metrics, daily_returns, specifications, scenarios)

RECENT = [('2026-01-01', '2026-07-01'), ('2026-07-01', '2026-08-31 20:00')]
WINDOWS = FOLDS+RECENT
OUT = Path('artifacts/experiment_008')


def dataset():
    bars, signals, audit = {}, {}, {}
    for symbol in SYMBOLS:
        bars[symbol], audit[symbol] = load(symbol)
        signals[symbol] = inputs(bars[symbol])
    return bars, signals, audit


def check_4h_sources(bars, signals):
    result = {}
    from .alphatrend import alpha_indicator
    for symbol, b in bars.items():
        old = load4h(symbol).loc[:'2026-08-31 20:00']
        resampled = aggregate(b, '4h').reindex(old.index)
        max_diffs, conflicts = {}, []
        for col in ['open', 'high', 'low', 'close', 'volume']:
            difference = float(np.abs(resampled[col]-old[col]).max())
            max_diffs[col] = difference
            bad = (resampled[col]-old[col]).abs() > (1e-6 if col == 'volume' else 1e-8)
            conflicts.extend({'time': str(t), 'field': col, 'from15m': float(resampled.loc[t, col]),
                              'old4h': float(old.loc[t, col])} for t in old.index[bad])
        # Same4h information content; fifteen-minute observations can still
        # expose additional intra4h drawdown without changing decisions.
        alpha = alpha_indicator(old).state.shift()
        valid = old.index[old.index >= '2024-01-01']
        state_disagreements = int((signals[symbol]['states']['4h'].reindex(valid) != alpha.reindex(valid)).sum())
        vol = features4h(old)['vol']
        np.testing.assert_allclose(signals[symbol]['vol'].reindex(valid), vol.reindex(valid), atol=1e-12, rtol=0)
        result[symbol] = {'price_volume_max_absolute_differences': max_diffs, 'conflicts': conflicts,
                          'alpha_state_disagreements': state_disagreements, 'common_vol_matches': True}
    return result


def source_sensitivity(bars):
    """Alternative, never silently chosen primary:1m aggregates on8 audited days."""
    result = {s: b.copy() for s, b in bars.items()}
    manifest = json.loads(Path('data/study008/conflicts1m/manifest.json').read_text())
    for entry in manifest:
        frame = read_archive(entry['path'])
        frame.index = pd.to_datetime(frame.time, unit='ms', utc=True)
        expected = pd.date_range(frame.index[0].normalize(), periods=1440, freq='min')
        if not frame.index.equals(expected):
            raise ValueError('Source-sensitivity day has incomplete1m data')
        aggregation = {c: 'sum' for c in frame.columns}
        aggregation.update(time='first', open='first', high='max', low='min', close='last', close_time='last', ignore='last')
        grouped = frame.resample('15min', origin='epoch', label='left', closed='left').agg(aggregation)
        symbol = Path(entry['path']).name.split('-')[0]
        result[symbol].loc[grouped.index, grouped.columns] = grouped.to_numpy()
    return result


def paired_interval(parts, reference):
    deltas = [np.log1p(a)-np.log1p(b) for a, b in zip(parts, reference)]
    intervals = {}
    for block in [7, 28]:
        rng = np.random.default_rng(808)
        samples = np.zeros(100000)
        for delta in deltas:
            for a in range(0, len(samples), 2000):
                starts = rng.integers(0, len(delta), size=(2000, int(np.ceil(len(delta)/block))))
                ix = ((starts[:, :, None]+np.arange(block)) % len(delta)).reshape(2000, -1)[:, :len(delta)]
                samples[a:a+2000] += delta[ix].sum(axis=1)
        low, high = np.quantile(samples, [.025/36, 1-.025/36])
        intervals[str(block)] = [float(low), float(high)]
    active = [np.abs(d) > 1e-12 for d in deltas]
    return {'log_wealth_excess': float(sum(d.sum() for d in deltas)), 'adjusted_intervals': intervals,
            'nonzero_difference_days': int(sum(a.sum() for a in active)),
            'nonzero_runs_within_windows': int(sum((a & ~np.r_[False, a[:-1]]).sum() for a in active)),
            'draws': 100000, 'contrasts': 18, 'intervals_corrected': 36, 'block_days': [7, 28],
            'note': 'Descriptive known-data comparison; no correction for previous adaptive research.'}


def grouped_summary(parts, records, endpoints):
    r = np.concatenate(parts)
    result = summary(metrics(r))
    daily = daily_returns(r, endpoints)
    count_index = (pd.DatetimeIndex(endpoints).tz_localize(None)-pd.Timedelta(nanoseconds=1)).to_period('D')
    counts = pd.Series(1, index=count_index).groupby(level=0).sum()
    full_days = daily[counts == 96]
    result['daily_sharpe'] = float(full_days.mean()/full_days.std(ddof=0)*np.sqrt(365.25)) if full_days.std(ddof=0) > 1e-12 else 0.
    result['annual_returns'] = {str(year): float(np.prod(1+values)-1) for year, values in daily.groupby(daily.index.year)}
    result['by_symbol_return'] = {s: float(np.prod([1+row['by_symbol'][s]['return'] for row in records])-1) for s in SYMBOLS}
    result['order_count'] = sum(row['by_symbol'][s]['order_count'] for row in records for s in SYMBOLS)
    weights = [row['observations'] for row in records for _ in SYMBOLS]
    result['mean_abs_exposure'] = float(np.average([row['by_symbol'][s]['mean_abs_exposure'] for row in records for s in SYMBOLS], weights=weights))
    result['mean_turnover_per_sleeve_window'] = float(np.mean([row['by_symbol'][s]['turnover'] for row in records for s in SYMBOLS]))
    result['mean_cost_per_sleeve_window'] = float(np.mean([row['by_symbol'][s]['cost_paid_initial_equity'] for row in records for s in SYMBOLS]))
    result['mean_funding_per_sleeve_window'] = float(np.mean([row['by_symbol'][s]['funding_paid_initial_equity'] for row in records for s in SYMBOLS]))
    return result


def run():
    if (OUT/'protocol.json').exists():
        raise ValueError('Use an unoccupied experiment directory; completed study is immutable')
    OUT.mkdir(parents=True, exist_ok=True)
    bars, signals, audit = dataset()
    alternative = source_sensitivity(bars)
    alternative_signals = {s: inputs(b) for s, b in alternative.items()}
    write_json(OUT/'source_audit.json', audit)
    write_json(OUT/'aggregation_4h_audit.json', check_4h_sources(bars, signals))
    hashes = snapshot(OUT)
    for path in [Path('reports/study008_method.md'), *Path('tests').glob('test_*.py')]:
        target = OUT/'source'/path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[path.as_posix()] = sha256(path.read_bytes()).hexdigest()
    source_manifest = json.loads(Path('data/study008/manifest.json').read_text())
    source_manifest += json.loads(Path('data/study008/conflicts1m/manifest.json').read_text())
    funding_paths = sorted({p for a in audit.values() for p in a['funding_files']})
    for p in funding_paths:
        path = Path(p)
        source_manifest.append({'path': p, 'sha256': sha256(path.read_bytes()).hexdigest(),
                                'bytes': path.stat().st_size, 'role': 'existing_official_funding'})
    write_json(OUT/'manifest.json', source_manifest)
    protocol = {'study': '008', 'specifications': specifications(), 'timeframes_minutes': TIMEFRAMES,
                'windows': WINDOWS, 'primary_windows': 4, 'recent_known_context_windows': 2,
                'alpha_period': 14, 'alpha_multiplier': 1., 'risk': 'same42x4h volatility clock',
                'fee_bps': 5, 'slippage_bps': 2, 'band': .05, 'accounting_minutes': 15,
                'scenarios': {tf: scenarios(tf) for tf in TIMEFRAMES},
                'selection': 'Comparison only; no deployment or automatic chosen timeframe',
                'method': 'source/reports/study008_method.md', 'source_hashes': hashes,
                'manifest_sha256': sha256((OUT/'manifest.json').read_bytes()).hexdigest(),
                'numpy_version': np.__version__, 'pandas_version': pd.__version__}
    write_json(OUT/'protocol.json', protocol)
    started = time.monotonic()
    records, sequences, endpoints = {}, {}, []
    for fold, (start, end) in enumerate(WINDOWS):
        markets = {s: make_market(s, bars[s], signals[s], start, end) for s in SYMBOLS}
        assert markets[SYMBOLS[0]].times.equals(markets[SYMBOLS[1]].times)
        endpoints.append(markets[SYMBOLS[0]].times[1:])
        for spec in specifications():
            name, tf = spec['name'], spec['timeframe']
            for scenario, options in scenarios(tf):
                results = {s: evaluate(m, signals[s], tf, spec['long_short'], passive=spec.get('passive'), **options) for s, m in markets.items()}
                port = portfolio(results)
                records.setdefault((name, scenario), []).append({'window': [start, end], 'observations': len(port['returns']),
                    'portfolio': summary(port), 'by_symbol': {s: summary(r) for s, r in results.items()}})
                sequences.setdefault((name, scenario), []).append(port['returns'])
            alt_markets = {s: make_market(s, alternative[s], alternative_signals[s], start, end) for s in SYMBOLS}
            alt_result = {s: evaluate(m, alternative_signals[s], tf, spec['long_short'], passive=spec.get('passive')) for s, m in alt_markets.items()}
            port = portfolio(alt_result)
            records.setdefault((name, 'source_sensitivity'), []).append({'window': [start, end], 'observations': len(port['returns']),
                'portfolio': summary(port), 'by_symbol': {s: summary(r) for s, r in alt_result.items()}})
            sequences.setdefault((name, 'source_sensitivity'), []).append(port['returns'])
        print(f'Completed22strategies x4scenarios + source sensitivity: window{fold} {start} to {end}', flush=True)
    dates = {'development': pd.DatetimeIndex(np.concatenate([v.to_numpy() for v in endpoints[:4]])),
             'recent': pd.DatetimeIndex(np.concatenate([v.to_numpy() for v in endpoints[4:]]))}
    rows = []
    for spec in specifications():
        name = spec['name']
        row = dict(spec)
        for group, sl in [('development', slice(0, 4)), ('recent', slice(4, 6))]:
            row[group] = {scenario: grouped_summary(sequences[name, scenario][sl], records[name, scenario][sl], dates[group])
                          for scenario, _ in scenarios(spec['timeframe'])}
            row[group]['source_sensitivity'] = grouped_summary(sequences[name, 'source_sensitivity'][sl], records[name, 'source_sensitivity'][sl], dates[group])
        row['folds'] = {scenario: records[name, scenario] for scenario, _ in scenarios(spec['timeframe'])}
        row['folds']['source_sensitivity'] = records[name, 'source_sensitivity']
        base = row['development']['base']
        row['descriptive_checks'] = {
            'both_development_years_positive': all(v > 0 for v in base['annual_returns'].values()),
            'both_assets_positive': all(v > 0 for v in base['by_symbol_return'].values()),
            'drawdown_at_most20pct': base['max_drawdown'] <= .20,
            'sharpe_at_least0_5': base['sharpe'] >= .5,
            'three_positive_halfyears': sum(r['portfolio']['return'] > 0 for r in row['folds']['base'][:4]) >= 3,
            'all_stresses_positive': all(v['return'] > 0 for k, v in row['development'].items() if k != 'source_sensitivity'),
            'known2026positive': row['recent']['base']['return'] > 0}
        row['all_descriptive_checks_pass'] = all(row['descriptive_checks'].values())
        rows.append(row)
    contrasts = []
    for spec in specifications():
        if spec['timeframe'] == '4h' or spec.get('passive'):
            continue
        ref = '4h_'+('ls' if spec['long_short'] else 'long')
        a = [daily_returns(r, d).to_numpy() for r, d in zip(sequences[spec['name'], 'base'][:4], endpoints[:4])]
        b = [daily_returns(r, d).to_numpy() for r, d in zip(sequences[ref, 'base'][:4], endpoints[:4])]
        contrasts.append({'candidate': spec['name'], 'reference': ref, **paired_interval(a, b)})
    for group, sl, start in [('development', slice(0, 4), FOLDS[0][0]), ('recent', slice(4, 6), RECENT[0][0])]:
        curve_index = pd.DatetimeIndex([pd.Timestamp(start, tz='UTC')]+dates[group].tolist())
        curve, month = {}, {}
        month_index = (dates[group].tz_localize(None)-pd.Timedelta(nanoseconds=1)).to_period('M')
        for spec in specifications():
            for scenario in [s for s, _ in scenarios(spec['timeframe'])]+['source_sensitivity']:
                name = spec['name']
                r = np.concatenate(sequences[name, scenario][sl])
                curve[name+'__'+scenario] = metrics(r)['equity']
                month[name+'__'+scenario] = pd.Series(1+r, index=month_index).groupby(level=0).prod()-1
        pd.DataFrame(curve, index=curve_index).to_csv(OUT/f'{group}_equity.csv')
        pd.DataFrame(month).to_csv(OUT/f'{group}_monthly.csv')
    write_json(OUT/'report.json', {'strategies': rows, 'contrasts': contrasts, 'selected': None,
        'primary_known_development': True, 'recent_known_context': True,
        'distinct_strategy_scenario_window_runs': 22*5*6, 'asset_simulations': 22*5*6*2,
        'elapsed_seconds': time.monotonic()-started})
    print(pd.DataFrame([{'name': r['name'], **{k: r['development']['base'][k] for k in ['return', 'max_drawdown', 'sharpe']},
                         '2026': r['recent']['base']['return'], 'checks': r['all_descriptive_checks_pass']} for r in rows]).to_string(index=False), flush=True)


if __name__ == '__main__':
    run()
