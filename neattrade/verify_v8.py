"""Separate segmented accounting and legacy4h parity for study008."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .batch import DailyMarket, fitness_accounting
from .research_v6 import write_json
from .research_v8 import OUT, WINDOWS, SYMBOLS, dataset, check_4h_sources, source_sensitivity
from .timeframes import (TIMEFRAMES, aggregate, make_market, orders, evaluate, portfolio,
                         metrics, specifications, scenarios, daily_returns, inputs)
from .verify_v7 import independent_alpha


def accounting(m, target, allowed, multiplier):
    anchors = np.unique(np.r_[0, np.flatnonzero(allowed)])
    positions = target[anchors].copy()
    if not allowed[0]:
        positions[0] = 0.
    segments = []
    for a, z in zip(anchors, np.r_[anchors[1:], len(target)]):
        cumulative = m.prices[a+1:z+1]-m.prices[a]-np.cumsum(m.marks[a+1:z+1]*m.funding[a+1:z+1])
        segments.append((int(a), int(z), cumulative))
    d = DailyMarket(m, anchors, m.x[anchors], segments)
    # Only financial results are used: batch.py's legacy4h years field is NOT
    # used for scoring. Every cumulative path here includes each15m boundary.
    return fitness_accounting(d, positions[None, :], cost_bps=7*multiplier, rebalance_band=.05)


def verify_legacy(bars, features):
    from .market import Market, simulate
    previous = json.loads(Path('artifacts/experiment_007/selection.json').read_text())
    old = {r['name']: r for r in previous['development']}
    checked, differences = 0, []
    for mode in ['long', 'ls']:
        name = 'alpha_4h_'+mode
        scenario_parts = {'base': [], 'double_cost': [], 'delay_4h': []}
        for fold, (start, end) in enumerate(WINDOWS[:4]):
            for scenario, options in [('base', {}), ('double_cost', {'multiplier': 2.}), ('delay_4h', {'delay': 16})]:
                scenario_results = {}
                for symbol in SYMBOLS:
                    m = make_market(symbol, bars[symbol], features[symbol], start, end)
                    target, allowed = orders(m, features[symbol], '4h', mode == 'ls', options.get('delay', 0))
                    result = evaluate(m, features[symbol], '4h', mode == 'ls', **options)
                    coarse = Market(symbol, m.times[::16], m.x[::16], m.prices[::16], m.marks[::16], m.funding[::16],
                                    m.lows.reshape(-1, 16).min(axis=1), m.highs.reshape(-1, 16).max(axis=1))
                    assert abs(coarse.funding.sum()-m.funding.sum()) < 1e-12
                    multiplier = options.get('multiplier', 1.)
                    ref = simulate(coarse, target[::16], 5*multiplier, 2*multiplier, trade_allowed=allowed[::16], rebalance_band=.05)
                    np.testing.assert_allclose(result['equity'][::16], ref['equity'], atol=2e-10, rtol=0)
                    for field in ['return', 'cost_paid_initial_equity', 'funding_paid_initial_equity']:
                        np.testing.assert_allclose(result[field], ref[field], atol=2e-10, rtol=0)
                    assert result['max_drawdown'] >= ref['max_drawdown']-1e-12
                    scenario_results[symbol] = result
                    checked += 1
                scenario_parts[scenario].append(portfolio(scenario_results)['returns'])
        for scenario, parts in scenario_parts.items():
            replay = metrics(np.concatenate(parts))
            differences.append({'name': name, 'scenario': scenario,
                'new15m_source_return': replay['return'], 'study007_return': old[name]['scenarios'][scenario]['return'],
                'return_difference_pp': 100*(replay['return']-old[name]['scenarios'][scenario]['return']),
                'new15m_drawdown': replay['max_drawdown'], 'study007_4h_drawdown': old[name]['scenarios'][scenario]['max_drawdown']})
    write_json(OUT/'legacy_4h_reconciliation.json', {'same_source_coarse_fine_accounting_checks': checked,
        'old_source_differences': differences, 'note': 'Old source prices differ; exact legacy-source equity parity is not claimed.'})
    return checked


def run():
    protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    report = json.loads((OUT/'report.json').read_text(encoding='utf-8'))
    assert protocol['specifications'] == specifications()
    assert protocol['manifest_sha256'] == sha256((OUT/'manifest.json').read_bytes()).hexdigest()
    for name, expected in protocol['source_hashes'].items():
        assert sha256(Path(name).read_bytes()).hexdigest() == expected, name
        assert sha256((OUT/'source'/name).read_bytes()).hexdigest() == expected, name
    manifest = json.loads((OUT/'manifest.json').read_text())
    for entry in manifest:
        assert sha256(Path(entry['path']).read_bytes()).hexdigest() == entry['sha256']
    bars, features, audit = dataset()
    check_4h_sources(bars, features)
    alternative = source_sensitivity(bars)
    alternative_features = {s: inputs(b) for s, b in alternative.items()}
    formulas = 0
    from .alphatrend import alpha_indicator
    for symbol, b in bars.items():
        for tf in TIMEFRAMES:
            agg = aggregate(b, tf)
            line, state = independent_alpha(agg)
            indicator = alpha_indicator(agg)
            np.testing.assert_allclose(indicator.line, line, atol=2e-9, rtol=0, equal_nan=True)
            np.testing.assert_array_equal(indicator.state, state)
            assert (indicator.line.loc['2024-01-01':] > 0).all()
            formulas += len(agg)
    print(f'Independent AlphaTrend formula bars verified: {formulas}', flush=True)
    legacy_checks = verify_legacy(bars, features)
    rows = {r['name']: r for r in report['strategies']}
    accounting_checks, seq, endpoints = 0, {}, []
    started = time.monotonic()
    for fold, (start, end) in enumerate(WINDOWS):
        markets = {s: make_market(s, bars[s], features[s], start, end) for s in SYMBOLS}
        endpoints.append(markets[SYMBOLS[0]].times[1:])
        for spec in specifications():
            for scenario, options in scenarios(spec['timeframe'])+[('source_sensitivity', {})]:
                use_features = alternative_features if scenario == 'source_sensitivity' else features
                use_markets = ({s: make_market(s, alternative[s], alternative_features[s], start, end) for s in SYMBOLS}
                               if scenario == 'source_sensitivity' else markets)
                results = {}
                saved = rows[spec['name']]['folds'][scenario][fold]
                for s, m in use_markets.items():
                    targets, allowed = orders(m, use_features[s], spec['timeframe'], spec['long_short'], options.get('delay', 0), spec.get('passive'))
                    fast = accounting(m, targets, allowed, options.get('multiplier', 1.))
                    results[s] = evaluate(m, use_features[s], spec['timeframe'], spec['long_short'], passive=spec.get('passive'), **options)
                    for field in ['return', 'max_drawdown', 'cost_paid_initial_equity', 'funding_paid_initial_equity']:
                        np.testing.assert_allclose(fast[field][0], results[s][field], atol=2e-10, rtol=0)
                        np.testing.assert_allclose(results[s][field], saved['by_symbol'][s][field], atol=2e-10, rtol=0)
                    accounting_checks += 1
                port = portfolio(results)
                seq.setdefault((spec['name'], scenario), []).append(port['returns'])
                for field in ['return', 'max_drawdown', 'sharpe']:
                    np.testing.assert_allclose(port[field], saved['portfolio'][field], atol=2e-10, rtol=0)
        print(f'Verified window{fold}: cumulative independent asset checks={accounting_checks}', flush=True)
    for group, sl in [('development', slice(0, 4)), ('recent', slice(4, 6))]:
        curves = pd.read_csv(OUT/f'{group}_equity.csv', index_col=0)
        months = pd.read_csv(OUT/f'{group}_monthly.csv', index_col=0)
        dates = pd.DatetimeIndex(np.concatenate([v.to_numpy() for v in endpoints[sl]]))
        for spec in specifications():
            for scenario in [s for s, _ in scenarios(spec['timeframe'])]+['source_sensitivity']:
                name = spec['name']
                returns = np.concatenate(seq[name, scenario][sl])
                replay = metrics(returns)
                key = name+'__'+scenario
                np.testing.assert_allclose(curves[key], replay['equity'], atol=2e-10, rtol=0)
                np.testing.assert_allclose(np.prod(1+months[key])-1, replay['return'], atol=2e-10, rtol=0)
                for field in ['return', 'max_drawdown', 'sharpe']:
                    np.testing.assert_allclose(replay[field], rows[name][group][scenario][field], atol=2e-10, rtol=0)
                daily = daily_returns(returns, dates)
                for year, values in daily.groupby(daily.index.year):
                    np.testing.assert_allclose(np.prod(1+values)-1, rows[name][group][scenario]['annual_returns'][str(year)], atol=2e-10, rtol=0)
    test = subprocess.run([sys.executable, '-m', 'pytest', '-q'], capture_output=True, text=True, check=True)
    (OUT/'pytest-output.txt').write_text(test.stdout, encoding='utf-8')
    import re
    tests = int(re.search(r'(\d+) passed', test.stdout)[1])
    write_json(OUT/'verification.json', {'tests_passed': tests, 'verified_source_archives': len(manifest),
        'formula_bars': formulas, 'independent_asset_accounting_checks': accounting_checks,
        'legacy4h_scenario_asset_replays': legacy_checks, 'same_source4h_boundary_equity_matches': True, 'original_study007_source_parity_claimed': False,
        'replayed_strategy_scenario_windows': accounting_checks//2,
        'annual_monthly_full_curves_verified': True, 'frozen_sources_verified': True,
        'pine_runtime_parity_tested': False, 'elapsed_seconds': time.monotonic()-started})
    print((OUT/'verification.json').read_text(), flush=True)


if __name__ == '__main__':
    run()
