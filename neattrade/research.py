import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import pickle
import random
import sys
import time

import neat
import numpy as np
import pandas as pd

from .data import download, load
from .market import BARS_YEAR, FEATURES, InsolventSimulation, make_market, score, simulate, summary

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = ('BTCUSDT', 'ETHUSDT')


def config(population=60):
    c = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                    neat.DefaultStagnation, str(ROOT / 'config-feedforward'))
    c.pop_size = population
    return c


def batch_activate(net, x):
    """Evaluate independent rows using exactly NEAT's sum/tanh (2.5 scale)."""
    if x.ndim != 2 or x.shape[1] != len(net.input_nodes) or len(net.output_nodes) != 1:
        raise ValueError('Feature count must match NEAT inputs, with exactly one output')
    values = {k: x[:, i] for i, k in enumerate(net.input_nodes)}
    for node, activation, aggregate, bias, response, links in net.node_evals:
        if activation.__name__ != 'tanh_activation' or aggregate.__name__ != 'sum_aggregation':
            raise ValueError('Batch evaluator only supports configured sum/tanh')
        total = np.zeros(len(x))
        for source, weight in links:
            total += values.get(source, np.zeros(len(x))) * weight
        values[node] = np.tanh(np.clip(2.5 * (bias + response * total), -60, 60))
    return values.get(net.output_nodes[0], np.zeros(len(x)))


def risk_size(m):
    # Annualized 20% volatility target; lagged volatility, no leverage.
    return np.minimum(0.5, 0.20 / np.maximum(m.x[:, 6], 0.10))


def positions(candidate, m, c):
    kind = candidate['kind']
    if kind == 'cash':
        return np.zeros(len(m.x))
    if kind == 'long':
        return np.full(len(m.x), 0.5)
    if kind == 'trend':
        signal = np.sign(m.x[:, candidate['feature']])
        if candidate.get('long_only'):
            signal = np.maximum(signal, 0)
        return signal * risk_size(m)
    if kind == 'neat':
        net = neat.nn.FeedForwardNetwork.create(candidate['genome'], c)
        raw = batch_activate(net, m.x)
        # Fixed dead zone and 24h decision cadence reduce noise-driven turnover.
        discrete = np.where(raw > 0.35, 1., np.where(raw < -0.35, -1., 0.))
        anchors = np.arange(len(raw)) - (m.times[:-1].hour.to_numpy() // 4)
        anchors = np.maximum(anchors, 0)
        return discrete[anchors] * risk_size(m)[anchors]
    if kind == 'ensemble':
        return np.mean([positions(member, m, c) for member in candidate['members']], axis=0)
    raise ValueError(kind)


def evaluate(candidate, markets, c, fee_bps=5, slippage_bps=2, delay=0, no_funding=False):
    assert_aligned(markets)
    results = {}
    for m in markets:
        p = positions(candidate, m, c)
        if delay:
            p = np.r_[np.zeros(delay), p[:-delay]]
        market = replace(m, funding=np.zeros_like(m.funding)) if no_funding else m
        results[m.symbol] = simulate(market, p, fee_bps, slippage_bps)
    return results


def robust_score(results):
    s = np.array([score(r) for r in results.values()])
    return float(s.mean() - 0.5 * s.std())


def assert_aligned(markets):
    if not markets or any(not m.times.equals(markets[0].times) for m in markets[1:]):
        raise ValueError('Portfolio markets must have identical execution timestamps')


class Progress(neat.reporting.BaseReporter):
    def __init__(self, seed):
        self.seed, self.generation = seed, 0

    def post_evaluate(self, config, population, species, best):
        self.generation += 1
        if self.generation == 1 or self.generation % 5 == 0:
            print(f'seed={self.seed} generation={self.generation} train fitness={best.fitness:.5f}', flush=True)


def train(markets, c, seed, generations):
    # NEAT config holds mutable node-index state: use a fresh instance per seed.
    c = config(c.pop_size)
    random.seed(seed)
    np.random.seed(seed)
    p = neat.Population(c, seed=seed)
    p.add_reporter(Progress(seed))

    def fitness(genomes, conf):
        for _, g in genomes:
            candidate = {'kind': 'neat', 'genome': g}
            scores = []
            for m in markets:
                try:
                    scores.append(score(simulate(m, positions(candidate, m, conf))))
                except InsolventSimulation:
                    scores.append(-100.)
            g.fitness = float(np.mean(scores) - 0.5 * np.std(scores)) - 0.0001 * len(g.connections)

    winner = p.run(fitness, generations)
    return {'name': f'neat_seed_{seed}', 'kind': 'neat', 'seed': seed, 'genome': winner}


def portfolio(results):
    # Equal initial capital sleeves; no inter-symbol rebalance.
    eq = np.mean([r['equity'] for r in results.values()], axis=0)
    r = eq[1:] / eq[:-1] - 1
    return {'return': float(eq[-1] - 1), 'max_drawdown': float((1 - eq / np.maximum.accumulate(eq)).max()),
            'sharpe': float(r.mean() / r.std() * np.sqrt(BARS_YEAR)) if r.std() > 1e-12 else 0.,
            'equity': eq, 'returns': r}


def bootstrap_ci(returns, seed=31415, samples=1000, block=42):
    """Paired portfolio circular block bootstrap; descriptive, not selection-adjusted."""
    rng = np.random.default_rng(seed)
    n = len(returns)
    starts = rng.integers(0, n, size=(samples, int(np.ceil(n / block))))
    indices = ((starts[:, :, None] + np.arange(block)) % n).reshape(samples, -1)[:, :n]
    draws = np.prod(1 + returns[indices], axis=1) - 1
    return np.quantile(draws, [0.025, 0.975]).tolist()


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'selection.json').exists() or (output / 'protocol.json').exists():
        raise ValueError('Use a fresh output directory; existing research runs are immutable')
    c = config(args.population)
    protocol = {'symbols': SYMBOLS, 'train': ['2022-01-01', '2024-01-01'],
                'validation': ['2024-01-01', '2025-01-01'], 'test': ['2025-01-01', '2026-01-01'],
                'seeds': [11, 29, 47], 'generations': args.generations, 'population': args.population,
                'fee_bps_per_side_assumption': 5, 'slippage_bps_per_side_assumption': 2,
                'max_notional_equity': 0.5, 'selection': 'mean log net growth minus 1.5 drawdown, minus 0.5 cross-symbol score std; cash eligible',
                'test_gate': 'positive return on both assets, portfolio Sharpe >= 0.5, drawdown <= 20%, positive portfolio return at double costs and delayed execution, outperform 50% long baseline',
                'warning': 'Test is descriptive and may reject an edge, not prove future profitability',
                'python': sys.version, 'neat': neat.__version__, 'numpy': np.__version__, 'pandas': pd.__version__,
                'source_hashes': {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / 'neattrade').glob('*.py'))},
                'config_hash': sha256((ROOT / 'config-feedforward').read_bytes()).hexdigest(),
                'manifest_hash': sha256(Path(args.data, 'manifest.json').read_bytes()).hexdigest()}
    start = time.monotonic()
    bars = {s: load(s, args.data) for s in SYMBOLS}
    protocol['manifest_hash'] = sha256(Path(args.data, 'manifest.json').read_bytes()).hexdigest()
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    # All training fitness periods terminate before validation. No supervised forward labels.
    train_markets = [make_market(f'{s}_{year}', b, f'{year}-01-01', f'{year+1}-01-01') for s, b in bars.items() for year in [2022, 2023]]
    validation = [make_market(s, b, *protocol['validation']) for s, b in bars.items()]
    winners = [train(train_markets, c, seed, args.generations) for seed in protocol['seeds']]
    candidates = [{'name': 'cash', 'kind': 'cash'}, {'name': 'long_50pct', 'kind': 'long'}]
    candidates += [{'name': f'trend_{n}_{mode}', 'kind': 'trend', 'feature': i, 'long_only': mode == 'long_only'}
                   for n, i in [(42, 4), (180, 5)] for mode in ['both', 'long_only']]
    candidates += winners + [{'name': 'neat_ensemble', 'kind': 'ensemble', 'members': winners}]
    rows = []
    for candidate in candidates:
        r = evaluate(candidate, validation, c)
        rows.append({'name': candidate['name'], 'selection_score': robust_score(r),
                     'by_symbol': {s: summary(v) for s, v in r.items()}, 'portfolio': summary(portfolio(r))})
    selected_row = max(rows, key=lambda x: x['selection_score'])
    selected = next(x for x in candidates if x['name'] == selected_row['name'])
    # Persist selection before accessing test metrics. All candidates are logged, no hidden search.
    selection = {'selected': selected['name'], 'validation_trials': len(rows), 'validation': rows,
                 'genome_evaluations_upper_bound': len(winners) * args.generations * args.population}
    (output / 'selection.json').write_text(json.dumps(selection, indent=2))
    with (output / 'models.pkl').open('wb') as f:
        pickle.dump({'selected': selected, 'candidates': candidates, 'config': c}, f)
    print(f'Frozen selection: {selected["name"]}', flush=True)
    test = [make_market(s, b, *protocol['test']) for s, b in bars.items()]
    scenarios = {}
    curves = pd.DataFrame(index=test[0].times)
    for label, options in [('base', {}), ('double_cost', {'fee_bps': 10, 'slippage_bps': 4}),
                           ('delay_4h', {'delay': 1}), ('no_funding', {'no_funding': True})]:
        r = evaluate(selected, test, c, **options)
        port = portfolio(r)
        scenarios[label] = {'by_symbol': {s: summary(v) for s, v in r.items()},
                            'portfolio': summary(port)}
        curves[label] = port['equity']
        if label == 'base':
            scenarios[label]['portfolio']['block_bootstrap_return_ci95'] = bootstrap_ci(port['returns'])
            for s, v in r.items():
                curves[s] = v['equity']
    # Predeclared comparators: long baseline and fixed ensemble; never reselect using test.
    comparators = {}
    for candidate in [candidates[1], candidates[-1]]:
        r = evaluate(candidate, test, c)
        comparators[candidate['name']] = {'by_symbol': {s: summary(v) for s, v in r.items()}, 'portfolio': summary(portfolio(r))}
        curves[candidate['name']] = portfolio(r)['equity']
    b = scenarios['base']
    gates = {'positive_each_symbol': all(v['return'] > 0 for v in b['by_symbol'].values()),
             'sharpe_at_least_0_5': b['portfolio']['sharpe'] >= 0.5,
             'drawdown_below_20pct': b['portfolio']['max_drawdown'] <= 0.2,
             'double_cost_positive': scenarios['double_cost']['portfolio']['return'] > 0,
             'delay_positive': scenarios['delay_4h']['portfolio']['return'] > 0,
             'beats_long': b['portfolio']['return'] > comparators['long_50pct']['portfolio']['return']}
    report = {'selected': selected['name'], 'test': scenarios, 'comparators': comparators, 'gates': gates,
              'research_gate_pass': all(gates.values()), 'elapsed_seconds': time.monotonic() - start,
              'data_coverage': {s: {'rows': len(v), 'start': str(v.index[0]), 'end': str(v.index[-1])} for s, v in bars.items()},
              'training_periods': {m.symbol: {'start': str(m.times[0]), 'end': str(m.times[-1]), 'bars': len(m.x)} for m in train_markets}}
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    curves.to_csv(output / 'equity.csv')
    print(json.dumps(report, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description='Binance USD-M offline NEAT research')
    p.add_argument('command', choices=['download', 'run'])
    p.add_argument('--data', default='data')
    p.add_argument('--output', default='artifacts/experiment_001')
    p.add_argument('--generations', type=int, default=20)
    p.add_argument('--population', type=int, default=60)
    args = p.parse_args()
    if args.command == 'download':
        download(args.data)
    else:
        if args.generations < 1 or args.population < 2:
            p.error('generations >= 1 and population >= 2 required')
        run(args)
