"""Second preregistered study: weekly NEAT trend filter, fresh 2026 holdout.

2025 was already inspected in study 001 and is explicitly validation here.
The original study is retained; this is a new hypothesis, not a retest of 2025.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
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

from .data import load
from .market import InsolventSimulation, make_market, score, simulate, summary
from .research import ROOT, SYMBOLS, Progress, assert_aligned, batch_activate, bootstrap_ci, config, portfolio, risk_size, robust_score


def schedule(m):
    # Monday 00:00 UTC only, plus initial entry. Hold fixed units between decisions.
    times = m.times[:-1]
    allowed = np.asarray((times.dayofweek == 0) & (times.hour == 0))
    allowed[0] = True
    return allowed


def desired(candidate, m, c):
    kind = candidate['kind']
    if kind == 'cash':
        return np.zeros(len(m.x))
    if kind == 'long':
        return np.full(len(m.x), 0.5)
    trend = np.sign(m.x[:, 5])
    if candidate.get('long_only'):
        trend = np.maximum(trend, 0)
    if kind == 'trend':
        trend = np.sign(m.x[:, candidate['feature']])
        if candidate.get('long_only'):
            trend = np.maximum(trend, 0)
        return trend * risk_size(m)
    if kind == 'filter':
        net = neat.nn.FeedForwardNetwork.create(candidate['genome'], c)
        raw = batch_activate(net, m.x)
        # NEAT controls permission/size of a slow trend strategy instead of freely flipping direction.
        confidence = np.where(raw < -0.35, 0., np.where(raw > 0.35, 1., 0.5))
        return trend * risk_size(m) * confidence
    if kind == 'ensemble':
        return np.mean([desired(member, m, c) for member in candidate['members']], axis=0)
    raise ValueError(kind)


def evaluate(candidate, markets, c, multiplier=1, delay=0, no_funding=False):
    assert_aligned(markets)
    results = {}
    for m in markets:
        p, allowed = desired(candidate, m, c), schedule(m)
        if delay:
            p = np.r_[np.zeros(delay), p[:-delay]]
            allowed = np.r_[np.zeros(delay, dtype=bool), allowed[:-delay]]
        market = replace(m, funding=np.zeros_like(m.funding)) if no_funding else m
        results[m.symbol] = simulate(market, p, 5 * multiplier, 2 * multiplier, trade_allowed=allowed)
    return results


def train_one(job):
    markets, population, generations, seed = job
    c = config(population)
    random.seed(seed)
    np.random.seed(seed)
    pop = neat.Population(c, seed=seed)
    pop.add_reporter(Progress(seed))

    def fitness(genomes, conf):
        for _, g in genomes:
            candidate = {'kind': 'filter', 'genome': g}
            values = []
            for m in markets:
                try:
                    values.append(score(simulate(m, desired(candidate, m, conf), trade_allowed=schedule(m))))
                except InsolventSimulation:
                    values.append(-100.)
            g.fitness = float(np.mean(values) - 0.5 * np.std(values)) - 0.0001 * len(g.connections)

    winner = pop.run(fitness, generations)
    return {'kind': 'filter', 'name': f'neat_filter_seed_{seed}', 'seed': seed, 'genome': winner}


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        raise ValueError('Use a fresh output directory')
    c = config(args.population)
    bars = {s: load(s, args.data) for s in SYMBOLS}
    protocol = {'study': '002_weekly_neat_filter', 'prior_study': '001',
                'prior_results_known': '2024 validation and 2025 test inspected; 2025 is now development validation, never claimed unseen again',
                'train': ['2022-01-01', '2025-01-01'], 'validation': ['2025-01-01', '2026-01-01'],
                'test': ['2026-01-01', '2026-08-31 20:00'],
                'test_end_note': 'Excludes final 4h candle of August to retain a known terminal open and fully archived funding',
                'symbols': SYMBOLS, 'seeds': [11, 29, 47], 'population': args.population, 'generations': args.generations,
                'changes': ['six half-year training blocks per asset', 'weekly fixed-unit holding', 'NEAT gates/sizes 180-bar EMA trend; no free directional forecasting'],
                'selection_rule': 'same risk-adjusted cross-symbol score as study 001; cash remains eligible',
                'fee_bps': 5, 'slippage_bps': 2, 'max_entry_exposure': 0.5,
                'test_gate': 'both assets positive; portfolio Sharpe >=0.5; drawdown<=20%; double cost and delayed return positive; beat weekly 50% long',
                'comparators': ['weekly_long_50pct', 'neat_filter_ensemble'],
                'versions': {'python': sys.version, 'neat': neat.__version__, 'numpy': np.__version__, 'pandas': pd.__version__},
                'source_hashes': {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'neattrade').glob('*.py'))},
                'config_hash': sha256((ROOT/'config-feedforward').read_bytes()).hexdigest(),
                'manifest_hash': sha256(Path(args.data, 'manifest.json').read_bytes()).hexdigest()}
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    start = time.monotonic()
    boundaries = pd.date_range('2022-01-01', '2025-01-01', freq='6MS')
    training = [make_market(f'{s}_{a:%Y-%m}', b, str(a.date()), str(z.date()))
                for s, b in bars.items() for a, z in zip(boundaries[:-1], boundaries[1:])]
    jobs = [(training, args.population, args.generations, seed) for seed in protocol['seeds']]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        winners = list(pool.map(train_one, jobs))
    candidates = [{'name': 'cash', 'kind': 'cash'}, {'name': 'weekly_long_50pct', 'kind': 'long'}]
    candidates += [{'name': f'weekly_trend_{n}_{mode}', 'kind': 'trend', 'feature': i, 'long_only': mode == 'long_only'}
                   for n, i in [(42, 4), (180, 5)] for mode in ['both', 'long_only']]
    candidates += winners + [{'name': 'neat_filter_ensemble', 'kind': 'ensemble', 'members': winners}]
    validation = [make_market(s, b, *protocol['validation']) for s, b in bars.items()]
    rows = []
    for candidate in candidates:
        r = evaluate(candidate, validation, c)
        rows.append({'name': candidate['name'], 'selection_score': robust_score(r),
                     'by_symbol': {s: summary(v) for s, v in r.items()}, 'portfolio': summary(portfolio(r))})
    selected_name = max(rows, key=lambda r: r['selection_score'])['name']
    selected = next(x for x in candidates if x['name'] == selected_name)
    (out/'selection.json').write_text(json.dumps({'selected': selected_name, 'validation': rows, 'validation_trials': len(rows),
                    'genome_evaluations_upper_bound': args.population * args.generations * 3}, indent=2))
    with (out/'models.pkl').open('wb') as f:
        pickle.dump({'selected': selected, 'candidates': candidates, 'config': c}, f)
    print(f'Frozen v2 selection: {selected_name}', flush=True)
    test = [make_market(s, b, *protocol['test']) for s, b in bars.items()]
    curves = pd.DataFrame(index=test[0].times)
    scenarios = {}
    for label, options in [('base', {}), ('double_cost', {'multiplier': 2}), ('delay_4h', {'delay': 1}), ('no_funding', {'no_funding': True})]:
        r = evaluate(selected, test, c, **options)
        port = portfolio(r)
        scenarios[label] = {'by_symbol': {s: summary(v) for s, v in r.items()}, 'portfolio': summary(port)}
        curves[label] = port['equity']
        if label == 'base':
            scenarios[label]['portfolio']['block_bootstrap_return_ci95'] = bootstrap_ci(port['returns'])
            for s, v in r.items():
                curves[s] = v['equity']
    comparators = {}
    for candidate in [candidates[1], candidates[-1]]:
        r = evaluate(candidate, test, c)
        comparators[candidate['name']] = {'by_symbol': {s: summary(v) for s, v in r.items()}, 'portfolio': summary(portfolio(r))}
        curves[candidate['name']] = portfolio(r)['equity']
    base = scenarios['base']
    gates = {'positive_each_symbol': all(v['return'] > 0 for v in base['by_symbol'].values()),
             'sharpe_at_least_0_5': base['portfolio']['sharpe'] >= 0.5,
             'drawdown_below_20pct': base['portfolio']['max_drawdown'] <= 0.2,
             'double_cost_positive': scenarios['double_cost']['portfolio']['return'] > 0,
             'delay_positive': scenarios['delay_4h']['portfolio']['return'] > 0,
             'beats_long': base['portfolio']['return'] > comparators['weekly_long_50pct']['portfolio']['return']}
    report = {'selected': selected_name, 'test': scenarios, 'comparators': comparators, 'gates': gates,
              'research_gate_pass': all(gates.values()), 'elapsed_seconds': time.monotonic()-start,
              'training_periods': {m.symbol: {'start': str(m.times[0]), 'end': str(m.times[-1]), 'bars': len(m.x)} for m in training}}
    (out/'report.json').write_text(json.dumps(report, indent=2))
    curves.to_csv(out/'equity.csv')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data')
    p.add_argument('--output', default='artifacts/experiment_002')
    p.add_argument('--population', type=int, default=80)
    p.add_argument('--generations', type=int, default=40)
    p.add_argument('--workers', type=int, default=3)
    args = p.parse_args()
    if args.population < 2 or args.generations < 1 or args.workers < 1:
        p.error('population>=2, generations>=1, workers>=1 required')
    run(args)
