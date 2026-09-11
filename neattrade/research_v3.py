"""Study 003: paired fitness ablation and expanding walk-forward development.

Previously inspected BTC/ETH data is never relabelled as a fresh holdout.
Unseen SOL/BNB transfer data is accessed only after a development candidate passes.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import pickle
import shutil
import time

import neat
import numpy as np
import pandas as pd

from .batch import fitness_accounting, growth_objective, legacy_objective, prepare
from .data import download, load
from .market import BARS_YEAR, InsolventSimulation, make_market, simulate, summary
from .research import ROOT, SYMBOLS, assert_aligned, batch_activate, bootstrap_ci, config, portfolio

SEEDS = [11, 29, 47]
FOLDS = [('2024-01-01', '2024-07-01'), ('2024-07-01', '2025-01-01'),
         ('2025-01-01', '2025-07-01'), ('2025-07-01', '2026-01-01')]


def target_from_output(raw, x):
    signal = np.where(raw > .35, 1., np.where(raw < -.35, -1., 0.))
    return signal * np.minimum(.5, .20/np.maximum(x[:, 6], .10))


def daily_schedule(m):
    mask = np.asarray(m.times[:-1].hour == 0)
    mask[0] = True
    return mask


def desired(candidate, m, c):
    kind = candidate['kind']
    if kind == 'cash':
        return np.zeros(len(m.x))
    if kind == 'long':
        return np.full(len(m.x), .5)
    if kind == 'trend':
        signal = np.sign(m.x[:, candidate['feature']])
        if candidate['long_only']:
            signal = np.maximum(signal, 0)
        return signal*np.minimum(.5, .20/np.maximum(m.x[:, 6], .10))
    if kind == 'neat':
        return target_from_output(batch_activate(neat.nn.FeedForwardNetwork.create(candidate['genome'], c), m.x), m.x)
    if kind == 'ensemble':
        return np.mean([desired(member, m, c) for member in candidate['members']], axis=0)
    raise ValueError(kind)


def evaluate(candidate, markets, c, multiplier=1, delay=0):
    assert_aligned(markets)
    result = {}
    for m in markets:
        p, allowed = desired(candidate, m, c), daily_schedule(m)
        if delay:
            p = np.r_[np.zeros(delay), p[:-delay]]
            allowed = np.r_[np.zeros(delay, dtype=bool), allowed[:-delay]]
        result[m.symbol] = simulate(m, p, 5*multiplier, 2*multiplier, trade_allowed=allowed)
        executed_targets = p[allowed]
        result[m.symbol]['decision_long_fraction'] = float(np.mean(executed_targets > 1e-12))
        result[m.symbol]['decision_short_fraction'] = float(np.mean(executed_targets < -1e-12))
        result[m.symbol]['decision_flat_fraction'] = float(np.mean(np.abs(executed_targets) <= 1e-12))
    return result


def training_markets(bars, validation_start):
    end = pd.Timestamp(validation_start) - pd.Timedelta(days=7)
    boundaries = list(pd.date_range('2022-01-01', end, freq='6MS')) + [end]
    markets = [make_market(f'{symbol}_{a:%Y-%m}', b, str(a.date()), str(z.date()))
               for symbol, b in bars.items() for a, z in zip(boundaries[:-1], boundaries[1:]) if z > a]
    assert all(m.times[-1] <= end.tz_localize('UTC') for m in markets)
    return markets


class Trace(neat.reporting.BaseReporter):
    def __init__(self, name):
        self.name = name
        self.rows = []

    def post_evaluate(self, conf, population, species, best):
        row = {'generation': len(self.rows)+1, 'best_fitness': best.fitness,
               'mean_fitness': float(np.mean([g.fitness for g in population.values()])),
               'cash_fraction': float(np.mean([g.research_cash for g in population.values()])),
               'best_cash': best.research_cash, 'best_nodes': len(best.nodes),
               'best_connections': len(best.connections), 'species': len(species.species)}
        self.rows.append(row)
        if row['generation'] == 1 or row['generation'] % 10 == 0:
            print(f'{self.name}: gen={row["generation"]} fitness={best.fitness:.4f} cash={row["cash_fraction"]:.0%}', flush=True)


def train_one(job):
    markets, population, generations, seed, objective, job_name, output = job
    path = Path(output)/'jobs'/f'{job_name}.pkl'
    if path.exists():
        with path.open('rb') as f:
            return pickle.load(f)
    prepared = [prepare(m) for m in markets]
    c = config(population)
    p = neat.Population(c, seed=seed)
    trace = Trace(job_name)
    p.add_reporter(trace)

    def fitness(genomes, conf):
        networks = [neat.nn.FeedForwardNetwork.create(g, conf) for _, g in genomes]
        results = []
        cash = np.ones(len(genomes), dtype=bool)
        for m in prepared:
            targets = np.stack([target_from_output(batch_activate(net, m.x), m.x) for net in networks])
            cash &= np.all(targets == 0, axis=1)
            results.append(fitness_accounting(m, targets))
        values = growth_objective(results) if objective == 'growth' else legacy_objective(results)
        for i, (_, g) in enumerate(genomes):
            g.fitness = float(values[i] - .0001*len(g.connections))
            g.research_cash = bool(cash[i])

    winner = p.run(fitness, generations)
    candidate = {'name': job_name, 'kind': 'neat', 'genome': winner, 'seed': seed,
                 'objective': objective, 'train_start': str(markets[0].times[0]),
                 'train_end': str(max(m.times[-1] for m in markets)), 'train_cash': winner.research_cash}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        pickle.dump(candidate, f)
    tmp.replace(path)
    pd.DataFrame(trace.rows).to_csv(path.with_suffix('.csv'), index=False)
    return candidate


def baselines():
    return [{'name': 'cash', 'kind': 'cash'}, {'name': 'daily_long_50pct', 'kind': 'long'}] + [
        {'name': f'daily_trend_{n}_{mode}', 'kind': 'trend', 'feature': i, 'long_only': mode == 'long_only'}
        for n, i in [(42, 4), (180, 5)] for mode in ['both', 'long_only']]


def stitched_metrics(returns):
    r = np.concatenate(returns)
    eq = np.r_[1., np.cumprod(1+r)]
    return {'return': float(eq[-1]-1), 'max_drawdown': float(np.max(1-eq/np.maximum.accumulate(eq))),
            'sharpe': float(r.mean()/r.std()*np.sqrt(BARS_YEAR)) if r.std() > 1e-12 else 0.,
            'equity': eq, 'returns': r}


def development_gate(row):
    base = row['scenarios']['base']
    return {'positive_net': base['return'] > 0,
            'drawdown_at_most_20pct': base['max_drawdown'] <= .20,
            'sharpe_at_least_0_5': base['sharpe'] >= .5,
            'three_positive_folds': sum(v['portfolio']['return'] > 0 for v in row['folds']) >= 3,
            'positive_each_asset': all(v > 0 for v in row['asset_compounded_returns'].values()),
            'double_cost_positive': row['scenarios']['double_cost']['return'] > 0,
            'delay_positive': row['scenarios']['delay_4h']['return'] > 0}


def snapshot(out):
    hashes = {}
    for p in sorted((ROOT/'neattrade').glob('*.py')):
        target = out/'source'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
        hashes[str(p.relative_to(ROOT))] = sha256(p.read_bytes()).hexdigest()
    for name in ['config-feedforward', 'requirements.txt']:
        shutil.copyfile(ROOT/name, out/'source'/name)
        hashes[name] = sha256((ROOT/name).read_bytes()).hexdigest()
    return hashes


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'report.json').exists():
        raise ValueError('Completed study is immutable')
    bars = {s: load(s, args.data) for s in SYMBOLS}
    if (out/'protocol.json').exists():
        if not args.resume:
            raise ValueError('Use --resume to reuse completed training jobs')
        protocol = json.loads((out/'protocol.json').read_text())
        if args.population != protocol['population'] or args.generations != protocol['generations']:
            raise ValueError('Resume parameters differ from frozen protocol')
        for file, expected in protocol['source_hashes'].items():
            if sha256((ROOT/file).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Resume source differs: {file}')
        if sha256(Path(args.data, 'manifest.json').read_bytes()).hexdigest() != protocol['data_manifest_hash']:
            raise ValueError('Resume data manifest differs')
    else:
        protocol = {'study': '003', 'seeds': SEEDS, 'population': args.population, 'generations': args.generations,
                    'folds': FOLDS, 'train_start': '2022-01-01', 'train_embargo_days': 7,
                    'paired_comparison': 'same daily direct NEAT class and seeds; only fitness differs: legacy versus growth',
                    'growth_fitness': 'duration-weighted mean annual log net growth minus 2*max(drawdown-0.20,0); complexity penalty 0.0001 per connection; no activity reward',
                    'legacy_fitness': 'mean(log net growth -1.5*drawdown) minus 0.5*std; same complexity penalty',
                    'development_known': 'All BTC/ETH 2022-Aug2026 previously available/inspected; all rolling fold results are development, not fresh confirmation',
                    'selection': 'cash if no candidate passes all development gates; otherwise maximum median half-year portfolio return, cumulative return breaks ties; only seed ensembles are candidates',
                    'development_gates': 'positive cumulative return on each asset; portfolio Sharpe>=0.5; drawdown<=20%; >=3/4 positive folds; double cost and 4h delay positive',
                    'transfer': {'symbols': ['SOLUSDT', 'BNBUSDT'], 'period': ['2026-01-01', '2026-08-31 20:00'],
                                 'access_rule': 'only after development candidate passes and final fit/selection is saved',
                                 'limitation': 'correlated cross-asset transfer in previously observed market regime; not an independent time holdout',
                                 'gates': 'positive each asset at base/double cost/delay; portfolio drawdown<=20%; outperform all five matched active baselines'},
                    'sizing': '20% lagged annual volatility target, max 0.5 initial exposure, fixed units between daily UTC midnight decisions',
                    'fee_bps_per_side': 5, 'slippage_bps_per_side': 2,
                    'neat_version': neat.__version__, 'numpy_version': np.__version__, 'pandas_version': pd.__version__,
                    'source_hashes': snapshot(out),
                    'data_manifest_hash': sha256(Path(args.data, 'manifest.json').read_bytes()).hexdigest()}
        (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
        shutil.copyfile(Path(args.data, 'manifest.json'), out/'data-manifest.json')
    start = time.monotonic()
    jobs = []
    for fold, (a, z) in enumerate(FOLDS):
        training = training_markets(bars, a)
        for objective in ['legacy', 'growth']:
            for seed in SEEDS:
                jobs.append((training, args.population, args.generations, seed, objective,
                             f'fold{fold}_{objective}_{seed}', str(out)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        winners = list(pool.map(train_one, jobs))
    models = {c['name']: c for c in winners}
    c = config(args.population)
    logs, sequences, asset_sequences = {}, {}, {}
    curve_dates = []
    per_seed = []
    for fold, (a, z) in enumerate(FOLDS):
        markets = [make_market(s, b, a, z) for s, b in bars.items()]
        curve_dates.extend(markets[0].times[1:])
        candidates = baselines() + [{'name': f'neat_{objective}', 'kind': 'ensemble',
                      'members': [models[f'fold{fold}_{objective}_{seed}'] for seed in SEEDS]} for objective in ['legacy', 'growth']]
        for candidate in candidates:
            name = candidate['name']
            for scenario, opts in [('base', {}), ('double_cost', {'multiplier': 2}), ('delay_4h', {'delay': 1})]:
                r = evaluate(candidate, markets, c, **opts)
                port = portfolio(r)
                sequences.setdefault((name, scenario), []).append(port['returns'])
                if scenario == 'base':
                    logs.setdefault(name, []).append({'period': [a, z], 'portfolio': summary(port),
                                                      'by_symbol': {s: summary(v) for s, v in r.items()}})
                    for symbol, v in r.items():
                        asset_sequences.setdefault((name, symbol), []).append(v['returns'])
        for objective in ['legacy', 'growth']:
            for seed in SEEDS:
                member = models[f'fold{fold}_{objective}_{seed}']
                r = evaluate(member, markets, c)
                per_seed.append({'fold': fold, 'objective': objective, 'seed': seed,
                                 'train_cash': member['train_cash'], 'nodes': len(member['genome'].nodes),
                                 'connections': len(member['genome'].connections),
                                 'portfolio': summary(portfolio(r)), 'by_symbol': {s: summary(v) for s, v in r.items()}})
    rows = []
    curves = pd.DataFrame(index=pd.DatetimeIndex([pd.Timestamp(FOLDS[0][0], tz='UTC')]+curve_dates))
    for name, folds in logs.items():
        row = {'name': name, 'folds': folds,
               'scenarios': {scenario: summary(stitched_metrics(sequences[name, scenario])) for scenario in ['base', 'double_cost', 'delay_4h']},
               'asset_compounded_returns': {s: stitched_metrics(asset_sequences[name, s])['return'] for s in SYMBOLS}}
        row['gates'] = development_gate(row)
        row['development_pass'] = all(row['gates'].values())
        row['median_fold_return'] = float(np.median([f['portfolio']['return'] for f in folds]))
        row['descriptive_bootstrap_ci95'] = bootstrap_ci(stitched_metrics(sequences[name, 'base'])['returns'])
        rows.append(row)
        curves[name] = stitched_metrics(sequences[name, 'base'])['equity']
    eligible = [r for r in rows if r['development_pass']]
    selected_name = max(eligible, key=lambda r: (r['median_fold_return'], r['scenarios']['base']['return']))['name'] if eligible else 'cash'
    selection = {'selected': selected_name, 'development_pass': bool(eligible), 'development': rows,
                 'per_seed_diagnostics': per_seed, 'validation_candidates': len(rows),
                 'genome_evaluations_upper_bound': len(jobs)*args.population*args.generations,
                 'note': 'All fold metrics are development. No individual seed selection.'}
    (out/'selection.json').write_text(json.dumps(selection, indent=2))
    curves.to_csv(out/'development_equity.csv')
    print(f'Frozen development selection: {selected_name}', flush=True)
    report = {'selected': selected_name, 'development_pass': bool(eligible), 'transfer_accessed': False,
              'research_gate_pass': False, 'elapsed_seconds': time.monotonic()-start}
    if eligible:
        if selected_name.startswith('neat_'):
            objective = selected_name.removeprefix('neat_')
            training = training_markets(bars, '2026-01-01')
            final_jobs = [(training, args.population, args.generations, seed, objective,
                           f'final_{objective}_{seed}', str(out)) for seed in SEEDS]
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                final_members = list(pool.map(train_one, final_jobs))
            candidate = {'name': selected_name, 'kind': 'ensemble', 'members': final_members}
        else:
            candidate = next(x for x in baselines() if x['name'] == selected_name)
        with (out/'frozen_candidate.pkl').open('wb') as f:
            pickle.dump({'candidate': candidate, 'config': c}, f)
        transfer_root = out/'transfer_data'
        download(str(transfer_root), start='2025-11', end='2026-08', symbols=protocol['transfer']['symbols'])
        transfer_bars = {s: load(s, str(transfer_root)) for s in protocol['transfer']['symbols']}
        markets = [make_market(s, b, *protocol['transfer']['period']) for s, b in transfer_bars.items()]
        report['transfer_accessed'] = True
        report['transfer'] = {}
        for scenario, opts in [('base', {}), ('double_cost', {'multiplier': 2}), ('delay_4h', {'delay': 1})]:
            r = evaluate(candidate, markets, c, **opts)
            report['transfer'][scenario] = {'by_symbol': {s: summary(v) for s, v in r.items()}, 'portfolio': summary(portfolio(r))}
        report['transfer_baselines'] = {b['name']: summary(portfolio(evaluate(b, markets, c))) for b in baselines() if b['kind'] != 'cash'}
        recent = [make_market(s, b, *protocol['transfer']['period']) for s, b in bars.items()]
        report['known_btc_eth_recent_context'] = {s: summary(v) for s, v in evaluate(candidate, recent, c).items()}
        report['research_gate_pass'] = (all(v['return'] > 0 for scenario in report['transfer'].values() for v in scenario['by_symbol'].values())
            and report['transfer']['base']['portfolio']['max_drawdown'] <= .20
            and report['transfer']['base']['portfolio']['return'] > max(b['return'] for b in report['transfer_baselines'].values()))
    report['elapsed_seconds'] = time.monotonic()-start
    (out/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data')
    p.add_argument('--output', default='artifacts/experiment_003')
    p.add_argument('--population', type=int, default=64)
    p.add_argument('--generations', type=int, default=30)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.population < 2 or args.generations < 1 or args.workers < 1:
        p.error('population>=2, generations>=1, workers>=1 required')
    run(args)
