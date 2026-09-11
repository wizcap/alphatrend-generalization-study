"""Frozen AlphaTrend-only and seven indicator-family augmentation comparisons."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import pickle
import shutil
import time

import neat
import numpy as np
import pandas as pd

from .alphatrend import (ALPHA_FEATURES, FEATURE_NAMES, alpha_features,
                        alpha_targets, gated_targets, select_inputs, specifications)
from .batch import fitness_accounting, prepare
from .features_v5 import ALL_FEATURES, load_dataset, source_lag, with_features
from .indicator_types import TYPES, fixed_risk_scale
from .market import simulate, summary
from .overlay import baseline_targets, incremental_objective
from .research import assert_aligned, batch_activate, portfolio
from .research_v3 import FOLDS, SEEDS, Trace, daily_schedule, development_gate, snapshot, stitched_metrics
from .research_v4 import relative_gates
from .research_v5 import model_config, paired_interval, training_markets
from .research_v6 import BAND, SCENARIOS, seed_compound, write_json


def dataset(data='data', extra='data/study005', audit_path=None):
    bars, original, audit = load_dataset(data, extra, audit_path)
    frames, late = {}, {}
    for symbol, b in bars.items():
        alpha = alpha_features(b)
        frames[symbol] = pd.concat([original[symbol], alpha], axis=1)[FEATURE_NAMES]
        # Do not apply study005's positional external-source lag to appended AT.
        late[symbol] = pd.concat([source_lag(original[symbol]), alpha], axis=1)[FEATURE_NAMES]
    return bars, frames, late, audit


def schedule(candidate, m):
    return np.ones(len(m.x), bool) if candidate.get('frequency') == '4h' else daily_schedule(m)


def desired(candidate, m):
    if candidate['kind'] == 'ensemble':
        return np.mean([desired(member, m) for member in candidate['members']], axis=0)
    allowed = schedule(candidate, m)
    x = m.x[allowed]
    kind = candidate['kind']
    if kind == 'gate':
        inputs = select_inputs(x, candidate['spec'])
        c = model_config(inputs=inputs.shape[1])
        net = neat.nn.FeedForwardNetwork.create(candidate['genome'], c)
        p = gated_targets(batch_activate(net, inputs)[None, :], x)[0]
    elif kind == 'alpha':
        p = alpha_targets(x, candidate.get('long_short', False))*candidate.get('scale', 1.)
        if candidate.get('rule'):
            p *= fixed_risk_scale(x[:, :len(ALL_FEATURES)], candidate['rule'])
    elif kind == 'baseline':
        p = baseline_targets(x)
        if candidate.get('rule'):
            p *= fixed_risk_scale(x[:, :len(ALL_FEATURES)], candidate['rule'])
    elif kind == 'cash':
        p = np.zeros(len(x))
    else:
        raise ValueError(kind)
    full = np.zeros(len(m.x))
    full[allowed] = p
    ix = np.maximum.accumulate(np.where(allowed, np.arange(len(m.x)), 0))
    return full[ix]


def evaluate(candidate, markets, multiplier=1., delay=0):
    assert_aligned(markets)
    result = {}
    for m in markets:
        p, allowed = desired(candidate, m), schedule(candidate, m)
        if delay:
            p = np.r_[np.zeros(delay), p[:-delay]]
            allowed = np.r_[np.zeros(delay, bool), allowed[:-delay]]
        result[m.symbol] = simulate(m, p, 5*multiplier, 2*multiplier,
                                    trade_allowed=allowed, rebalance_band=BAND)
    return result


def controls():
    return ([{'name': 'cash', 'kind': 'cash'}, {'name': 'baseline', 'kind': 'baseline'},
             {'name': 'ema_premium', 'kind': 'baseline', 'rule': 'premium'},
             {'name': 'alpha_only', 'kind': 'alpha'}, {'name': 'alpha_half', 'kind': 'alpha', 'scale': .5}]
            + [{'name': 'alpha_'+r, 'kind': 'alpha', 'rule': r} for r in ['funding', 'premium', 'both']]
            + [{'name': 'alpha_daily_ls', 'kind': 'alpha', 'long_short': True},
               {'name': 'alpha_4h_long', 'kind': 'alpha', 'frequency': '4h'},
               {'name': 'alpha_4h_ls', 'kind': 'alpha', 'frequency': '4h', 'long_short': True}])


def train_one(job):
    markets, popsize, generations, seed, spec, name, output, fingerprint, resume = job
    path = Path(output, 'jobs', name+'.pkl')
    if resume and path.exists():
        meta = json.loads(path.with_suffix('.json').read_text())
        if meta['protocol_sha256'] != fingerprint or meta['model_sha256'] != sha256(path.read_bytes()).hexdigest():
            raise ValueError('Invalid cached model provenance')
        with path.open('rb') as f:
            member = pickle.load(f)
        if member['spec'] != spec or member['seed'] != seed or member['train_end'] != str(max(m.times[-1] for m in markets)):
            raise ValueError('Cached training metadata differs')
        return member
    prepared = [prepare(m) for m in markets]
    benchmarks = [fitness_accounting(d, alpha_targets(d.x)[None, :], rebalance_band=BAND) for d in prepared]
    inputs = [select_inputs(d.x, spec) for d in prepared]
    c = model_config(popsize, inputs=inputs[0].shape[1])
    population = neat.Population(c, seed=seed)
    trace = Trace(name)
    population.add_reporter(trace)

    def fitness(genomes, conf):
        nets = [neat.nn.FeedForwardNetwork.create(g, conf) for _, g in genomes]
        results, cash = [], np.ones(len(nets), bool)
        for d, x in zip(prepared, inputs):
            raw = np.stack([batch_activate(net, x) for net in nets])
            targets = gated_targets(raw, d.x)
            results.append(fitness_accounting(d, targets, rebalance_band=BAND))
            cash &= np.all(targets == 0, axis=1)
        values = incremental_objective(results, benchmarks)
        for i, (_, genome) in enumerate(genomes):
            genome.fitness = float(values[i]-.0001*len(genome.connections))
            genome.research_cash = bool(cash[i])

    winner = population.run(fitness, generations)
    member = {'name': name, 'kind': 'gate', 'spec': spec, 'seed': seed, 'genome': winner,
              'train_cash': winner.research_cash, 'train_end': str(max(m.times[-1] for m in markets))}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        pickle.dump(member, f)
    tmp.replace(path)
    pd.DataFrame(trace.rows).to_csv(path.with_suffix('.csv'), index=False)
    write_json(path.with_suffix('.json'), {'protocol_sha256': fingerprint, 'model_sha256': sha256(path.read_bytes()).hexdigest()})
    return member


def validate_resume(out, args):
    protocol = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    if protocol['population'] != args.population or protocol['generations'] != args.generations or protocol['specifications'] != specifications():
        raise ValueError('Resume parameters differ')
    if protocol['seeds'] != SEEDS or protocol['folds'] != [list(v) for v in FOLDS]:
        raise ValueError('Resume experiment definitions differ')
    for name, expected in protocol['source_hashes'].items():
        if sha256(Path(name).read_bytes()).hexdigest() != expected or sha256((out/'source'/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Resume source differs: {name}')
    for label, root in [('base', args.data), ('extra', args.extra)]:
        if sha256(Path(root, 'manifest.json').read_bytes()).hexdigest() != protocol['manifest_hashes'][label]:
            raise ValueError('Resume data manifest differs')
    return protocol


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'report.json').exists():
        raise ValueError('Completed results are immutable')
    if (out/'protocol.json').exists() and not args.resume:
        raise ValueError('Use a new output directory or exact --resume')
    bars, frames, late, audit = dataset(args.data, args.extra, out/'source_audit.json')
    specs = specifications()
    if args.resume:
        protocol = validate_resume(out, args)
    else:
        hashes = snapshot(out)
        for path in [Path('reports/study007_method.md'), *Path('references/alphatrend').glob('*')]:
            target = out/'source'/path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            hashes[path.as_posix()] = sha256(path.read_bytes()).hexdigest()
        protocol = {'study': '007', 'specifications': specs, 'feature_names': FEATURE_NAMES, 'taxonomy': TYPES,
                    'population': args.population, 'generations': args.generations, 'seeds': SEEDS, 'folds': FOLDS,
                    'controls': controls(), 'embargo_days': 7, 'alpha_period': 14, 'alpha_multiplier': 1.,
                    'band': BAND, 'fee_bps': 5, 'slippage_bps': 2, 'primary_frequency': 'daily',
                    'primary_direction': 'long_only', 'fitness_reference': 'alpha_only',
                    'scenarios': [s for s, _ in SCENARIOS], 'source_hashes': hashes,
                    'manifest_hashes': {}, 'known_data': 'BTCETH2024-25 previously observed development; no2026refits or transfer access',
                    'method': 'Full frozen details in source/reports/study007_method.md',
                    'neat_version': neat.__version__, 'numpy_version': np.__version__, 'pandas_version': pd.__version__}
        for label, root in [('base', args.data), ('extra', args.extra)]:
            p = Path(root, 'manifest.json')
            protocol['manifest_hashes'][label] = sha256(p.read_bytes()).hexdigest()
            shutil.copyfile(p, out/f'{label}-manifest.json')
        write_json(out/'protocol.json', protocol)
    fingerprint = sha256((out/'protocol.json').read_bytes()).hexdigest()
    started = time.monotonic()
    jobs = []
    for fold, (a, z) in enumerate(FOLDS):
        train = training_markets(bars, frames, a)
        jobs.extend((train, args.population, args.generations, seed, spec, f'fold{fold}_{spec["name"]}_{seed}',
                     str(out), fingerprint, args.resume) for spec in specs for seed in SEEDS)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        models = {m['name']: m for m in pool.map(train_one, jobs)}
    logs, sequences, assets, seeds, dates = {}, {}, {}, [], []
    for fold, (a, z) in enumerate(FOLDS):
        markets = [with_features(s, b, frames[s], a, z) for s, b in bars.items()]
        delayed = [with_features(s, b, late[s], a, z) for s, b in bars.items()]
        dates.extend(markets[0].times[1:])
        candidates = controls()+[{'name': spec['name'], 'kind': 'ensemble',
                                 'members': [models[f'fold{fold}_{spec["name"]}_{seed}'] for seed in SEEDS]} for spec in specs]
        for candidate in candidates:
            name = candidate['name']
            for scenario, options in SCENARIOS:
                opts = dict(options)
                use = delayed if opts.pop('lagged', False) else markets
                r = evaluate(candidate, use, **opts)
                port = portfolio(r)
                sequences.setdefault((name, scenario), []).append(port['returns'])
                if scenario == 'base':
                    logs.setdefault(name, []).append({'period': [a, z], 'portfolio': summary(port),
                                                     'by_symbol': {s: summary(v) for s, v in r.items()}})
                    for s, v in r.items():
                        assets.setdefault((name, s), []).append(v['returns'])
        for spec in specs:
            for seed in SEEDS:
                member = models[f'fold{fold}_{spec["name"]}_{seed}']
                r = evaluate(member, markets)
                seeds.append({'fold': fold, 'spec': spec['name'], 'seed': seed,
                              'input_count': max(1, len(spec['features'])), 'train_cash': member['train_cash'],
                              'nodes': len(member['genome'].nodes), 'connections': len(member['genome'].connections),
                              'portfolio': summary(portfolio(r)), 'by_symbol': {s: summary(v) for s, v in r.items()}})
    curves = pd.DataFrame(index=pd.DatetimeIndex([pd.Timestamp(FOLDS[0][0], tz='UTC')]+dates))
    monthly = pd.DataFrame()
    month_index = (pd.DatetimeIndex(dates).tz_localize(None)-pd.Timedelta(nanoseconds=1)).to_period('M')
    rows = []
    for name, folds in logs.items():
        row = {'name': name, 'folds': folds,
               'scenarios': {s: summary(stitched_metrics(sequences[name, s])) for s, _ in SCENARIOS},
               'asset_compounded_returns': {s: stitched_metrics(assets[name, s])['return'] for s in bars}}
        row['absolute_gates'] = development_gate(row)
        row['absolute_pass'] = all(row['absolute_gates'].values())
        rows.append(row)
        combined = stitched_metrics(sequences[name, 'base'])
        curves[name] = combined['equity']
        monthly[name] = pd.Series(1+combined['returns'], index=month_index).groupby(level=0).prod()-1
    by_name = {r['name']: r for r in rows}
    for row in rows:
        row['relative_gates'] = {ref: relative_gates(row, by_name[ref]) for ref in ['alpha_only', 'baseline']}
        row['source_lag_pass'] = row['scenarios']['source_lag_4h']['return'] > max(
            0., *(by_name[ref]['scenarios']['source_lag_4h']['return'] for ref in ['alpha_only', 'baseline']))
        row['incremental_neat_pass'] = (row['name'] in {s['name'] for s in specs} and row['absolute_pass']
            and row['source_lag_pass'] and all(all(g.values()) for g in row['relative_gates'].values()))
        row['median_paired_excess'] = float(np.median([a['portfolio']['return']-b['portfolio']['return']
                                                     for a, b in zip(row['folds'], by_name['alpha_only']['folds'])]))
    eligible = [r for r in rows if r['incremental_neat_pass']]
    selected = max(eligible, key=lambda r: r['median_paired_excess'])['name'] if eligible else None
    selection = {'selected': selected, 'development': rows, 'per_seed': seeds, 'candidate_count': len(rows),
                 'population_evaluation_budget': len(jobs)*args.population*args.generations, 'development_only': True}
    write_json(out/'selection.json', selection)
    curves.to_csv(out/'development_equity.csv')
    monthly.to_csv(out/'monthly_returns.csv')
    contrasts = []
    pairs = [(s['name'], ref) for s in specs[2:] for ref in ['alpha_only', 'at_neat']]+[('at_neat', 'at_constant')]
    for name, ref in pairs:
        interval = paired_interval(sequences[name, 'base'], sequences[ref, 'base'], draws=5000, comparisons=len(pairs))
        interval['note'] = 'Descriptive only:17predeclared contrasts, paired7day blocks within folds; no correction for adaptive research or refitting uncertainty.'
        contrasts.append({'candidate': name, 'reference': ref, 'interval': interval,
            'return_difference_pp': 100*(by_name[name]['scenarios']['base']['return']-by_name[ref]['scenarios']['base']['return']),
            'seed_compound_excess': [seed_compound(seeds, name, seed)-(seed_compound(seeds, ref, seed)
                if ref.startswith('at_') else by_name[ref]['scenarios']['base']['return']) for seed in SEEDS]})
    write_json(out/'report.json', {'selected': selected, 'contrasts': contrasts, 'transfer_accessed': False,
                                 'distinct_training_jobs': len(jobs), 'elapsed_seconds': time.monotonic()-started})
    print('Study007 selected:', selected, flush=True)
    print(pd.DataFrame([{'name': r['name'], **r['scenarios']['base'], 'pass': r['incremental_neat_pass']} for r in rows]).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data')
    parser.add_argument('--extra', default='data/study005')
    parser.add_argument('--output', default='artifacts/experiment_007')
    parser.add_argument('--population', type=int, default=64)
    parser.add_argument('--generations', type=int, default=30)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--resume', action='store_true')
    run(parser.parse_args())
