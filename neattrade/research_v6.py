"""Study006: only-type and retrained leave-one-type-out NEAT comparisons."""
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

from .batch import fitness_accounting, prepare
from .features_v5 import ALL_FEATURES, load_dataset, source_lag, with_features
from .indicator_types import LABELS, TYPES, fixed_risk_scale, select_inputs, specifications
from .market import summary, simulate
from .overlay import baseline_targets, incremental_objective, overlay_targets
from .research import assert_aligned, batch_activate, portfolio
from .research_v3 import FOLDS, SEEDS, Trace, daily_schedule, development_gate, snapshot, stitched_metrics
from .research_v4 import relative_gates
from .research_v5 import model_config, paired_interval, training_markets

BAND = .05
SCENARIOS = [('base', {}), ('double_cost', {'multiplier': 2.}),
             ('delay_4h', {'delay': 1}), ('source_lag_4h', {'lagged': True})]


def write_json(path, value):
    def convert(v):
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        raise TypeError(f'Unsupported JSON value: {type(v)}')
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=convert), encoding='utf-8')


def desired(candidate, m):
    d = prepare(m)
    kind = candidate['kind']
    if kind == 'ensemble':
        return np.mean([desired(member, m) for member in candidate['members']], axis=0)
    if kind == 'gate':
        x = select_inputs(d.x, candidate['spec'])
        c = model_config(inputs=x.shape[1])
        raw = batch_activate(neat.nn.FeedForwardNetwork.create(candidate['genome'], c), x)
        daily = overlay_targets(raw[None, :], d.x, 'plain')[0]
    elif kind == 'baseline':
        daily = baseline_targets(d.x)*candidate.get('scale', 1.)
    elif kind == 'rule':
        daily = baseline_targets(d.x)*fixed_risk_scale(d.x, candidate['rule'])
    elif kind == 'cash':
        daily = np.zeros(len(d.x))
    else:
        raise ValueError(kind)
    full = np.zeros(len(m.x))
    full[d.anchors] = daily
    ix = np.maximum.accumulate(np.where(daily_schedule(m), np.arange(len(m.x)), 0))
    return full[ix]


def evaluate(candidate, markets, multiplier=1., delay=0):
    assert_aligned(markets)
    result = {}
    for m in markets:
        p, allowed = desired(candidate, m), daily_schedule(m)
        if delay:
            p = np.r_[np.zeros(delay), p[:-delay]]
            allowed = np.r_[np.zeros(delay, dtype=bool), allowed[:-delay]]
        r = simulate(m, p, 5*multiplier, 2*multiplier, trade_allowed=allowed, rebalance_band=BAND)
        r['decision_flat_fraction'] = float(np.mean(p[allowed] == 0))
        result[m.symbol] = r
    return result


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
    benchmarks = [fitness_accounting(d, baseline_targets(d.x)[None, :], rebalance_band=BAND) for d in prepared]
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
            targets = overlay_targets(raw, d.x, 'plain')
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


def controls():
    return [{'name': 'cash', 'kind': 'cash'}, {'name': 'baseline', 'kind': 'baseline'},
            {'name': 'half_baseline', 'kind': 'baseline', 'scale': .5}] + [
                {'name': 'rule_'+r, 'kind': 'rule', 'rule': r} for r in ['funding', 'premium', 'both']]


def seed_compound(seed_rows, spec, seed):
    rows = [r for r in seed_rows if r['spec'] == spec and r['seed'] == seed]
    return float(np.prod([1+r['portfolio']['return'] for r in rows])-1)


def comparison(parts, reference):
    r = paired_interval(parts, reference, comparisons=14)
    r['note'] = 'Descriptive only:14type contrasts, paired7day circular blocks within folds. No correction for broader adaptive research or refitting uncertainty.'
    return r


def validate_resume(out, args, specs):
    protocol = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    if protocol['population'] != args.population or protocol['generations'] != args.generations or protocol['specifications'] != specs:
        raise ValueError('Resume parameters differ')
    if protocol['seeds'] != SEEDS or protocol['folds'] != [list(v) for v in FOLDS] or protocol['band'] != BAND:
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
    bars, frames, audit = load_dataset(args.data, args.extra, out/'source_audit.json')
    lagged = {s: source_lag(f) for s, f in frames.items()}
    specs = specifications()
    if args.resume:
        protocol = validate_resume(out, args, specs)
    else:
        protocol = {'study': '006', 'taxonomy': TYPES, 'labels': LABELS, 'specifications': specs,
            'input_selection': 'Compact selected columns, no zero-masked dead inputs. Constant-only control has one constant1input.',
            'conditional_scope': 'Types refer to NEAT gate inputs. Every policy still uses fixedEMA180eligibility andvolatilitysizing; normalizations may includevolatility.',
            'interpretation': 'Retraining ablations include information, architecture, regularization and search effects. Indicator groups are pragmatic, not independent causes.',
            'population': args.population, 'generations': args.generations, 'seeds': SEEDS, 'folds': FOLDS,
            'embargo_days': 7, 'band': BAND, 'fee_bps': 5, 'slippage_bps': 2, 'cash_yield': 0,
            'fitness': 'Same study005 incremental loggrowth, dispersion, excess20pctdrawdown andcomplexitypenalty0.0001perconnection.',
            'scenarios': [s for s, _ in SCENARIOS], 'source_availability': 'Same lagged features asstudy005; external inputs receiveoneextra4h lag in separate stress.',
            'fixed_rules': 'Previously set thresholds unchanged: funding_last>3bps, premium1d>=5bps; each and theirOR separately halve target.',
            'selection': 'All inherited absolute gates plus fullbaseline relative total/median/3of4fold/cost/delay gates and positivesource-lagexcess. Best medianpaired excess if eligible; no deployment.',
            'contrasts': 'Seven only-type minusconstant-only; seven all-types minusretrained-without-type. Report full pairedseedresults and14comparison descriptiveintervals.',
            'known_data': 'BTCETH2024-25alreadyobserved development; no newholdout claim. No2026refits andnoSOLBNBaccess inthisbatch.',
            'stopping': '16specifications fixedbeforetraining; no post-result combinations or threshold search.',
            'source_hashes': snapshot(out), 'manifest_hashes': {},
            'neat_version': neat.__version__, 'numpy_version': np.__version__, 'pandas_version': pd.__version__}
        for label, root in [('base', args.data), ('extra', args.extra)]:
            p = Path(root, 'manifest.json')
            protocol['manifest_hashes'][label] = sha256(p.read_bytes()).hexdigest()
            shutil.copyfile(p, out/f'{label}-manifest.json')
        write_json(out/'protocol.json', protocol)
    fingerprint = sha256((out/'protocol.json').read_bytes()).hexdigest()
    train_features = pd.concat([f.loc[(f.index >= '2022-01-31 04:00') & (f.index < '2023-12-25')]
                                for f in frames.values()], ignore_index=True)
    train_features.corr().to_csv(out/'training_feature_correlations.csv')
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
        late = [with_features(s, b, lagged[s], a, z) for s, b in bars.items()]
        dates.extend(markets[0].times[1:])
        candidates = controls()+[{'name': spec['name'], 'kind': 'ensemble', 'spec': spec,
                                 'members': [models[f'fold{fold}_{spec["name"]}_{seed}'] for seed in SEEDS]} for spec in specs]
        for candidate in candidates:
            name = candidate['name']
            for scenario, options in SCENARIOS:
                opts = dict(options)
                use = late if opts.pop('lagged', False) else markets
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
    baseline = by_name['baseline']
    for row in rows:
        row['relative_gates'] = relative_gates(row, baseline)
        row['relative_gates']['positive_source_lag_excess'] = row['scenarios']['source_lag_4h']['return'] > max(0., baseline['scenarios']['source_lag_4h']['return'])
        row['incremental_neat_pass'] = row['name'] in {s['name'] for s in specs} and row['absolute_pass'] and all(row['relative_gates'].values())
        row['median_paired_excess'] = float(np.median([a['portfolio']['return']-b['portfolio']['return'] for a, b in zip(row['folds'], baseline['folds'])]))
    eligible = [r for r in rows if r['incremental_neat_pass']]
    selected = max(eligible, key=lambda r: r['median_paired_excess'])['name'] if eligible else None
    selection = {'selected': selected, 'development': rows, 'per_seed': seeds, 'candidate_count': len(rows),
                 'population_evaluation_budget': len(jobs)*args.population*args.generations, 'development_only': True}
    write_json(out/'selection.json', selection)
    curves.to_csv(out/'development_equity.csv')
    monthly.to_csv(out/'monthly_returns.csv')
    contrasts = []
    seed_fold = {(r['spec'], r['seed'], r['fold']): r['portfolio']['return'] for r in seeds}
    for group in TYPES:
        only, without = 'only_'+group, 'without_'+group
        contrasts.append({'type': group, 'label': LABELS[group],
            'only_return': by_name[only]['scenarios']['base']['return'],
            'without_return': by_name[without]['scenarios']['base']['return'],
            'only_minus_intercept_pp': 100*(by_name[only]['scenarios']['base']['return']-by_name['intercept']['scenarios']['base']['return']),
            'all_minus_without_pp': 100*(by_name['all']['scenarios']['base']['return']-by_name[without]['scenarios']['base']['return']),
            'only_paired_seed_excess': [seed_compound(seeds, only, seed)-seed_compound(seeds, 'intercept', seed) for seed in SEEDS],
            'all_paired_seed_minus_without': [seed_compound(seeds, 'all', seed)-seed_compound(seeds, without, seed) for seed in SEEDS],
            'seed_fold_differences': [{'seed': seed, 'fold': fold,
                'only_minus_intercept': seed_fold[only, seed, fold]-seed_fold['intercept', seed, fold],
                'all_minus_without': seed_fold['all', seed, fold]-seed_fold[without, seed, fold]}
                for seed in SEEDS for fold in range(len(FOLDS))],
            'only_vs_intercept_interval': comparison(sequences[only, 'base'], sequences['intercept', 'base']),
            'all_vs_without_interval': comparison(sequences['all', 'base'], sequences[without, 'base'])})
    write_json(out/'report.json', {'selected': selected, 'contrasts': contrasts, 'transfer_accessed': False,
                                 'distinct_training_jobs': len(jobs), 'elapsed_seconds': time.monotonic()-started})
    print('Study006 selected:', selected, flush=True)
    print(pd.DataFrame([{'name': r['name'], **r['scenarios']['base'], 'pass': r['incremental_neat_pass']} for r in rows]).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data')
    parser.add_argument('--extra', default='data/study005')
    parser.add_argument('--output', default='artifacts/experiment_006')
    parser.add_argument('--population', type=int, default=64)
    parser.add_argument('--generations', type=int, default=30)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--resume', action='store_true')
    run(parser.parse_args())
