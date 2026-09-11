"""Study005: bounded indicator/source ablations, without opening transfer data."""
import argparse
import ast
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

from .batch import fitness_accounting, prepare
from .features_v5 import (ALL_FEATURES, CROSS, DERIVATIVES, GROUPS, PACKS, TECHNICAL,
                          load_dataset, pack_matrix, source_lag, with_features)
from .market import FEATURES, summary, simulate
from .overlay import baseline_targets, incremental_objective, overlay_targets
from .research import assert_aligned, batch_activate, config, portfolio
from .research_v3 import FOLDS, SEEDS, Trace, daily_schedule, development_gate, snapshot, stitched_metrics
from .research_v4 import relative_gates

BAND = .05
SCENARIOS = [('base', {}), ('cost_1_5', {'multiplier': 1.5}),
             ('double_cost', {'multiplier': 2.}), ('delay_4h', {'delay': 1}),
             ('source_lag_4h', {'lagged': True})]


def model_config(population=64, inputs=len(ALL_FEATURES)):
    c = config(population)
    c.genome_config.num_inputs = inputs
    c.genome_config.input_keys = [-i-1 for i in range(inputs)]
    return c


def simple_scale(x, group):
    def col(name):
        return x[:, ALL_FEATURES.index(name)]
    technical = np.where((col('adx14') >= -.5) & (col('rsi14') <= .4), 1., .5)
    # Inputs are clipped at +/-5, so >=5 retains the intended premium threshold.
    derivatives = np.where((col('funding_last_bps') > 3.) | (col('premium1d_bps') >= 5.), .5, 1.)
    cross = np.where((col('spot_buy1d') < 0) & (col('peer_trend180') < 0), .5, 1.)
    return {'technical': technical, 'derivatives': derivatives, 'cross': cross,
            'combined': (technical + derivatives + cross)/3}[group]


def desired(candidate, m, c, drop_group=None):
    d = prepare(m)
    if candidate['kind'] == 'ensemble':
        return np.mean([desired(member, m, c, drop_group) for member in candidate['members']], axis=0)
    x = d.x.copy()
    if drop_group:
        x[:, [ALL_FEATURES.index(n) for n in GROUPS[drop_group]]] = 0
    if candidate['kind'] == 'gate':
        x = pack_matrix(x, candidate['pack'])
        raw = batch_activate(neat.nn.FeedForwardNetwork.create(candidate['genome'], c), x)
        daily = overlay_targets(raw[None, :], d.x, 'plain')[0]
    elif candidate['kind'] == 'compact':
        cc = model_config(c.pop_size, 10)
        raw = batch_activate(neat.nn.FeedForwardNetwork.create(candidate['genome'], cc), x[:, :10])
        daily = overlay_targets(raw[None, :], d.x, 'plain')[0]
    elif candidate['kind'] == 'rule':
        daily = baseline_targets(d.x) * simple_scale(x, candidate['group'])
    elif candidate['kind'] == 'baseline':
        daily = baseline_targets(d.x) * candidate.get('scale', 1.)
    elif candidate['kind'] == 'cash':
        daily = np.zeros(len(d.x))
    else:
        raise ValueError(candidate['kind'])
    full = np.zeros(len(m.x))
    full[d.anchors] = daily
    anchors = np.maximum.accumulate(np.where(daily_schedule(m), np.arange(len(m.x)), 0))
    return full[anchors]


def evaluate(candidate, markets, c, multiplier=1., delay=0, drop_group=None):
    assert_aligned(markets)
    result = {}
    for m in markets:
        p, allowed = desired(candidate, m, c, drop_group), daily_schedule(m)
        if delay:
            p = np.r_[np.zeros(delay), p[:-delay]]
            allowed = np.r_[np.zeros(delay, dtype=bool), allowed[:-delay]]
        r = simulate(m, p, 5*multiplier, 2*multiplier, trade_allowed=allowed, rebalance_band=BAND)
        r['decision_flat_fraction'] = float(np.mean(p[allowed] == 0))
        result[m.symbol] = r
    return result


def training_markets(bars, frames, start):
    end = pd.Timestamp(start) - pd.Timedelta(days=7)
    bounds = list(pd.date_range('2022-01-01', end, freq='6MS')) + [end]
    return [with_features(f'{s}_{a:%Y-%m}', b, frames[s], str(a.date()), str(z.date()))
            for s, b in bars.items() for a, z in zip(bounds[:-1], bounds[1:]) if z > a]


def train_one(job):
    markets, popsize, generations, seed, pack, name, output = job
    prepared = [prepare(m) for m in markets]
    benchmarks = [fitness_accounting(d, baseline_targets(d.x)[None, :], rebalance_band=BAND) for d in prepared]
    c = model_config(popsize)
    population = neat.Population(c, seed=seed)
    trace = Trace(name)
    population.add_reporter(trace)
    masked = [pack_matrix(d.x, pack) for d in prepared]

    def fitness(genomes, conf):
        nets = [neat.nn.FeedForwardNetwork.create(g, conf) for _, g in genomes]
        results, cash = [], np.ones(len(nets), bool)
        for d, x in zip(prepared, masked):
            raw = np.stack([batch_activate(net, x) for net in nets])
            targets = overlay_targets(raw, d.x, 'plain')
            results.append(fitness_accounting(d, targets, rebalance_band=BAND))
            cash &= np.all(targets == 0, axis=1)
        values = incremental_objective(results, benchmarks)
        for i, (_, g) in enumerate(genomes):
            g.fitness = float(values[i] - .0001*len(g.connections))
            g.research_cash = bool(cash[i])

    winner = population.run(fitness, generations)
    candidate = {'name': name, 'kind': 'gate', 'pack': pack, 'seed': seed, 'genome': winner,
                 'train_cash': winner.research_cash, 'train_end': str(max(m.times[-1] for m in markets))}
    path = Path(output, 'jobs', name + '.pkl')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        pickle.dump(candidate, f)
    temp.replace(path)
    pd.DataFrame(trace.rows).to_csv(path.with_suffix('.csv'), index=False)
    return candidate


def controls():
    return [{'name': 'cash', 'kind': 'cash'}, {'name': 'baseline', 'kind': 'baseline'},
            {'name': 'half_baseline', 'kind': 'baseline', 'scale': .5}] + [
                {'name': f'rule_{g}', 'kind': 'rule', 'group': g} for g in PACKS if g != 'core']


def topology(member):
    enabled = FEATURES + (GROUPS.get(member['pack'], []) if member['pack'] != 'combined' else TECHNICAL+DERIVATIVES+CROSS)
    live_keys = {-ALL_FEATURES.index(n)-1 for n in enabled}
    connections = [v for v in member['genome'].connections.values() if v.enabled]
    return {'nodes': len(member['genome'].nodes), 'connections': len(member['genome'].connections),
            'enabled_connections': len(connections),
            'enabled_from_masked_inputs': sum(v.key[0] < 0 and v.key[0] not in live_keys for v in connections)}


def paired_interval(parts, reference, draws=2000, comparisons=4):
    """Descriptive paired, within-fold circular 7-day blocks; not a discovery test."""
    rng = np.random.default_rng(505)
    observed = sum(np.sum(np.log1p(a)-np.log1p(b)) for a, b in zip(parts, reference))
    samples = np.zeros(draws)
    for a, b in zip(parts, reference):
        delta = np.log1p(a) - np.log1p(b)
        starts = rng.integers(0, len(delta), size=(draws, int(np.ceil(len(delta)/42))))
        ix = ((starts[:, :, None] + np.arange(42)) % len(delta)).reshape(draws, -1)[:, :len(delta)]
        samples += delta[ix].sum(axis=1)
    low, high = np.quantile(samples, [.025/comparisons, 1-.025/comparisons])
    return {'log_wealth_excess': float(observed), 'family_adjusted_interval': [float(low), float(high)],
            'relative_wealth_excess': float(np.expm1(observed)), 'draws': draws,
            'note': 'Descriptive fixed-model resampling, four augmentation comparisons only; no correction for previous studies, training or adaptive research.'}


def compact_reference(fold):
    members = []
    for seed in SEEDS:
        path = Path('artifacts/experiment_004/jobs', f'fold{fold}_plain_{seed}.pkl')
        with path.open('rb') as f:
            member = pickle.load(f)
        members.append(dict(member, kind='compact'))
    return {'name': 'compact_core_reference', 'kind': 'ensemble', 'members': members}


def validate_model_cache(cache, args):
    """Reuse interrupted deterministic jobs only when training code/data match."""
    root = Path(cache)
    old = json.loads((root/'protocol.json').read_text())
    if old['population'] != args.population or old['generations'] != args.generations or old['packs'] != PACKS or old['seeds'] != SEEDS or old['folds'] != [list(v) for v in FOLDS] or old['band'] != BAND:
        raise ValueError('Cached training protocol differs')
    for label, data_root in [('base', args.data), ('extra', args.extra)]:
        if sha256(Path(data_root, 'manifest.json').read_bytes()).hexdigest() != old['manifest_hashes'][label]:
            raise ValueError('Cached data manifest differs')
    for name, expected in old['source_hashes'].items():
        frozen = root/'source'/name
        if sha256(frozen.read_bytes()).hexdigest() != expected:
            raise ValueError('Corrupt cached source snapshot')
        if Path(name).as_posix() != 'neattrade/research_v5.py' and sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Cached dependency differs: {name}')
    original = ast.parse((root/'source/neattrade/research_v5.py').read_text())
    current = ast.parse(Path(__file__).read_text())
    functions = lambda tree: {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}
    before, after = functions(original), functions(current)
    # run() contains orchestration/JSON export; all model and evaluation routines must match.
    for name in before.keys()-{'run'}:
        if before[name] != after[name]:
            raise ValueError(f'Cached model/evaluation function differs: {name}')
    return {'path': str(root), 'protocol_sha256': sha256((root/'protocol.json').read_bytes()).hexdigest(),
            'reason': 'Original run stopped on NumPy boolean JSON serialization after60saved jobs; no fitness, data, model or decision-rule changes.'}


def execute_job(task):
    job, cache = task
    if cache:
        path = Path(cache, 'jobs', job[5]+'.pkl')
        if path.exists():
            with path.open('rb') as f:
                candidate = pickle.load(f)
            if candidate['seed'] != job[3] or candidate['pack'] != job[4] or candidate['train_end'] != str(max(m.times[-1] for m in job[0])):
                raise ValueError('Cached model metadata differs')
            dest = Path(job[6], 'jobs', path.name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            shutil.copyfile(path.with_suffix('.csv'), dest.with_suffix('.csv'))
            return candidate
    return train_one(job)


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        raise ValueError('Use a fresh study directory')
    bars, frames, audit = load_dataset(args.data, args.extra, out/'source_audit.json')
    lagged = {s: source_lag(f) for s, f in frames.items()}
    reuse = validate_model_cache(args.models_from, args) if args.models_from else None
    protocol = {'study': '005', 'steps': ['source inventory and provenance', 'fixed grouped indicators',
                'causal alignment and accounting checks', 'matched rolling ablations', 'stress, contribution and known-context review'],
        'packs': PACKS, 'feature_names': ALL_FEATURES, 'input_slots': len(ALL_FEATURES),
        'zero_mask_note': 'Common topology size, but neutral mutations mean different effective search dynamics; compact core replay and dead-input counts disclosed.',
        'seeds': SEEDS, 'population': args.population, 'generations': args.generations, 'folds': FOLDS,
        'embargo_days': 7, 'confidence_thresholds': [-.35, .35], 'band': BAND,
        'fee_bps': 5, 'slippage_bps': 2, 'cash_yield': 0,
        'fitness': 'Unchanged study004 duration weighted baseline excess log growth, cross-block dispersion, excess20pct drawdown and0.0001 per connection penalty',
        'source_availability': 'Completed candles and settled funding lagged one4h bar; archives are retrospective and do not prove historical delivery latency or revision vintages.',
        'scenarios': [s for s, _ in SCENARIOS],
        'selection': 'Absolute study004 gates plus total, median paired, 3of4fold, doublecost and delay excess over baseline; source-lag excess and positive source-lag return also required. Best median paired excess if eligible, else none.',
        'added_source_screen': 'Augmented pack must additionally beat masked core in total, source-lag and median paired seed compounded return. Descriptive, not alpha proof.',
        'simple_rules': {'technical': 'half unless ADX>=25 and RSI<=70',
                         'derivatives': 'half if last settled funding>3bps or average premium>=5bps',
                         'cross': 'half if spot1d taker imbalance<0 AND peer slowtrend<0',
                         'combined': 'arithmetic mean of three fixed rule multipliers'},
        'known2026': ['2026-01-01', '2026-08-31 20:00'],
        'all_results_are_previously_seen_market_development': True,
        'transfer': 'No SOL/BNB access in this five-step batch, regardless of screen; joint review first.',
        'inference_ablation': 'Combined model each non-core group replaced with zero, fixed weights; OOD sensitivity only, not causal source importance.',
        'bootstrap': '2000 paired7day circular draws within each fold; Bonferroni four augmentation comparisons; descriptive only',
        'source_hashes': snapshot(out), 'manifest_hashes': {},
        'compact_reference_hashes': {str(p): sha256(p.read_bytes()).hexdigest() for p in sorted(Path('artifacts/experiment_004/jobs').glob('fold*_plain_*.pkl'))},
        'reused_training': reuse,
        'neat_version': neat.__version__, 'numpy_version': np.__version__, 'pandas_version': pd.__version__}
    for label, root in [('base', args.data), ('extra', args.extra)]:
        path = Path(root, 'manifest.json')
        protocol['manifest_hashes'][label] = sha256(path.read_bytes()).hexdigest()
        shutil.copyfile(path, out/f'{label}-manifest.json')
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    start = time.monotonic()
    jobs = [(training_markets(bars, frames, a), args.population, args.generations, seed, pack,
             f'fold{fold}_{pack}_{seed}', str(out)) for fold, (a, z) in enumerate(FOLDS) for pack in PACKS for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        models = {m['name']: m for m in pool.map(execute_job, [(job, args.models_from) for job in jobs])}
    c = model_config(args.population)
    sequences, assets, logs, seed_rows, ablations, dates = {}, {}, {}, [], {}, []
    for fold, (a, z) in enumerate(FOLDS):
        markets = [with_features(s, b, frames[s], a, z) for s, b in bars.items()]
        late = [with_features(s, b, lagged[s], a, z) for s, b in bars.items()]
        dates.extend(markets[0].times[1:])
        candidates = controls() + [{'name': f'neat_{p}', 'kind': 'ensemble', 'pack': p,
                    'members': [models[f'fold{fold}_{p}_{seed}'] for seed in SEEDS]} for p in PACKS] + [compact_reference(fold)]
        for candidate in candidates:
            name = candidate['name']
            for scenario, options in SCENARIOS:
                opts = dict(options)
                use = late if opts.pop('lagged', False) else markets
                r = evaluate(candidate, use, c, **opts)
                port = portfolio(r)
                sequences.setdefault((name, scenario), []).append(port['returns'])
                if scenario == 'base':
                    logs.setdefault(name, []).append({'period': [a, z], 'portfolio': summary(port),
                                                     'by_symbol': {s: summary(v) for s, v in r.items()}})
                    for s, v in r.items():
                        assets.setdefault((name, s), []).append(v['returns'])
            if name == 'neat_combined':
                for group in ['technical', 'derivatives', 'cross']:
                    r = evaluate(candidate, markets, c, drop_group=group)
                    ablations.setdefault(group, []).append(portfolio(r)['returns'])
        for pack in PACKS:
            for seed in SEEDS:
                member = models[f'fold{fold}_{pack}_{seed}']
                r = evaluate(member, markets, c)
                seed_rows.append({'fold': fold, 'pack': pack, 'seed': seed, 'train_cash': member['train_cash'],
                                  **topology(member), 'portfolio': summary(portfolio(r)),
                                  'by_symbol': {s: summary(v) for s, v in r.items()}})
    rows = []
    curves = pd.DataFrame(index=pd.DatetimeIndex([pd.Timestamp(FOLDS[0][0], tz='UTC')] + dates))
    for name, folds in logs.items():
        row = {'name': name, 'folds': folds,
               'scenarios': {s: summary(stitched_metrics(sequences[name, s])) for s, _ in SCENARIOS},
               'asset_compounded_returns': {s: stitched_metrics(assets[name, s])['return'] for s in bars}}
        row['absolute_gates'] = development_gate(row)
        row['absolute_pass'] = all(row['absolute_gates'].values())
        rows.append(row)
        curves[name] = stitched_metrics(sequences[name, 'base'])['equity']
    baseline = next(r for r in rows if r['name'] == 'baseline')
    core = next(r for r in rows if r['name'] == 'neat_core')
    for row in rows:
        row['relative_gates'] = relative_gates(row, baseline)
        row['relative_gates']['source_lag_excess_positive'] = row['scenarios']['source_lag_4h']['return'] > max(0., baseline['scenarios']['source_lag_4h']['return'])
        row['incremental_neat_pass'] = row['name'].startswith('neat_') and row['absolute_pass'] and all(row['relative_gates'].values())
        row['median_paired_excess'] = float(np.median([a['portfolio']['return']-b['portfolio']['return'] for a, b in zip(row['folds'], baseline['folds'])]))
        if row['name'].startswith('neat_') and row['name'] != 'neat_core':
            pack = row['name'][5:]
            paired = []
            for seed in SEEDS:
                r = np.prod([1+v['portfolio']['return'] for v in seed_rows if v['pack'] == pack and v['seed'] == seed])-1
                b = np.prod([1+v['portfolio']['return'] for v in seed_rows if v['pack'] == 'core' and v['seed'] == seed])-1
                paired.append(float(r-b))
            row['paired_seed_excess_vs_core'] = paired
            row['added_source_screen'] = (row['scenarios']['base']['return'] > core['scenarios']['base']['return'] and
                                         row['scenarios']['source_lag_4h']['return'] > core['scenarios']['source_lag_4h']['return'] and bool(np.median(paired) > 0))
            row['paired_core_bootstrap'] = paired_interval(sequences[row['name'], 'base'], sequences['neat_core', 'base'])
    eligible = [r for r in rows if r['incremental_neat_pass']]
    selected = max(eligible, key=lambda r: r['median_paired_excess'])['name'] if eligible else None
    selection = {'selected': selected, 'development': rows, 'per_seed': seed_rows, 'candidate_count': len(rows),
                 'population_evaluation_budget': len(jobs)*args.population*args.generations, 'development_only': True}
    (out/'selection.json').write_text(json.dumps(selection, indent=2))
    curves.to_csv(out/'development_equity.csv')
    print(f'Frozen study005 NEAT selection: {selected}', flush=True)
    finals = [(training_markets(bars, frames, '2026-01-01'), args.population, args.generations, seed, pack,
               f'final_{pack}_{seed}', str(out)) for pack in PACKS for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        final_models = list(pool.map(train_one, finals))
    final_candidates = [{'name': f'neat_{p}', 'kind': 'ensemble', 'pack': p,
                         'members': [m for m in final_models if m['pack'] == p]} for p in PACKS]
    with (out/'frozen_final_models.pkl').open('wb') as f:
        pickle.dump({'config': c, 'candidates': final_candidates, 'selected': selected}, f)
    recent = [with_features(s, b, frames[s], *protocol['known2026']) for s, b in bars.items()]
    context, recent_curves = {}, pd.DataFrame(index=recent[0].times)
    for candidate in controls() + final_candidates:
        r = evaluate(candidate, recent, c)
        port = portfolio(r)
        context[candidate['name']] = {'portfolio': summary(port), 'by_symbol': {s: summary(v) for s, v in r.items()}}
        recent_curves[candidate['name']] = port['equity']
    recent_curves.to_csv(out/'known2026_equity.csv')
    report = {'selected': selected, 'known2026_context': context, 'transfer_accessed': False,
              'inference_zero_group_sensitivity': {k: summary(stitched_metrics(v)) for k, v in ablations.items()},
              'final_refit_evaluation_budget': len(finals)*args.population*args.generations,
              'elapsed_seconds': time.monotonic()-start}
    (out/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data')
    parser.add_argument('--extra', default='data/study005')
    parser.add_argument('--output', default='artifacts/experiment_005')
    parser.add_argument('--population', type=int, default=64)
    parser.add_argument('--generations', type=int, default=30)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--models-from', default=None, help='Verified interrupted study005 training cache')
    run(parser.parse_args())
