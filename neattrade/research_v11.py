"""Frozen candle-only feature x architecture x timeframe discovery study."""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import distributions
import json
from pathlib import Path
import pickle
import shutil
import sys
import time

import neat
import numpy as np
import pandas as pd

from .batch import DailyMarket
from .fast_accounting import population_accounting
from .indicators_v11 import (SYMBOLS, FRAMES, FEATURES, PACKS, COLUMNS, dataset,
    market, train_markets, inputs, risk, alpha_target, learned_target, allowed_times)
from .market import simulate, summary
from .overlay import incremental_objective
from .policies_v11 import forward, dimensions, fit
from .research import batch_activate
from .research_v3 import FOLDS, SEEDS, Trace, snapshot
from .research_v5 import model_config
from .research_v6 import write_json
from .timeframes import metrics, portfolio, daily_returns

OUT = Path('artifacts/experiment_011')
KINDS = ['linear', 'mlp', 'neat']
SPECS = [(k, p) for k in KINDS for p in PACKS]
CONTROLS = ['alpha', 'alpha_half', 'ema', 'passive_vol', 'cash']
WINDOWS = FOLDS+[('2026-01-01', '2026-07-01'), ('2026-07-01', '2026-08-31 20:00')]
POPULATION, GENERATIONS = 48, 24
SCENARIOS = [('base', {}), ('double_cost', {'multiplier': 2.}), ('delay15m', {'delay': 1})]
INTERVAL_COUNT, DRAWS = 126, 100000


def verify_raw_sources():
    source = Path('artifacts/experiment_008/manifest.json')
    prior = json.loads(Path('artifacts/experiment_008/protocol.json').read_text())
    assert sha256(source.read_bytes()).hexdigest() == prior['manifest_sha256']
    manifest = json.loads(source.read_text())
    for item in manifest:
        assert 'book' not in item['path'].lower()
        assert sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
    return len(manifest)


def prepare(m, tf):
    anchors = np.flatnonzero(allowed_times(m, tf))
    return DailyMarket(m, anchors, m.x[anchors], [])


def raw_output(member, x):
    if member['kind'] in ['linear', 'mlp']:
        return forward(member['weights'], x, member['kind'])[0]
    net = neat.nn.FeedForwardNetwork.create(member['genome'], model_config(POPULATION, x.shape[1]))
    return batch_activate(net, x)


def desired(candidate, m, tf):
    allowed = allowed_times(m, tf)
    x = m.x[allowed]
    if isinstance(candidate, list):
        p = np.mean([desired(c, m, tf)[0][allowed] for c in candidate], axis=0)
    elif isinstance(candidate, dict):
        p = learned_target(raw_output(candidate, inputs(x, candidate['pack'])), x)
    elif candidate in ['alpha', 'alpha_half']:
        p = alpha_target(x)*(.5 if candidate == 'alpha_half' else 1.)
    elif candidate == 'ema':
        p = (x[:, FEATURES.index('ema12_26')] > 0)*risk(x)
    elif candidate == 'passive_vol':
        p = risk(x)
    elif candidate == 'cash':
        p = np.zeros(len(x))
    else:
        raise ValueError('Unknown candidate')
    anchors = np.flatnonzero(allowed)
    return np.repeat(p, np.diff(np.r_[anchors, len(m.x)])), allowed


def evaluate(candidate, markets, tf, multiplier=1., delay=0):
    result = {}
    for s, m in markets.items():
        p, allowed = desired(candidate, m, tf)
        if delay:
            p = np.r_[np.zeros(delay), p[:-delay]]
            allowed = np.r_[np.zeros(delay, bool), allowed[:-delay]]
        r = simulate(m, p, 5*multiplier, 2*multiplier, trade_allowed=allowed, rebalance_band=.05)
        r['sharpe'] = metrics(r['returns'])['sharpe']
        result[s] = r
    return result


def scores(results, benchmarks, connections):
    value = incremental_objective(results, benchmarks)-.0001*np.asarray(connections)
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite objective')
    return value


def train_one(job):
    markets, tf, kind, pack, seed, name, protocol_hash = job
    path = OUT/'jobs'/f'{name}.pkl'
    if path.exists():
        meta = json.loads(path.with_suffix('.json').read_text())
        assert meta['protocol_sha256'] == protocol_hash
        assert meta['model_sha256'] == sha256(path.read_bytes()).hexdigest()
        assert meta['history_sha256'] == sha256(path.with_suffix('.csv').read_bytes()).hexdigest()
        result = pickle.loads(path.read_bytes())
        assert (result['name'], result['timeframe'], result['kind'], result['pack'], result['seed']) == (name, tf, kind, pack, seed)
        assert result['train_end'] == str(max(m.times[-1] for m in markets))
        return result
    started = time.monotonic()
    prepared = [prepare(m, tf) for m in markets]
    benchmarks = [population_accounting(d, alpha_target(d.x)[None, :]) for d in prepared]
    matrices = [inputs(d.x, pack) for d in prepared]
    n = len(PACKS[pack])
    if kind in ['linear', 'mlp']:
        parameters, connections = dimensions(kind, n)
        def fitness(weights):
            results = [population_accounting(d, learned_target(forward(weights, x, kind), d.x))
                       for d, x in zip(prepared, matrices)]
            return scores(results, benchmarks, connections)
        weights, value, history = fit(fitness, parameters, seed, POPULATION, GENERATIONS)
        result = {'weights': weights, 'fitness': value, 'connections': connections, 'parameters': parameters}
    else:
        config = model_config(POPULATION, n)
        population = neat.Population(config, seed=seed)
        trace = Trace(name)
        population.add_reporter(trace)
        evaluations = []
        def fitness(genomes, c):
            assert len(genomes) == POPULATION
            evaluations.append(len(genomes))
            nets = [neat.nn.FeedForwardNetwork.create(g, c) for _, g in genomes]
            results, cash = [], np.ones(len(nets), bool)
            for d, x in zip(prepared, matrices):
                p = learned_target(np.stack([batch_activate(net, x) for net in nets]), d.x)
                results.append(population_accounting(d, p))
                cash &= np.all(p == 0, axis=1)
            values = scores(results, benchmarks, [len(g.connections) for _, g in genomes])
            for i, (_, genome) in enumerate(genomes):
                genome.fitness, genome.research_cash = float(values[i]), bool(cash[i])
        winner = population.run(fitness, GENERATIONS)
        result = {'genome': winner, 'connections': len(winner.connections), 'nodes': len(winner.nodes)}
        history = trace.rows
        assert len(history) == len(evaluations) == GENERATIONS
        for row, count in zip(history, evaluations):
            row['candidate_evaluations'] = count
    count = sum(row['candidate_evaluations'] for row in history)
    assert count == POPULATION*GENERATIONS
    result.update({'name': name, 'kind': kind, 'timeframe': tf, 'pack': pack, 'seed': seed,
        'input_features': PACKS[pack], 'input_count': n, 'candidate_evaluations': count,
        'train_end': str(max(m.times[-1] for m in markets)), 'training_seconds': time.monotonic()-started})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(pickle.dumps(result))
    history_tmp = path.with_suffix('.csv.tmp')
    pd.DataFrame(history).to_csv(history_tmp, index=False)
    meta_tmp = path.with_suffix('.json.tmp')
    write_json(meta_tmp, {'protocol_sha256': protocol_hash,
        'model_sha256': sha256(temporary.read_bytes()).hexdigest(),
        'history_sha256': sha256(history_tmp.read_bytes()).hexdigest()})
    history_tmp.replace(path.with_suffix('.csv'))
    meta_tmp.replace(path.with_suffix('.json'))
    temporary.replace(path)
    print(f'{name}: complete in {result["training_seconds"]:.1f}s', flush=True)
    return result


def candidates_for(models, tf, fold):
    candidates = {n: n for n in CONTROLS}
    for kind, pack in SPECS:
        spec = kind+'_'+pack
        members = [models[f'{tf}_fold{min(fold, 4)}_{spec}_{seed}'] for seed in SEEDS]
        candidates[spec] = members
        candidates.update({f'{spec}_seed{seed}': m for seed, m in zip(SEEDS, members)})
    return candidates


def grouped(parts, records, endpoints):
    r = np.concatenate(parts)
    result = summary(metrics(r))
    daily = daily_returns(r, endpoints)
    result['annual_returns'] = {str(y): float(np.prod(1+v)-1) for y, v in daily.groupby(daily.index.year)}
    days = (pd.DatetimeIndex(endpoints).tz_localize(None)-pd.Timedelta(nanoseconds=1)).to_period('D')
    counts = pd.Series(1, index=days).groupby(level=0).sum()
    full = daily[counts == 96]
    result['daily_sharpe'] = float(full.mean()/full.std(ddof=0)*np.sqrt(365.25)) if full.std(ddof=0)>1e-12 else 0.
    result['by_symbol_return'] = {s: float(np.prod([1+v['by_symbol'][s]['return'] for v in records])-1) for s in SYMBOLS}
    result['orders'] = sum(v['by_symbol'][s]['order_count'] for v in records for s in SYMBOLS)
    result['mean_abs_exposure'] = float(np.average([v['by_symbol'][s]['mean_abs_exposure'] for v in records for s in SYMBOLS],
        weights=[v['observations'] for v in records for _ in SYMBOLS]))
    return result


def contrasts():
    pairs = []
    for tf in FRAMES:
        for kind, pack in SPECS:
            name = f'{tf}_{kind}_{pack}'
            pairs.append((name, tf+'_alpha'))
            if pack != 'alpha':
                pairs.append((name, f'{tf}_{kind}_alpha'))
    assert len(pairs)*2 == INTERVAL_COUNT
    return pairs


def paired_interval(parts, refs):
    differences = [np.log1p(a)-np.log1p(b) for a, b in zip(parts, refs)]
    intervals = {}
    for block in [7, 28]:
        rng = np.random.default_rng(1011)
        draws = np.zeros(DRAWS)
        for d in differences:
            n, number = len(d), int(np.ceil(len(d)/block))
            remainder = n-(number-1)*block
            cumulative = np.r_[0., np.cumsum(np.tile(d, int(np.ceil(block/n))+2))]
            for i in range(0, DRAWS, 2000):
                starts = rng.integers(0, n, size=(2000, number))
                draws[i:i+2000] += (cumulative[starts[:, :-1]+block]-cumulative[starts[:, :-1]]).sum(axis=1)
                draws[i:i+2000] += cumulative[starts[:, -1]+remainder]-cumulative[starts[:, -1]]
        intervals[str(block)] = np.quantile(draws, [.025/INTERVAL_COUNT, 1-.025/INTERVAL_COUNT]).tolist()
    return {'log_wealth_excess': float(sum(d.sum() for d in differences)), 'adjusted_intervals': intervals,
        'draws': DRAWS, 'intervals_corrected': INTERVAL_COUNT,
        'nonzero_days': int(sum((np.abs(d)>1e-12).sum() for d in differences))}


def run():
    if (OUT/'report.json').exists():
        raise ValueError('Completed experiment is immutable')
    OUT.mkdir(parents=True, exist_ok=True)
    raw_count = verify_raw_sources()
    if (OUT/'protocol.json').exists():
        protocol = json.loads((OUT/'protocol.json').read_text())
        for path, digest in {**protocol['source_hashes'], **protocol['evidence_hashes']}.items():
            assert sha256(Path(path).read_bytes()).hexdigest() == digest, path
    else:
        hashes = snapshot(OUT)
        for p in [Path('reports/study011_method.md'), Path('requirements-study011.txt'),
                  Path('requirements-study009.txt'), Path('requirements-study010.txt'),
                  Path('pytest.ini'), *Path('tests').glob('test_*.py')]:
            target = OUT/'source'/p
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
            hashes[p.as_posix()] = sha256(p.read_bytes()).hexdigest()
        evidence = [Path('artifacts/experiment_008/manifest.json'), Path('artifacts/experiment_008/protocol.json'),
                    Path('artifacts/experiment_008/report.json'), Path('artifacts/study011_preflight.json')]
        protocol = {'study': '011', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'source_hashes': hashes, 'evidence_hashes': {p.as_posix(): sha256(p.read_bytes()).hexdigest() for p in evidence},
            'runtime': {'python': sys.version, 'packages': {d.metadata['Name']: d.version for d in distributions()}},
            'timeframes': FRAMES, 'packs': PACKS, 'kinds': KINDS, 'columns': COLUMNS,
            'windows': WINDOWS, 'seeds': SEEDS, 'population': POPULATION, 'generations': GENERATIONS,
            'training_jobs': 540, 'candidate_budget': 622080, 'scenarios': SCENARIOS,
            'contrasts': contrasts(), 'adjusted_intervals': INTERVAL_COUNT,
            'raw_archives': raw_count, 'selected': None, 'known_history_only': True,
            'depth_inputs': 0, 'depth_placeholder_inputs': 0, 'availability_fallback': False}
        write_json(OUT/'protocol.json', protocol)
        shutil.copyfile(evidence[0], OUT/'manifest.json')
    fingerprint = sha256((OUT/'protocol.json').read_bytes()).hexdigest()
    bars, frames, quality = dataset()
    write_json(OUT/'data_quality.json', quality)
    jobs = []
    for tf in FRAMES:
        for fold, start in enumerate([a for a, _ in FOLDS]+['2026-01-01']):
            training = train_markets(bars, frames[tf], start)
            jobs += [(training, tf, kind, pack, seed, f'{tf}_fold{fold}_{kind}_{pack}_{seed}', fingerprint)
                     for kind, pack in SPECS for seed in SEEDS]
    started = time.monotonic()
    assert len(jobs) == protocol['training_jobs']
    with ProcessPoolExecutor(max_workers=3) as pool:
        models = {m['name']: m for m in pool.map(train_one, jobs)}
    assert sum(m['candidate_evaluations'] for m in models.values()) == protocol['candidate_budget']
    rows, decisions, all_daily = [], [], {}
    for tf in FRAMES:
        records, sequences, endpoints = {}, {}, []
        for fold, (start, end) in enumerate(WINDOWS):
            markets = {s: market(s, bars[s], frames[tf][s], start, end) for s in SYMBOLS}
            endpoints.append(markets[SYMBOLS[0]].times[1:])
            candidates = candidates_for(models, tf, fold)
            for arm, candidate in candidates.items():
                name = tf+'_'+arm
                for scenario, options in SCENARIOS:
                    r = evaluate(candidate, markets, tf, **options)
                    port = portfolio(r)
                    records.setdefault((name, scenario), []).append({'window': [start, end], 'observations': len(port['returns']),
                        'portfolio': summary(port), 'by_symbol': {s: summary(v) for s, v in r.items()}})
                    sequences.setdefault((name, scenario), []).append(port['returns'])
                if isinstance(candidate, list):
                    for s, m in markets.items():
                        p, a = desired(candidate, m, tf)
                        alpha, _ = desired('alpha', m, tf)
                        decisions.append({'timeframe': tf, 'name': name, 'fold': fold, 'symbol': s,
                            'decisions': int(a.sum()), 'target_differences_vs_alpha': int((np.abs(p[a]-alpha[a])>1e-12).sum()),
                            'long_while_alpha_flat': int(((p[a]>0) & (alpha[a]==0)).sum()),
                            'flat_fraction': float(np.mean(p[a] == 0))})
            checkpoint = OUT/'evaluation'/f'{tf}_window{fold}.json'
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            payload = {'protocol_sha256': fingerprint,
                'records': {name+'__'+scenario: parts[-1] for (name, scenario), parts in records.items()}}
            if checkpoint.exists():
                assert json.loads(checkpoint.read_text()) == payload
            else:
                temp = checkpoint.with_suffix('.tmp')
                write_json(temp, payload)
                temp.replace(checkpoint)
            print(f'Evaluated {tf} window{fold}', flush=True)
        for arm in candidates:
            name = tf+'_'+arm
            row = {'name': name, 'timeframe': tf, 'arm': arm, 'folds': {s: records[name, s] for s, _ in SCENARIOS}}
            for group, sl in [('development', slice(0, 4)), ('recent', slice(4, 6))]:
                dates = pd.DatetimeIndex(np.concatenate([v.to_numpy() for v in endpoints[sl]]))
                row[group] = {s: grouped(sequences[name, s][sl], records[name, s][sl], dates) for s, _ in SCENARIOS}
            rows.append(row)
            all_daily[name] = [daily_returns(r, t).to_numpy() for r, t in zip(sequences[name, 'base'][:4], endpoints[:4])]
        for group, sl in [('development', slice(0, 4)), ('recent', slice(4, 6))]:
            dates = pd.DatetimeIndex([pd.Timestamp(WINDOWS[sl.start][0], tz='UTC')]+[t for e in endpoints[sl] for t in e])
            curves = {name+'__'+s: metrics(np.concatenate(parts[sl]))['equity'] for (name, s), parts in sequences.items()}
            np.savez_compressed(OUT/f'{tf}_{group}_equity.npz', timestamps=dates.as_unit('ns').asi8, **curves)
    for parts in all_daily.values():
        assert [len(v) for v in parts] == [(pd.Timestamp(z)-pd.Timestamp(a)).days for a, z in FOLDS]
    comparisons = [{'candidate': c, 'reference': r, **paired_interval(all_daily[c], all_daily[r])} for c, r in contrasts()]
    assert len(rows) == 159
    write_json(OUT/'report.json', {'strategies': rows, 'comparisons': comparisons, 'decision_diagnostics': decisions,
        'training_times': [{k: m[k] for k in ['name', 'kind', 'timeframe', 'pack', 'input_count', 'connections', 'training_seconds']} for m in models.values()],
        'selected': None, 'training_jobs': len(jobs), 'actual_candidate_evaluations': sum(m['candidate_evaluations'] for m in models.values()),
        'strategy_scenario_windows': len(rows)*len(SCENARIOS)*len(WINDOWS), 'elapsed_seconds': time.monotonic()-started})
    print('Study011 complete. No model selected for deployment.', flush=True)


if __name__ == '__main__':
    run()
