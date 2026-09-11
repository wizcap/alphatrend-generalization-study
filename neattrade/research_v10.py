"""Frozen short-timeframe book-feature experiment, using study009 learners."""
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
from .evolution import fit_cma
from .fast_accounting import population_accounting
from .market import simulate, summary
from .orderbook import ROOT as BOOK_ROOT, SYMBOLS
from .policies import mlp, CONNECTIONS
from .research import batch_activate
from .research_v3 import FOLDS, SEEDS, Trace, snapshot
from .research_v5 import model_config
from .research_v6 import write_json
from .research_v9 import (SPECS, WINDOWS, POPULATION, GENERATIONS, train_markets, market,
                         base_targets, network_inputs, targets, population_scores, grouped)
from .short_book import ROOT, SHORT, alpha_inputs
from .sources_v8 import load
from .timeframes import TIMEFRAMES, common_volatility, metrics, portfolio, daily_returns

OUT = Path('artifacts/experiment_010')
SCENARIOS = [('base', {}), ('double_cost', {'multiplier': 2.}),
             ('delay15m', {'delay': 1}), ('book_lag1bar', {'lag_book': True}),
             ('book_zeroed', {'zero_book': True})]
INTERVAL_COUNT = 42


def verify_raw_sources():
    source = Path('artifacts/experiment_009/manifest.json')
    prior = json.loads(Path('artifacts/experiment_009/protocol.json').read_text())
    assert sha256(source.read_bytes()).hexdigest() == prior['manifest_sha256']
    manifest = json.loads(source.read_text())
    for item in manifest['market']+manifest['book']:
        if item.get('status') != 'missing_404':
            assert sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']


def dataset():
    meta = json.loads((ROOT/'feature_manifest.json').read_text())
    assert sha256(Path('neattrade/short_book.py').read_bytes()).hexdigest() == meta['module_sha256']
    assert sha256((BOOK_ROOT/'feature_manifest.json').read_bytes()).hexdigest() == meta['source_feature_manifest_sha256']
    for path, digest in meta['files'].items():
        assert sha256(Path(path).read_bytes()).hexdigest() == digest, path
    bars, frames, late = {}, {tf: {} for tf in SHORT}, {tf: {} for tf in SHORT}
    for s in SYMBOLS:
        b, _ = load(s)
        bars[s] = b
        vol = common_volatility(b)
        for tf in SHORT:
            book = pd.read_csv(ROOT/f'{s}_{tf}.csv', index_col=0, parse_dates=True)
            book.index = pd.to_datetime(book.index, utc=True)
            book = book.reindex(b.index)
            f = alpha_inputs(b, tf)
            f['book'] = book.book_imbalance.fillna(0.)
            f['book_valid'] = book.book_valid.fillna(False).astype(bool)
            f['vol'] = vol
            frames[tf][s] = f
            lagged = f.copy()
            lagged[['book', 'book_valid']] = f[['book', 'book_valid']].shift(TIMEFRAMES[tf]//15)
            lagged['book'] = lagged.book.fillna(0.)
            lagged['book_valid'] = lagged.book_valid.fillna(False).astype(bool)
            late[tf][s] = lagged
    return bars, frames, late, meta


def allowed_times(m, tf):
    ix = m.times[:-1]
    allowed = np.asarray((ix.hour*60+ix.minute) % TIMEFRAMES[tf] == 0)
    allowed[0] = True
    return allowed


def prepare(m, tf):
    anchors = np.flatnonzero(allowed_times(m, tf))
    return DailyMarket(m, anchors, m.x[anchors], [])


def desired(candidate, m, tf, zero_book=False):
    allowed = allowed_times(m, tf)
    x = m.x[allowed]
    if isinstance(candidate, list):
        p = np.mean([desired(c, m, tf, zero_book)[0][allowed] for c in candidate], axis=0)
    elif isinstance(candidate, dict):
        inputs = network_inputs(x, candidate['spec'], zero_book)
        if candidate['kind'] == 'cma':
            raw = mlp(candidate['weights'], inputs)[0]
        else:
            net = neat.nn.FeedForwardNetwork.create(candidate['genome'], model_config(POPULATION, 5))
            raw = batch_activate(net, inputs)
        p = targets(raw, x)
    elif candidate in ['alpha', 'alpha_half']:
        p = base_targets(x)*(.5 if candidate == 'alpha_half' else 1.)
    elif candidate == 'passive_vol':
        p = np.minimum(.5, .20/np.maximum(x[:, 6], .10))
    elif candidate == 'cash':
        p = np.zeros(len(x))
    else:
        raise ValueError('Unknown candidate')
    full = np.zeros(len(m.x))
    full[allowed] = p
    anchors = np.maximum.accumulate(np.where(allowed, np.arange(len(full)), 0))
    return full[anchors], allowed


def evaluate(candidate, markets, tf, multiplier=1., delay=0, zero_book=False):
    result = {}
    for s, m in markets.items():
        p, allowed = desired(candidate, m, tf, zero_book)
        if delay:
            p = np.r_[np.zeros(delay), p[:-delay]]
            allowed = np.r_[np.zeros(delay, bool), allowed[:-delay]]
        r = simulate(m, p, 5*multiplier, 2*multiplier, trade_allowed=allowed, rebalance_band=.05)
        r['sharpe'] = metrics(r['returns'])['sharpe']
        eligible = allowed_times(m, tf)
        r['fallback_decisions'] = int((~m.x[eligible, 5].astype(bool)).sum())
        r['total_decisions'] = int(eligible.sum())
        result[s] = r
    return result


def train_one(job):
    markets, tf, spec, seed, name, protocol_hash = job
    path = OUT/'jobs'/f'{name}.pkl'
    if path.exists():
        meta = json.loads(path.with_suffix('.json').read_text())
        assert meta['protocol_sha256'] == protocol_hash
        assert meta['model_sha256'] == sha256(path.read_bytes()).hexdigest()
        assert meta['history_sha256'] == sha256(path.with_suffix('.csv').read_bytes()).hexdigest()
        return pickle.loads(path.read_bytes())
    started = time.monotonic()
    prepared = [prepare(m, tf) for m in markets]
    benchmarks = [population_accounting(d, base_targets(d.x)[None, :]) for d in prepared]
    inputs = [network_inputs(d.x, spec) for d in prepared]
    if spec.startswith('cma_'):
        def fitness(weights):
            results = [population_accounting(d, targets(mlp(weights, x), d.x)) for d, x in zip(prepared, inputs)]
            return population_scores(results, benchmarks, CONNECTIONS)
        weights, value, history = fit_cma(fitness, seed, POPULATION, GENERATIONS)
        result = {'kind': 'cma', 'weights': weights, 'fitness': value}
    else:
        config = model_config(POPULATION, inputs=5)
        population = neat.Population(config, seed=seed)
        trace = Trace(name)
        population.add_reporter(trace)
        evaluations = []
        def fitness(genomes, c):
            assert len(genomes) == POPULATION
            evaluations.append(len(genomes))
            nets = [neat.nn.FeedForwardNetwork.create(g, c) for _, g in genomes]
            results = []
            cash = np.ones(len(nets), dtype=bool)
            for d, x in zip(prepared, inputs):
                raw = np.stack([batch_activate(net, x) for net in nets])
                p = targets(raw, d.x)
                results.append(population_accounting(d, p))
                cash &= np.all(p == 0, axis=1)
            scores = population_scores(results, benchmarks, [len(g.connections) for _, g in genomes])
            for i, (_, genome) in enumerate(genomes):
                genome.fitness = float(scores[i])
                genome.research_cash = bool(cash[i])
        winner = population.run(fitness, GENERATIONS)
        result = {'kind': 'neat', 'genome': winner}
        history = trace.rows
        assert len(history) == len(evaluations) == GENERATIONS
        for row, count in zip(history, evaluations):
            row['candidate_evaluations'] = count
    evaluation_count = sum(row['candidate_evaluations'] for row in history)
    assert evaluation_count == POPULATION*GENERATIONS
    result.update({'name': name, 'timeframe': tf, 'spec': spec, 'seed': seed,
        'train_end': str(max(m.times[-1] for m in markets)), 'training_seconds': time.monotonic()-started,
        'candidate_evaluations': evaluation_count})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(pickle.dumps(result))
    history_tmp = path.with_suffix('.csv.tmp')
    pd.DataFrame(history).to_csv(history_tmp, index=False)
    meta_tmp = path.with_suffix('.json.tmp')
    write_json(meta_tmp, {'protocol_sha256': protocol_hash,
        'model_sha256': sha256(temporary.read_bytes()).hexdigest(),
        'history_sha256': sha256(history_tmp.read_bytes()).hexdigest(), 'train_end': result['train_end']})
    history_tmp.replace(path.with_suffix('.csv'))
    meta_tmp.replace(path.with_suffix('.json'))
    # The model file is the final commit marker; incomplete side files can retry.
    temporary.replace(path)
    print(f'{name}: complete in {result["training_seconds"]:.1f}s', flush=True)
    return result


def candidates_for(models, tf, fold):
    candidates = {n: n for n in ['alpha', 'alpha_half', 'passive_vol', 'cash']}
    for spec in SPECS:
        members = [models[f'{tf}_fold{min(fold, 4)}_{spec}_{seed}'] for seed in SEEDS]
        candidates[spec] = members
        candidates.update({f'{spec}_seed{seed}': m for seed, m in zip(SEEDS, members)})
    return candidates


def paired_interval(parts, refs):
    differences = [np.log1p(a)-np.log1p(b) for a, b in zip(parts, refs)]
    intervals = {}
    for block in [7, 28]:
        rng = np.random.default_rng(1010)
        draws = np.zeros(100000)
        for d in differences:
            for i in range(0, len(draws), 2000):
                starts = rng.integers(0, len(d), size=(2000, int(np.ceil(len(d)/block))))
                ix = ((starts[:, :, None]+np.arange(block)) % len(d)).reshape(2000, -1)[:, :len(d)]
                draws[i:i+2000] += d[ix].sum(axis=1)
        intervals[str(block)] = np.quantile(draws, [.025/INTERVAL_COUNT, 1-.025/INTERVAL_COUNT]).tolist()
    return {'log_wealth_excess': float(sum(d.sum() for d in differences)),
        'adjusted_intervals': intervals, 'draws': 100000, 'intervals_corrected': INTERVAL_COUNT,
        'nonzero_days': int(sum((np.abs(d)>1e-12).sum() for d in differences))}


def run():
    if (OUT/'report.json').exists():
        raise ValueError('Completed experiment is immutable')
    OUT.mkdir(parents=True, exist_ok=True)
    verify_raw_sources()
    bars, frames, late, meta = dataset()
    preflight = json.loads(Path('artifacts/study010_preflight.json').read_text())
    assert preflight['feature_manifest_sha256'] == sha256((ROOT/'feature_manifest.json').read_bytes()).hexdigest()
    assert preflight['book_window_checks'] == 449904 and preflight['max_feature_error'] <= 2e-10
    if (OUT/'protocol.json').exists():
        protocol = json.loads((OUT/'protocol.json').read_text())
        for path, digest in protocol['source_hashes'].items():
            assert sha256(Path(path).read_bytes()).hexdigest() == digest, path
        for path, digest in protocol['evidence_hashes'].items():
            assert sha256(Path(path).read_bytes()).hexdigest() == digest, path
    else:
        hashes = snapshot(OUT)
        for p in [Path('reports/study010_method.md'), Path('requirements-study009.txt'),
                  Path('requirements-study010.txt'), Path('pytest.ini'), *Path('tests').glob('test_*.py')]:
            target = OUT/'source'/p
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
            hashes[p.as_posix()] = sha256(p.read_bytes()).hexdigest()
        paths = [ROOT/'feature_manifest.json', BOOK_ROOT/'feature_manifest.json',
                 Path('artifacts/study010_preflight.json'),
                 Path('artifacts/experiment_009/report.json'), Path('artifacts/experiment_009/protocol.json'),
                 Path('artifacts/experiment_009/verification.json'), Path('artifacts/experiment_009/development_equity.csv'),
                 Path('artifacts/experiment_008/report.json'), Path('artifacts/experiment_009/manifest.json')]
        protocol = {'study': '010', 'source_hashes': hashes,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'runtime': {'python': sys.version, 'packages': {d.metadata['Name']: d.version for d in distributions()}},
            'evidence_hashes': {p.as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths},
            'timeframes': SHORT, 'specs': SPECS, 'windows': WINDOWS, 'seeds': SEEDS,
            'population': POPULATION, 'generations': GENERATIONS, 'training_jobs': 180,
            'candidate_budget': 207360, 'scenarios': SCENARIOS, 'comparisons': 21,
            'adjusted_intervals': INTERVAL_COUNT, 'selected': None, 'known_history_only': True}
        write_json(OUT/'protocol.json', protocol)
        shutil.copyfile(ROOT/'feature_manifest.json', OUT/'feature_manifest.json')
    fingerprint = sha256((OUT/'protocol.json').read_bytes()).hexdigest()
    jobs = []
    for tf in SHORT:
        for fold, start in enumerate([a for a, _ in FOLDS]+['2026-01-01']):
            training = train_markets(bars, frames[tf], start)
            jobs += [(training, tf, spec, seed, f'{tf}_fold{fold}_{spec}_{seed}', fingerprint)
                     for spec in SPECS for seed in SEEDS]
    started = time.monotonic()
    assert len(jobs) == protocol['training_jobs'] == 180
    with ProcessPoolExecutor(max_workers=3) as pool:
        models = {m['name']: m for m in pool.map(train_one, jobs)}
    assert sum(m['candidate_evaluations'] for m in models.values()) == protocol['candidate_budget']
    rows, decisions, all_daily = [], [], {}
    for tf in SHORT:
        records, sequences, endpoints = {}, {}, []
        for fold, (start, end) in enumerate(WINDOWS):
            markets = {s: market(s, bars[s], frames[tf][s], start, end) for s in SYMBOLS}
            lagged = {s: market(s, bars[s], late[tf][s], start, end) for s in SYMBOLS}
            endpoints.append(markets[SYMBOLS[0]].times[1:])
            candidates = candidates_for(models, tf, fold)
            for arm, candidate in candidates.items():
                name = tf+'_'+arm
                for scenario, options in SCENARIOS:
                    opts = dict(options)
                    use = lagged if opts.pop('lag_book', False) else markets
                    r = evaluate(candidate, use, tf, **opts)
                    port = portfolio(r)
                    records.setdefault((name, scenario), []).append({'window': [start, end], 'observations': len(port['returns']),
                        'portfolio': summary(port), 'by_symbol': {s: summary(v) for s, v in r.items()}})
                    sequences.setdefault((name, scenario), []).append(port['returns'])
                if arm in SPECS:
                    for s, m in markets.items():
                        p, a = desired(candidate, m, tf)
                        off, _ = desired(candidate, m, tf, True)
                        alpha, _ = desired('alpha', m, tf)
                        decisions.append({'timeframe': tf, 'name': name, 'fold': fold, 'symbol': s,
                            'total': int(a.sum()), 'missing_fallbacks': int((~m.x[a, 5].astype(bool)).sum()),
                            'book_zeroing_changed_targets': int((np.abs(p[a]-off[a])>1e-12).sum()),
                            'target_differences_vs_alpha': int((np.abs(p[a]-alpha[a])>1e-12).sum())})
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
    old = pd.read_csv('artifacts/experiment_009/development_equity.csv', index_col=0)
    old.index = pd.to_datetime(old.index, utc=True)
    for arm in ['at_book', 'cma_book']:
        eq = old[arm+'__base']
        r = eq.pct_change().iloc[1:]
        all_daily['4h_'+arm] = [daily_returns(part.to_numpy(), part.index).to_numpy()
            for start, end in FOLDS for part in [r.loc[(r.index > pd.Timestamp(start, tz='UTC')) & (r.index <= pd.Timestamp(end, tz='UTC'))]]]
    contrasts = []
    for tf in SHORT:
        contrasts += [(tf+'_at_book', tf+'_at_zero'), (tf+'_cma_book', tf+'_cma_zero'),
                      (tf+'_cma_book', tf+'_at_book')]
        for arm in ['at_book', 'cma_book']:
            contrasts += [(tf+'_'+arm, tf+'_alpha'), (tf+'_'+arm, '4h_'+arm)]
    assert len(contrasts) == 21
    for parts in all_daily.values():
        assert [len(v) for v in parts] == [(pd.Timestamp(z)-pd.Timestamp(a)).days for a, z in FOLDS]
    assert len(rows)*len(SCENARIOS)*len(WINDOWS) == 1800
    comparisons = [{'candidate': c, 'reference': r, **paired_interval(all_daily[c], all_daily[r])} for c, r in contrasts]
    write_json(OUT/'report.json', {'strategies': rows, 'comparisons': comparisons,
        'decision_diagnostics': decisions, 'training_times': [{'name': m['name'], 'kind': m['kind'],
            'timeframe': m['timeframe'], 'seconds': m['training_seconds']} for m in models.values()],
        'selected': None, 'training_jobs': len(jobs), 'candidate_budget': len(jobs)*POPULATION*GENERATIONS,
        'actual_candidate_evaluations': sum(m['candidate_evaluations'] for m in models.values()),
        'strategy_scenario_windows': 1800, 'elapsed_seconds': time.monotonic()-started})
    print('Study010 completed: 180 jobs, 1800 portfolio scenario windows. No model selected.', flush=True)


if __name__ == '__main__':
    run()
