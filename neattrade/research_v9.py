"""Matched NEAT/CMA-ES learners with zero or book imbalance input."""
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

from .alphatrend import alpha_features, ALPHA_FEATURES
from .batch import DailyMarket, fitness_accounting
from .market import Market, simulate, summary
from .orderbook import ROOT, SYMBOLS
from .policies import mlp, CONNECTIONS
from .evolution import fit_cma
from .overlay import incremental_objective
from .research import batch_activate
from .research_v3 import FOLDS, SEEDS, Trace, snapshot
from .research_v5 import model_config
from .research_v6 import write_json
from .sources_v8 import load
from .timeframes import aggregate, common_volatility, metrics, portfolio, daily_returns, PER_YEAR

OUT = Path('artifacts/experiment_009')
RECENT = [('2026-01-01', '2026-07-01'), ('2026-07-01', '2026-08-31 20:00')]
WINDOWS = FOLDS+RECENT
TRAIN_START = '2023-01-02'
SPECS = ['at_zero', 'at_book', 'cma_zero', 'cma_book']
POPULATION, GENERATIONS = 48, 24
SCENARIOS = [('base', {}), ('double_cost', {'multiplier': 2.}),
             ('delay15m', {'delay': 1}), ('book_lag4h', {'lag_book': True}),
             ('book_zeroed', {'zero_book': True})]


def alpha_inputs(bars):
    result=alpha_features(aggregate(bars,'4h')).reindex(bars.index,method='ffill')
    if list(result.columns)!=list(ALPHA_FEATURES):
        raise ValueError('Unexpected AlphaTrend feature order')
    return result


def population_scores(results,benchmarks,connections):
    value=incremental_objective(results,benchmarks)-.0001*np.asarray(connections)
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite objective; shared optimizer failure policy')
    return value


def dataset():
    meta = json.loads((ROOT/'feature_manifest.json').read_text())
    assert sha256((ROOT/'manifest.json').read_bytes()).hexdigest() == meta['manifest_sha256']
    assert sha256(Path('neattrade/orderbook.py').read_bytes()).hexdigest() == meta['module_sha256']
    bars, frames, late = {}, {}, {}
    for s in SYMBOLS:
        path = ROOT/f'{s}_features.csv'
        assert sha256(path.read_bytes()).hexdigest() == meta['symbols'][s]['feature_sha256']
        b, _ = load(s)
        bars[s] = b
        alpha = alpha_inputs(b)
        book = pd.read_csv(path, index_col=0, parse_dates=True)
        book.index = pd.to_datetime(book.index, utc=True)
        book = book.reindex(b.index)
        f = alpha.copy()
        f['book'] = book.book_imbalance.fillna(0.)
        f['book_valid'] = book.book_valid.fillna(False).astype(bool)
        f['vol'] = common_volatility(b)
        frames[s] = f
        late[s] = f.copy()
        late[s][['book', 'book_valid']] = f[['book', 'book_valid']].shift(16)
        late[s]['book'] = late[s].book.fillna(0.)
        late[s]['book_valid'] = late[s].book_valid.fillna(False).astype(bool)
    return bars, frames, late, meta


def market(symbol, b, f, start, end):
    start, end = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')
    ix = b.index[(b.index >= start) & (b.index <= end)]
    if len(ix) < 2 or ix[0] != start or ix[-1] != end:
        raise ValueError('Exact market endpoints unavailable')
    x = f.reindex(ix[:-1]).to_numpy(float)
    if x.shape[1] != 7 or not np.isfinite(x).all():
        raise ValueError('Expected four AT inputs, book, validity, volatility')
    block = b.reindex(ix)
    return Market(symbol, ix, x, block.open.to_numpy(), block.mark.to_numpy(),
                  block.funding.to_numpy(), block.low.to_numpy()[:-1], block.high.to_numpy()[:-1])


def allowed_times(m):
    ix = m.times[:-1]
    allowed = np.asarray((ix.hour % 4 == 0) & (ix.minute == 0))
    allowed[0] = True
    return allowed


def prepare(m, allowed=None):
    allowed = allowed_times(m) if allowed is None else np.asarray(allowed)
    anchors = np.unique(np.r_[0, np.flatnonzero(allowed)])
    segments = [(int(a), int(z), m.prices[a+1:z+1]-m.prices[a]
                 - np.cumsum(m.marks[a+1:z+1]*m.funding[a+1:z+1]))
                for a, z in zip(anchors, np.r_[anchors[1:], len(m.x)])]
    return DailyMarket(m, anchors, m.x[anchors], segments)


def base_targets(x):
    return np.maximum(x[:, 0], 0.)*np.minimum(.5, .20/np.maximum(x[:, 6], .10))


def network_inputs(x, spec, zero_book=False):
    if spec not in SPECS:
        raise ValueError('Unknown learned specification')
    value = x[:, :5].copy()
    if spec.endswith('_zero') or zero_book:
        value[:, 4] = 0.
    return value


def targets(raw, x):
    raw = np.asarray(raw)
    scale = np.where(raw < -.35, 0., np.where(raw > .35, 1., .5))
    # This external fallback is shared by BOTH matched learners.
    return np.where(x[:, 5].astype(bool), scale, 1.)*base_targets(x)


def train_markets(bars, frames, test_start):
    end = pd.Timestamp(test_start)-pd.Timedelta(days=7)
    bounds = [pd.Timestamp(TRAIN_START)]+list(pd.date_range('2023-07-01', end, freq='6MS'))+[end]
    return [market(f'{s}_{a.date()}', b, frames[s], str(a.date()), str(z.date()))
            for s, b in bars.items() for a, z in zip(bounds[:-1], bounds[1:]) if z > a]


def train_one(job):
    markets, spec, seed, name, protocol_hash = job
    path = OUT/'jobs'/f'{name}.pkl'
    if path.exists():
        meta = json.loads(path.with_suffix('.json').read_text())
        assert meta['protocol_sha256'] == protocol_hash
        assert meta['model_sha256'] == sha256(path.read_bytes()).hexdigest()
        with path.open('rb') as f:
            return pickle.load(f)
    job_started=time.monotonic()
    prepared = [prepare(m) for m in markets]
    benchmarks = [fitness_accounting(d, base_targets(d.x)[None, :], rebalance_band=.05) for d in prepared]
    for d,b in zip(prepared,benchmarks): b['years']=len(d.market.x)/PER_YEAR
    values = [network_inputs(d.x, spec) for d in prepared]
    if spec.startswith('cma_'):
        def fitness_cma(weights):
            results=[]
            for d,x in zip(prepared,values):
                r=fitness_accounting(d,targets(mlp(weights,x),d.x),rebalance_band=.05)
                r['years']=len(d.market.x)/PER_YEAR
                results.append(r)
            return population_scores(results,benchmarks,CONNECTIONS)
        weights,fitness,history=fit_cma(fitness_cma,seed,POPULATION,GENERATIONS)
        result={'name':name,'kind':'cma','spec':spec,'seed':seed,'weights':weights,'fitness':fitness,
                'train_end':str(max(m.times[-1] for m in markets)), 'training_seconds':time.monotonic()-job_started}
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.tmp')
        temporary.write_bytes(pickle.dumps(result));temporary.replace(path)
        pd.DataFrame(history).to_csv(path.with_suffix('.csv'),index=False)
        write_json(path.with_suffix('.json'),{'protocol_sha256':protocol_hash,
            'model_sha256':sha256(path.read_bytes()).hexdigest(),'train_end':result['train_end']})
        print(f'{name}: CMA-ES completed {POPULATION*GENERATIONS} evaluations, fitness={fitness:.4f}',flush=True)
        return result
    config = model_config(POPULATION, inputs=5)
    population = neat.Population(config, seed=seed)
    trace = Trace(name)
    population.add_reporter(trace)

    def fitness(genomes, c):
        nets = [neat.nn.FeedForwardNetwork.create(g, c) for _, g in genomes]
        results, cash = [], np.ones(len(nets), bool)
        for d, x in zip(prepared, values):
            raw = np.stack([batch_activate(net, x) for net in nets])
            p = targets(raw, d.x)
            r = fitness_accounting(d, p, rebalance_band=.05)
            r['years'] = len(d.market.x)/PER_YEAR
            results.append(r)
            cash &= np.all(p == 0, axis=1)
        scores = population_scores(results, benchmarks,[len(g.connections) for _,g in genomes])
        for i, (_, genome) in enumerate(genomes):
            genome.fitness = float(scores[i])
            genome.research_cash = bool(cash[i])

    winner = population.run(fitness, GENERATIONS)
    result = {'name': name, 'kind': 'neat', 'spec': spec, 'seed': seed, 'genome': winner,
              'train_end': str(max(m.times[-1] for m in markets)), 'training_seconds':time.monotonic()-job_started}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(pickle.dumps(result))
    temporary.replace(path)
    pd.DataFrame(trace.rows).to_csv(path.with_suffix('.csv'), index=False)
    write_json(path.with_suffix('.json'), {'protocol_sha256': protocol_hash,
        'model_sha256': sha256(path.read_bytes()).hexdigest(), 'train_end': result['train_end']})
    return result


def desired(candidate, m, zero_book=False):
    allowed = allowed_times(m)
    x = m.x[allowed]
    if isinstance(candidate, list):
        p = np.mean([desired(c, m, zero_book)[0][allowed] for c in candidate], axis=0)
    elif isinstance(candidate, dict):
        selected=network_inputs(x,candidate['spec'],zero_book)
        if candidate.get('kind')=='cma':
            raw=mlp(candidate['weights'],selected)[0]
        else:
            config = model_config(POPULATION, 5)
            net = neat.nn.FeedForwardNetwork.create(candidate['genome'], config)
            raw = batch_activate(net, selected)
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


def evaluate(candidate, markets, multiplier=1., delay=0, zero_book=False):
    results = {}
    for s, m in markets.items():
        p, allowed = desired(candidate, m, zero_book)
        if delay:
            p = np.r_[np.zeros(delay), p[:-delay]]
            allowed = np.r_[np.zeros(delay, bool), allowed[:-delay]]
        result = simulate(m, p, 5*multiplier, 2*multiplier, trade_allowed=allowed, rebalance_band=.05)
        result['sharpe'] = metrics(result['returns'])['sharpe']
        result['fallback_decisions'] = int((~m.x[allowed_times(m), 5].astype(bool)).sum())
        result['total_decisions'] = int(allowed_times(m).sum())
        results[s] = result
    return results


def paired_interval(parts, refs):
    differences = [np.log1p(a)-np.log1p(b) for a, b in zip(parts, refs)]
    intervals = {}
    for block in [7, 28]:
        rng = np.random.default_rng(909)
        draws = np.zeros(100000)
        for d in differences:
            for i in range(0, len(draws), 2000):
                starts = rng.integers(0, len(d), size=(2000, int(np.ceil(len(d)/block))))
                ix = ((starts[:, :, None]+np.arange(block)) % len(d)).reshape(2000, -1)[:, :len(d)]
                draws[i:i+2000] += d[ix].sum(axis=1)
        intervals[str(block)] = np.quantile(draws, [.025/6, 1-.025/6]).tolist()
    return {'log_wealth_excess': float(sum(d.sum() for d in differences)), 'adjusted_intervals': intervals,
            'draws': 100000, 'intervals_corrected': 6, 'nonzero_days': int(sum((np.abs(d)>1e-12).sum() for d in differences))}


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


def run():
    if (OUT/'report.json').exists():
        raise ValueError('Completed experiment is immutable')
    OUT.mkdir(parents=True, exist_ok=True)
    bars, frames, late, audit = dataset()
    if (OUT/'protocol.json').exists():
        protocol = json.loads((OUT/'protocol.json').read_text())
        for p, h in protocol['source_hashes'].items():
            assert sha256(Path(p).read_bytes()).hexdigest() == h, p
        assert protocol['feature_manifest_sha256'] == sha256((ROOT/'feature_manifest.json').read_bytes()).hexdigest()
    else:
        hashes = snapshot(OUT)
        for p in [Path('reports/study009_method.md'), Path('requirements-study009.txt'), Path('pytest.ini'), *Path('tests').glob('test_*.py')]:
            target = OUT/'source'/p
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
            hashes[p.as_posix()] = sha256(p.read_bytes()).hexdigest()
        previous = json.loads(Path('artifacts/experiment_008/manifest.json').read_text())
        books = json.loads((ROOT/'manifest.json').read_text())
        write_json(OUT/'manifest.json', {'market': previous, 'book': books})
        shutil.copyfile(ROOT/'feature_manifest.json', OUT/'feature_manifest.json')
        protocol = {'study': '009', 'source_hashes': hashes, 'specs': SPECS, 'windows': WINDOWS, 'seeds': SEEDS,
            'population': POPULATION, 'generations': GENERATIONS, 'train_start': TRAIN_START, 'embargo_days': 7,
            'feature_manifest_sha256': sha256((ROOT/'feature_manifest.json').read_bytes()).hexdigest(),
            'manifest_sha256': sha256((OUT/'manifest.json').read_bytes()).hexdigest(), 'selected': None,
            'training_jobs': 60, 'genome_evaluation_budget': 60*POPULATION*GENERATIONS,
            'architecture_amendment': 'Before training, user asked about replacing NEAT: add fixed5-4-1 MLP with CMA-ES as a matched comparison.',
            'cma': {'version':'4.4.4','architecture':[5,4,1],'parameters':29,'initial_mean':0.,'sigma':.5,'bounds':[-5,5],
                    'population':POPULATION,'generations':GENERATIONS,'connection_penalty':.0001*CONNECTIONS},
            'input_names': ['alpha_state', 'alpha_distance', 'alpha_slope', 'alpha_mfi', 'book_or_zero'],
            'scenarios': SCENARIOS, 'known_development_and_recent_context': True}
        write_json(OUT/'protocol.json', protocol)
    fingerprint = sha256((OUT/'protocol.json').read_bytes()).hexdigest()
    jobs = []
    for fold, start in enumerate([a for a, _ in FOLDS]+['2026-01-01']):
        training = train_markets(bars, frames, start)
        jobs += [(training, spec, seed, f'fold{fold}_{spec}_{seed}', fingerprint) for spec in SPECS for seed in SEEDS]
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=3) as pool:
        models = {m['name']: m for m in pool.map(train_one, jobs)}
    records, sequences, endpoints, decisions = {}, {}, [], []
    for fold, (start, end) in enumerate(WINDOWS):
        markets = {s: market(s, bars[s], frames[s], start, end) for s in SYMBOLS}
        lagged = {s: market(s, bars[s], late[s], start, end) for s in SYMBOLS}
        endpoints.append(markets[SYMBOLS[0]].times[1:])
        fitted = min(fold, 4)
        candidates = {n: n for n in ['alpha', 'alpha_half', 'passive_vol', 'cash']}
        for spec in SPECS:
            members = [models[f'fold{fitted}_{spec}_{seed}'] for seed in SEEDS]
            candidates[spec] = members
            candidates.update({f'{spec}_seed{seed}': member for seed, member in zip(SEEDS, members)})
        for name, candidate in candidates.items():
            for scenario, options in SCENARIOS:
                opts = dict(options)
                use = lagged if opts.pop('lag_book', False) else markets
                r = evaluate(candidate, use, **opts)
                port = portfolio(r)
                records.setdefault((name, scenario), []).append({'window': [start, end], 'observations': len(port['returns']),
                    'portfolio': summary(port), 'by_symbol': {s: summary(v) for s, v in r.items()}})
                sequences.setdefault((name, scenario), []).append(port['returns'])
            if name in SPECS:
                for s, m in markets.items():
                    p, a = desired(candidate, m)
                    off, _ = desired(candidate, m, zero_book=True)
                    decisions.append({'fold': fold, 'symbol': s, 'name': name, 'total': int(a.sum()),
                        'missing_fallbacks': int((~m.x[a,5].astype(bool)).sum()),
                        'book_zeroing_changed_targets': int((np.abs(p[a]-off[a])>1e-12).sum())})
        print(f'Evaluated window{fold}', flush=True)
    rows = []
    for name in candidates:
        row = {'name': name, 'folds': {s: records[name,s] for s, _ in SCENARIOS}}
        for group, sl in [('development', slice(0,4)), ('recent', slice(4,6))]:
            dates = pd.DatetimeIndex(np.concatenate([v.to_numpy() for v in endpoints[sl]]))
            row[group] = {s: grouped(sequences[name,s][sl], records[name,s][sl], dates) for s, _ in SCENARIOS}
        rows.append(row)
    comparisons=[]
    for candidate,reference in [('at_book','at_zero'),('cma_book','cma_zero'),('cma_book','at_book')]:
        comparison = paired_interval([daily_returns(r,t).to_numpy() for r,t in zip(sequences[candidate,'base'][:4], endpoints[:4])],
                                     [daily_returns(r,t).to_numpy() for r,t in zip(sequences[reference,'base'][:4], endpoints[:4])])
        comparisons.append({'candidate':candidate,'reference':reference,**comparison})
    for group, sl in [('development',slice(0,4)), ('recent',slice(4,6))]:
        start = WINDOWS[sl.start][0]
        dates = pd.DatetimeIndex([pd.Timestamp(start,tz='UTC')]+[t for e in endpoints[sl] for t in e])
        curves = {name+'__'+s: metrics(np.concatenate(parts[sl]))['equity'] for (name,s),parts in sequences.items()}
        pd.DataFrame(curves,index=dates).to_csv(OUT/f'{group}_equity.csv')
    write_json(OUT/'report.json', {'strategies': rows, 'comparisons': comparisons, 'decision_diagnostics': decisions,
        'training_times': [{'name':m['name'],'kind':m['kind'],'seconds':m['training_seconds']} for m in models.values()],
        'selected': None, 'training_jobs': len(jobs), 'genome_evaluation_budget': len(jobs)*POPULATION*GENERATIONS,
        'strategy_scenario_windows': len(candidates)*len(SCENARIOS)*len(WINDOWS),
        'elapsed_seconds': time.monotonic()-started})
    print(pd.DataFrame([{'name': r['name'], **r['development']['base']} for r in rows]).to_string(index=False), flush=True)


if __name__ == '__main__':
    run()
