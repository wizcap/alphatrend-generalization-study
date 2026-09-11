"""Complete fixed comparison, immutable fit registry and scalar published accounts."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import distributions
import json
from pathlib import Path
import pickle
import shutil
import sys
import time
import numpy as np
import pandas as pd

from .generalize_v15 import FRAMES, SYMBOLS, MONTHS, ARMS, specifications, key, train, policy, utc
from .repair_v12 import dataset, context, attribution
from .research_v12 import WINDOWS, SCENARIOS, repair_summary
from .research_v11 import grouped, verify_raw_sources
from .research_v3 import snapshot
from .research_v6 import write_json
from .market import simulate, summary
from .timeframes import portfolio, metrics, daily_returns

OUT = Path('artifacts/experiment_015')
DRAWS, INTERVAL_COUNT = 100000, 40


def contrasts():
    return [(tf+'_'+a, tf+'_'+b) for tf in FRAMES for a, b in [
        ('compact8', 'current'), ('robust8', 'compact8'), ('temporal_mean', 'robust8'),
        ('temporal_guard', 'temporal_mean'), ('ridge8', 'robust8'), ('transfer_current', 'current'),
        ('transfer_compact', 'compact8'), ('transfer_robust', 'robust8'),
        ('temporal_guard', 'current'), ('temporal_guard', 'alpha')]]


def interval(parts, refs):
    differences = [np.log1p(a)-np.log1p(b) for a, b in zip(parts, refs)]
    result = {}
    for block in [7, 28]:
        rng, draws = np.random.default_rng(1015), np.zeros(DRAWS)
        for d in differences:
            n, count = len(d), int(np.ceil(len(d)/block))
            cs = np.r_[0., np.cumsum(np.tile(d, 3))]
            for first in range(0, DRAWS, 2000):
                starts = rng.integers(0, n, (2000, count))
                draws[first:first+2000] += (cs[starts[:, :-1]+block]-cs[starts[:, :-1]]).sum(axis=1)
                draws[first:first+2000] += cs[starts[:, -1]+n-(count-1)*block]-cs[starts[:, -1]]
        result[str(block)] = np.quantile(draws, [.025/INTERVAL_COUNT, 1-.025/INTERVAL_COUNT]).tolist()
    return dict(log_wealth_excess=float(sum(d.sum() for d in differences)), intervals=result)


def diagnostics(actions, labels):
    catalog = {tf: {r['id']: r for r in rows if r['head']=='exit'} for tf, rows in labels.items()}
    result = []
    for a in actions:
        if a['scenario'] != 'base' or a['head'] != 'exit':
            continue
        tf = a['name'].split('_')[0]
        r = catalog[tf].get(f'{a["symbol"]}_{utc(a["entry"]).isoformat()}_exit')
        status = ('boundary' if a['boundary_protected'] else 'unmatched' if r is None else
                  'censored' if r['left_censored'] or r['known'] is None or utc(r['known'])>utc(WINDOWS[a['fold']][1]) else 'resolved')
        y = r['y'] if status=='resolved' else None
        score = float(np.mean(a['component_predictions'])) if a['component_predictions'] else None
        result.append({**a, 'status': status, 'raw_y': y,
            'payoff_prediction': score,
            'absolute_error': abs(score-y) if y is not None and score is not None else None,
            'regret': (max(y, 0.) if a['blocked'] else max(-y, 0.)) if y is not None else None,
            'alpha_regret': max(-y, 0.) if y is not None else None})
    return result


def freeze():
    OUT.mkdir(exist_ok=True, parents=True)
    path = OUT/'protocol.json'
    if path.exists():
        p = json.loads(path.read_text())
        for f, h in {**p['source_hashes'], **p['evidence_hashes']}.items():
            assert sha256(Path(f).read_bytes()).hexdigest()==h, f
        return sha256(path.read_bytes()).hexdigest()
    hashes = snapshot(OUT)
    for f in [Path('reports/study015_method.md'), Path('pytest.ini'),
              *Path('.').glob('requirements-study*.txt'), *Path('tests').glob('test_*.py')]:
        dest = OUT/'source'/f; dest.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(f, dest); hashes[f.as_posix()] = sha256(f.read_bytes()).hexdigest()
    evidence = [*[Path(f'artifacts/experiment_{s}')/n for s in ['012', '013']
                 for n in ['protocol.json', 'report.json', 'verification.json']],
                Path('artifacts/experiment_012/labels.json'), Path('artifacts/experiment_008/manifest.json'),
                Path('artifacts/study015_preflight.json'), Path('reports/study015_claude_design_input.txt'),
                Path('reports/study015_claude_design_review.txt'),
                *Path('artifacts/experiment_013/models').glob('*_expanding_*.pkl'),
                *[f for f in Path('artifacts/shadow_014').rglob('*') if f.is_file() and 'source' not in f.parts]]
    p = dict(study='015', created_utc=datetime.now(timezone.utc).isoformat(), source_hashes=hashes,
        evidence_hashes={f.as_posix(): sha256(f.read_bytes()).hexdigest() for f in evidence},
        arms=ARMS, months=MONTHS, timeframes=FRAMES, fit_slots=704, cached_references=64,
        windows=WINDOWS, scenarios=SCENARIOS, contrasts=contrasts(), interval_count=INTERVAL_COUNT,
        runtime=dict(python=sys.version, packages={d.metadata['Name']: d.version for d in distributions()}),
        known_history_only=True, production_selection=None)
    write_json(path, p)
    return sha256(path.read_bytes()).hexdigest()


def run():
    if (OUT/'report.json').exists():
        raise ValueError('Completed study015 cannot be overwritten')
    verify_raw_sources()
    protocol_hash = freeze()
    started = time.monotonic()
    bars, frames, atr = dataset()
    labels = json.loads(Path('artifacts/experiment_012/labels.json').read_text())
    models = {}
    (OUT/'models').mkdir(exist_ok=True); (OUT/'accounts').mkdir(exist_ok=True)
    for tf in FRAMES:
        for date in MONTHS:
            old = Path('artifacts/experiment_013/models')/f'{tf}_expanding_{utc(date):%Y%m}.pkl'
            m = pickle.loads(old.read_bytes()); models[m['name']] = m
            for family, days, heldout in specifications():
                name = key(tf, family, days, heldout, date); path = OUT/'models'/f'{name}.pkl'
                if path.exists():
                    meta = json.loads(path.with_suffix('.json').read_text())
                    assert meta['protocol_hash']==protocol_hash and meta['model_hash']==sha256(path.read_bytes()).hexdigest()
                    model = pickle.loads(path.read_bytes())
                else:
                    model = train(labels[tf], tf, family, days, heldout, date)
                    tmp = path.with_suffix('.tmp'); tmp.write_bytes(pickle.dumps(model))
                    write_json(path.with_suffix('.json'), dict(protocol_hash=protocol_hash, model_hash=sha256(tmp.read_bytes()).hexdigest()))
                    tmp.replace(path)
                assert model['name']==name
                models[name] = model
        print(f'{tf}: 352 new slots + 32 cached references ready', flush=True)
    all_daily, rows, actions, trades = {}, [], [], []
    for tf in FRAMES:
        sequences, records, endpoints = {}, {}, []
        for fold, (start, end) in enumerate(WINDOWS):
            contexts = {s: context(s, bars[s], frames[tf][s], atr[tf][s], start, end) for s in SYMBOLS}
            endpoints.append(contexts[SYMBOLS[0]]['market'].times[1:])
            arrays = {}
            for sc, options in SCENARIOS:
                baseline = {}
                for arm in ARMS:
                    name, results = tf+'_'+arm, {}
                    for symbol, ctx in contexts.items():
                        p, a, ev = policy(ctx, tf, arm, models, options['delay'])
                        r = simulate(ctx['market'], p, options['cost'], 0., trade_allowed=a, rebalance_band=.05)
                        r['sharpe'] = metrics(r['returns'])['sharpe']; results[symbol] = r
                        if arm=='alpha':
                            baseline[symbol] = r
                        prefix = f'{arm}__{sc}__{symbol}__'
                        arrays.update({prefix+'equity': r['equity'], prefix+'positions': p, prefix+'allowed': a})
                        actions += [dict(name=name, scenario=sc, fold=fold, symbol=symbol, **e) for e in ev]
                        trades += [dict(name=name, scenario=sc, fold=fold, symbol=symbol, **t)
                                   for t in attribution(ctx, tf, baseline[symbol], r, options['delay'])]
                    port = portfolio(results)
                    sequences.setdefault((name, sc), []).append(port['returns'])
                    records.setdefault((name, sc), []).append(dict(window=[start, end], observations=len(port['returns']),
                        portfolio=summary(port), by_symbol={s: summary(r) for s, r in results.items()}))
            np.savez_compressed(OUT/'accounts'/f'{tf}_{fold}.npz', **arrays)
            print(f'Scalar evaluation: {tf} window {fold} complete', flush=True)
        for arm in ARMS:
            name = tf+'_'+arm
            row = dict(name=name, timeframe=tf, arm=arm, folds={sc: records[name, sc] for sc, _ in SCENARIOS})
            for group, sl in [('development', slice(0, 4)), ('recent', slice(4, 6))]:
                dates = pd.DatetimeIndex(np.concatenate([t.to_numpy() for t in endpoints[sl]]))
                row[group] = {sc: grouped(sequences[name, sc][sl], records[name, sc][sl], dates) for sc, _ in SCENARIOS}
                row[group+'_repair'] = {}
                for sc, _ in SCENARIOS:
                    use = [t for t in trades if t['name']==name and t['scenario']==sc and t['fold'] in range(sl.start, sl.stop)]
                    row[group+'_repair'][sc] = dict(all=repair_summary(use), natural_only=repair_summary(
                        [t for t in use if not t['left_censored'] and not t['right_censored']]))
            rows.append(row)
            all_daily[name] = [daily_returns(r, t).to_numpy() for r, t in zip(sequences[name, 'base'][:4], endpoints[:4])]
        for group, sl in [('development', slice(0, 4)), ('recent', slice(4, 6))]:
            dates = pd.DatetimeIndex([utc(WINDOWS[sl.start][0])]+[t for e in endpoints[sl] for t in e])
            np.savez_compressed(OUT/f'{tf}_{group}_equity.npz', timestamps=dates.as_unit('ns').asi8,
                **{name+'__'+sc: metrics(np.concatenate(parts[sl]))['equity'] for (name, sc), parts in sequences.items()})
    comparisons = [dict(candidate=a, reference=b, **interval(all_daily[a], all_daily[b])) for a, b in contrasts()]
    for filename, value in [('actions', actions), ('trades', trades), ('diagnostics', diagnostics(actions, labels))]:
        write_json(OUT/(filename+'.json'), value)
    model_summary = [{k: m.get(k) for k in ['name', 'family', 'timeframe', 'available', 'cutoff', 'days', 'heldout',
        'training_count', 'last_label_known', 'effective_weight_count', 'fitted', 'clip_limits', 'group_counts']} for m in models.values()]
    write_json(OUT/'report.json', dict(strategies=rows, comparisons=comparisons, models=model_summary,
        new_fit_slots=704, new_models_fitted=sum(m.get('fitted', False) for m in models.values()),
        cached_references=64, asset_windows=720, portfolio_windows=360,
        elapsed_seconds=time.monotonic()-started, production_selection=None))
    print('Study015 research complete; no deployment selection.', flush=True)


if __name__=='__main__':
    run()
