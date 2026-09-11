"""Frozen monthly learning comparison and prospective update-controller replay."""
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

from .adaptive_v13 import (FRAMES, SYMBOLS, MONTHS, ARMS, train_version, version_id,
    feedback_catalog, Controller, policy, utc, digest)
from .repair_v12 import dataset, context, attribution
from .research_v12 import WINDOWS, SCENARIOS, repair_summary
from .research_v11 import grouped, verify_raw_sources
from .research_v3 import snapshot
from .research_v6 import write_json
from .market import simulate, summary
from .timeframes import portfolio, metrics, daily_returns

OUT = Path('artifacts/experiment_013')
DRAWS, INTERVAL_COUNT = 100000, 24


def contrasts():
    return [(tf+'_'+a, tf+'_'+b) for tf in FRAMES for a, b in [
        ('monthly', 'frozen'), ('monthly', 'semiannual'), ('weighted', 'monthly'),
        ('guarded', 'weighted'), ('guarded', 'semiannual'), ('guarded', 'alpha')]]


def interval(parts, refs, direct=False):
    differences = [np.log1p(a)-np.log1p(b) for a, b in zip(parts, refs)]
    intervals = {}
    for block in [7, 28]:
        rng, draws = np.random.default_rng(1013), np.zeros(DRAWS)
        for d in differences:
            n, count = len(d), int(np.ceil(len(d)/block))
            cs = np.r_[0., np.cumsum(np.tile(d, 3))]
            for first in range(0, DRAWS, 2000):
                starts = rng.integers(0, n, (2000, count))
                if direct:
                    ix = ((starts[:, :, None]+np.arange(block))%n).reshape(2000, -1)[:, :n]
                    draws[first:first+2000] += d[ix].sum(axis=1)
                else:
                    draws[first:first+2000] += (cs[starts[:, :-1]+block]-cs[starts[:, :-1]]).sum(axis=1)
                    draws[first:first+2000] += cs[starts[:, -1]+n-(count-1)*block]-cs[starts[:, -1]]
        intervals[str(block)] = np.quantile(draws, [.025/INTERVAL_COUNT, 1-.025/INTERVAL_COUNT]).tolist()
    return {'log_wealth_excess': float(sum(d.sum() for d in differences)), 'intervals': intervals}


def run():
    if (OUT/'report.json').exists():
        raise ValueError('Completed study013 cannot be overwritten')
    OUT.mkdir(exist_ok=True, parents=True)
    verify_raw_sources()
    if (OUT/'protocol.json').exists():
        protocol = json.loads((OUT/'protocol.json').read_text())
        for f, h in {**protocol['source_hashes'], **protocol['evidence_hashes']}.items():
            assert sha256(Path(f).read_bytes()).hexdigest() == h, f
    else:
        hashes = snapshot(OUT)
        for f in [Path('reports/study013_method.md'), Path('pytest.ini'), *Path('.').glob('requirements-study*.txt'), *Path('tests').glob('test_*.py')]:
            dst = OUT/'source'/f; dst.parent.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(f, dst); hashes[f.as_posix()] = sha256(f.read_bytes()).hexdigest()
        files = [Path('artifacts/study013_preflight.json'), *[Path('artifacts/experiment_012')/n for n in ['protocol.json', 'labels.json', 'report.json', 'verification.json']],
                 Path('artifacts/experiment_008/manifest.json')]
        protocol = {'study': '013', 'created_utc': datetime.now(timezone.utc).isoformat(), 'source_hashes': hashes,
            'evidence_hashes': {f.as_posix(): sha256(f.read_bytes()).hexdigest() for f in files},
            'months': MONTHS, 'arms': ARMS, 'timeframes': FRAMES, 'fits': 128, 'windows': WINDOWS,
            'contrasts': contrasts(), 'scenarios': SCENARIOS, 'interval_count': INTERVAL_COUNT,
            'runtime': {'python': sys.version, 'packages': {d.metadata['Name']: d.version for d in distributions()}},
            'known_history_only': True, 'production_selection': None}
        write_json(OUT/'protocol.json', protocol)
    protocol_hash = sha256((OUT/'protocol.json').read_bytes()).hexdigest()
    started = time.monotonic()
    bars, frames, atr = dataset()
    labels = json.loads(Path('artifacts/experiment_012/labels.json').read_text())
    feedback = feedback_catalog(bars, frames, atr)
    preflight = json.loads(Path('artifacts/study013_preflight.json').read_text())
    for path, expected in preflight['module_hashes'].items():
        assert sha256(Path(path).read_bytes()).hexdigest() == expected, path
    assert digest(feedback) == preflight['feedback_sha256']
    write_json(OUT/'feedback.json', feedback)
    models, schedules = {}, {}
    (OUT/'models').mkdir(exist_ok=True)
    for tf in FRAMES:
        for date in MONTHS:
            for method in ['expanding', 'weighted']:
                name = version_id(tf, method, date); f = OUT/'models'/f'{name}.pkl'
                if f.exists():
                    meta = json.loads(f.with_suffix('.json').read_text())
                    assert meta['protocol_hash'] == protocol_hash and meta['feedback_hash'] == preflight['feedback_sha256']
                    assert sha256(f.read_bytes()).hexdigest() == meta['model_hash']
                    model = pickle.loads(f.read_bytes())
                else:
                    model = train_version(labels[tf], tf, method, date)
                    tmp = f.with_suffix('.tmp'); tmp.write_bytes(pickle.dumps(model))
                    meta_tmp = f.with_suffix('.json.tmp')
                    write_json(meta_tmp, {'protocol_hash': protocol_hash, 'feedback_hash': preflight['feedback_sha256'],
                        'model_hash': sha256(tmp.read_bytes()).hexdigest()})
                    meta_tmp.replace(f.with_suffix('.json')); tmp.replace(f)
                assert (model['name'], model['method'], model['available']) == (name, method, str(utc(date)))
                models[name] = model
        path = OUT/f'{tf}_controller.json'
        controller = Controller.load(path, tf, protocol_hash) if path.exists() else Controller(tf, protocol_hash)
        for date in MONTHS:
            if controller.state['last_boundary'] is None or utc(date) > utc(controller.state['last_boundary']):
                controller.step(date, feedback[tf], models)
                controller.save(path)
        schedules[tf] = {r['boundary']: r['active'] for r in controller.state['history']}
        print(f'{tf}:64 monthly fits; controller replayed32 updates', flush=True)
    all_daily, rows, actions, trades = {}, [], [], []
    for tf in FRAMES:
        sequences, records, endpoints = {}, {}, []
        for fold, (start, end) in enumerate(WINDOWS):
            contexts = {s: context(s, bars[s], frames[tf][s], atr[tf][s], start, end) for s in SYMBOLS}
            endpoints.append(contexts[SYMBOLS[0]]['market'].times[1:])
            for scenario, options in SCENARIOS:
                baseline = {}
                for arm in ARMS:
                    name, results = tf+'_'+arm, {}
                    for symbol, ctx in contexts.items():
                        p, a, ev = policy(ctx, tf, arm, models, schedules[tf], options['delay'])
                        r = simulate(ctx['market'], p, options['cost'], 0., trade_allowed=a, rebalance_band=.05)
                        r['sharpe'] = metrics(r['returns'])['sharpe']
                        results[symbol] = r
                        if arm == 'alpha':
                            baseline[symbol] = r
                        actions += [{'name': name, 'scenario': scenario, 'fold': fold, 'symbol': symbol, **e} for e in ev]
                        trades += [{'name': name, 'scenario': scenario, 'fold': fold, 'symbol': symbol, **t}
                                   for t in attribution(ctx, tf, baseline[symbol], r, options['delay'])]
                    port = portfolio(results)
                    sequences.setdefault((name, scenario), []).append(port['returns'])
                    records.setdefault((name, scenario), []).append({'window': [start, end], 'observations': len(port['returns']),
                        'portfolio': summary(port), 'by_symbol': {s: summary(r) for s, r in results.items()}})
            print(f'Published scalar evaluation:{tf} window{fold}', flush=True)
        for arm in ARMS:
            name = tf+'_'+arm
            row = {'name': name, 'timeframe': tf, 'arm': arm, 'folds': {sc: records[name, sc] for sc, _ in SCENARIOS}}
            for group, sl in [('development', slice(0, 4)), ('recent', slice(4, 6))]:
                dates = pd.DatetimeIndex(np.concatenate([t.to_numpy() for t in endpoints[sl]]))
                row[group] = {sc: grouped(sequences[name, sc][sl], records[name, sc][sl], dates) for sc, _ in SCENARIOS}
                row[group+'_repair'] = {}
                for sc, _ in SCENARIOS:
                    use = [t for t in trades if t['name'] == name and t['scenario'] == sc and t['fold'] in range(sl.start, sl.stop)]
                    row[group+'_repair'][sc] = {'all': repair_summary(use),
                        'natural_only': repair_summary([t for t in use if not t['left_censored'] and not t['right_censored']])}
            rows.append(row)
            all_daily[name] = [daily_returns(r, t).to_numpy() for r, t in zip(sequences[name, 'base'][:4], endpoints[:4])]
        for group, sl in [('development', slice(0, 4)), ('recent', slice(4, 6))]:
            dates = pd.DatetimeIndex([utc(WINDOWS[sl.start][0])]+[t for e in endpoints[sl] for t in e])
            np.savez_compressed(OUT/f'{tf}_{group}_equity.npz', timestamps=dates.as_unit('ns').asi8,
                **{name+'__'+sc: metrics(np.concatenate(parts[sl]))['equity'] for (name, sc), parts in sequences.items()})
    comparisons = [{'candidate': a, 'reference': b, **interval(all_daily[a], all_daily[b])} for a, b in contrasts()]
    for filename, value in [('actions', actions), ('trades', trades)]:
        write_json(OUT/(filename+'.json'), value)
    write_json(OUT/'report.json', {'strategies': rows, 'comparisons': comparisons,
        'models': [{k: m[k] for k in ['name', 'method', 'timeframe', 'available', 'cutoff', 'training_count', 'last_label_known', 'effective_weight_count']} for m in models.values()],
        'controller_history': {tf: json.loads((OUT/f'{tf}_controller.json').read_text())['state']['history'] for tf in FRAMES},
        'fits': len(models), 'asset_windows': 504, 'portfolio_windows': 252,
        'elapsed_seconds': time.monotonic()-started, 'production_selection': None})
    print('Study013 completed. Controller promotions are simulated only.', flush=True)


if __name__ == '__main__':
    run()
