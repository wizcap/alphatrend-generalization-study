"""Frozen supervised residual correction of AlphaTrend losing trades."""
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

from .market import simulate, summary
from .repair_v12 import (SYMBOLS, FRAMES, KINDS, MODES, CONTROLS, TRAIN_START, END,
    dataset, context, labelled_events, fit_model, policy, attribution)
from .research_v3 import FOLDS, snapshot
from .research_v6 import write_json
from .research_v11 import grouped, verify_raw_sources
from .timeframes import metrics, portfolio, daily_returns

OUT = Path('artifacts/experiment_012')
WINDOWS = FOLDS+[('2026-01-01', '2026-07-01'), ('2026-07-01', END)]
SCENARIOS = [('base', {'cost': 7., 'delay': 1}), ('double_cost', {'cost': 14., 'delay': 1}),
             ('delay30m', {'cost': 7., 'delay': 2})]
INTERVAL_COUNT, DRAWS = 64, 100000


def catalog(bars, frames, atr):
    labels, diagnostics = {tf: [] for tf in FRAMES}, []
    for tf in FRAMES:
        for s in SYMBOLS:
            ctx = context(s, bars[s], frames[tf][s], atr[tf][s], TRAIN_START, END)
            rows, ds = labelled_events(ctx, tf)
            labels[tf].extend(rows)
            diagnostics += [{'timeframe': tf, 'symbol': s, **v} for v in ds]
    return labels, diagnostics


def contrasts():
    pairs = []
    for tf in FRAMES:
        for kind in ['ridge', 'boost']:
            for mode in MODES:
                name = f'{tf}_{kind}_{mode}'
                pairs.extend([(name, tf+'_alpha'), (name, tf+'_mean_'+mode)])
            pairs.extend([(f'{tf}_{kind}_both', f'{tf}_{kind}_{mode}') for mode in ['entry', 'exit']])
    assert len(pairs)*2 == INTERVAL_COUNT
    return pairs


def paired_interval(parts, refs):
    diffs = [np.log1p(a)-np.log1p(b) for a, b in zip(parts, refs)]
    intervals = {}
    for block in [7, 28]:
        rng = np.random.default_rng(1012)
        draws = np.zeros(DRAWS)
        for d in diffs:
            n, number = len(d), int(np.ceil(len(d)/block))
            last = n-(number-1)*block
            cs = np.r_[0., np.cumsum(np.tile(d, int(np.ceil(block/n))+2))]
            for i in range(0, DRAWS, 2000):
                starts = rng.integers(0, n, size=(2000, number))
                draws[i:i+2000] += (cs[starts[:, :-1]+block]-cs[starts[:, :-1]]).sum(axis=1)
                draws[i:i+2000] += cs[starts[:, -1]+last]-cs[starts[:, -1]]
        intervals[str(block)] = np.quantile(draws, [.025/INTERVAL_COUNT, 1-.025/INTERVAL_COUNT]).tolist()
    return {'log_wealth_excess': float(sum(v.sum() for v in diffs)), 'intervals': intervals,
            'draws': DRAWS, 'corrected_intervals': INTERVAL_COUNT}


def candidate_models(models, tf, fold):
    result = {c: c for c in CONTROLS}
    for kind in KINDS:
        pair = {h: models[f'{tf}_fold{min(fold, 4)}_{kind}_{h}'] for h in ['entry', 'exit']}
        for mode in MODES:
            result[kind+'_'+mode] = {'mode': mode, **pair}
    return result


def evaluate(ctxs, tf, candidate, cost, delay):
    results, events = {}, []
    for s, ctx in ctxs.items():
        p, a, actions = policy(ctx, tf, candidate, delay)
        r = simulate(ctx['market'], p, cost, 0., trade_allowed=a, rebalance_band=.05)
        r['sharpe'] = metrics(r['returns'])['sharpe']
        results[s] = r
        events += [{'symbol': s, **v} for v in actions]
    return results, events


def repair_summary(records):
    loss = [r for r in records if r['baseline_loss']]
    win = [r for r in records if not r['baseline_loss']]
    lost = -sum(r['baseline_log'] for r in loss)
    won = sum(r['baseline_log'] for r in win)
    return {'episodes': len(records), 'baseline_losers': len(loss),
        'baseline_loss_log': lost, 'baseline_win_log': won,
        'loss_saved_log': sum(r['delta_log'] for r in loss),
        'winner_change_log': sum(r['delta_log'] for r in win),
        'total_delta_log': sum(r['delta_log'] for r in records),
        'loss_repair_fraction': sum(r['delta_log'] for r in loss)/lost if lost > 0 else None,
        'winner_retention': sum(r['candidate_log'] for r in win)/won if won > 0 else None,
        'losers_improved': sum(r['delta_log'] > 1e-12 for r in loss),
        'winners_harmed': sum(r['delta_log'] < -1e-12 for r in win)}


def run():
    if (OUT/'report.json').exists():
        raise ValueError('Completed study012 is immutable')
    OUT.mkdir(parents=True, exist_ok=True)
    verify_raw_sources()
    if (OUT/'protocol.json').exists():
        protocol = json.loads((OUT/'protocol.json').read_text())
        for p, h in {**protocol['source_hashes'], **protocol['evidence_hashes']}.items():
            assert sha256(Path(p).read_bytes()).hexdigest() == h, p
    else:
        hashes = snapshot(OUT)
        for p in [Path('reports/study012_method.md'), *Path('.').glob('requirements-study*.txt'),
                  Path('pytest.ini'), *Path('tests').glob('test_*.py')]:
            target = OUT/'source'/p
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
            hashes[p.as_posix()] = sha256(p.read_bytes()).hexdigest()
        evidence = [Path('artifacts/experiment_008/manifest.json'), Path('artifacts/experiment_008/protocol.json'),
                    Path('artifacts/experiment_011/report.json'), Path('artifacts/experiment_011/protocol.json'),
                    Path('artifacts/experiment_011/verification.json'), Path('artifacts/study012_preflight.json')]
        protocol = {'study': '012', 'created_utc': datetime.now(timezone.utc).isoformat(), 'source_hashes': hashes,
            'evidence_hashes': {p.as_posix(): sha256(p.read_bytes()).hexdigest() for p in evidence},
            'timeframes': FRAMES, 'kinds': KINDS, 'modes': MODES, 'scenarios': SCENARIOS,
            'windows': WINDOWS, 'fit_count': 60, 'contrasts': contrasts(), 'interval_count': INTERVAL_COUNT,
            'runtime': {'python': sys.version, 'packages': {d.metadata['Name']: d.version for d in distributions()}},
            'known_history_only': True, 'selected': None}
        write_json(OUT/'protocol.json', protocol)
    fingerprint = sha256((OUT/'protocol.json').read_bytes()).hexdigest()
    started = time.monotonic()
    bars, frames, atr = dataset()
    labels, diag = catalog(bars, frames, atr)
    text = json.dumps(labels, sort_keys=True)
    label_hash = sha256(text.encode()).hexdigest()
    preflight = json.loads(Path('artifacts/study012_preflight.json').read_text())
    assert label_hash == preflight['labels_sha256']
    for p, h in preflight['module_hashes'].items():
        assert sha256(Path(p).read_bytes()).hexdigest() == h
    labels_file = OUT/'labels.json'
    if labels_file.exists():
        assert labels_file.read_text() == text
    else:
        temp = labels_file.with_suffix('.tmp')
        temp.write_text(text)
        temp.replace(labels_file)
    write_json(OUT/'baseline_episodes.json', diag)
    models = {}
    for tf in FRAMES:
        for fold, (start, _) in enumerate(WINDOWS[:5]):
            cutoff = str((pd.Timestamp(start)-pd.Timedelta(days=7)).date())
            for kind in KINDS:
                for head in ['entry', 'exit']:
                    name = f'{tf}_fold{fold}_{kind}_{head}'
                    path = OUT/'models'/f'{name}.pkl'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.exists():
                        meta = json.loads(path.with_suffix('.json').read_text())
                        assert meta['protocol_sha256'] == fingerprint and meta['labels_sha256'] == label_hash
                        assert meta['model_sha256'] == sha256(path.read_bytes()).hexdigest()
                        fitted = pickle.loads(path.read_bytes())
                    else:
                        fitted = fit_model(labels[tf], head, kind, cutoff)
                        fitted.update({'name': name, 'timeframe': tf, 'fold': fold})
                        tmp = path.with_suffix('.tmp')
                        tmp.write_bytes(pickle.dumps(fitted))
                        meta_tmp = path.with_suffix('.json.tmp')
                        write_json(meta_tmp, {'protocol_sha256': fingerprint, 'labels_sha256': label_hash,
                            'model_sha256': sha256(tmp.read_bytes()).hexdigest()})
                        meta_tmp.replace(path.with_suffix('.json'))
                        tmp.replace(path)
                    assert (fitted['name'], fitted['kind'], fitted['head'], fitted['cutoff']) == (name, kind, head, str(pd.Timestamp(cutoff, tz='UTC')))
                    models[name] = fitted
            print(f'Fitted {tf} fold{fold}: entry={models[f"{tf}_fold{fold}_ridge_entry"]["training_count"]}, exit={models[f"{tf}_fold{fold}_ridge_exit"]["training_count"]}', flush=True)
    all_daily, rows, trades, actions = {}, [], [], []
    for tf in FRAMES:
        sequences, records, endpoints = {}, {}, []
        for fold, (start, end) in enumerate(WINDOWS):
            ctxs = {s: context(s, bars[s], frames[tf][s], atr[tf][s], start, end) for s in SYMBOLS}
            endpoints.append(ctxs[SYMBOLS[0]]['market'].times[1:])
            candidates = candidate_models(models, tf, fold)
            bases = {scenario: evaluate(ctxs, tf, 'alpha', **opts)[0] for scenario, opts in SCENARIOS}
            for arm, candidate in candidates.items():
                name = tf+'_'+arm
                for scenario, opts in SCENARIOS:
                    results, events = evaluate(ctxs, tf, candidate, **opts)
                    port = portfolio(results)
                    sequences.setdefault((name, scenario), []).append(port['returns'])
                    records.setdefault((name, scenario), []).append({'window': [start, end], 'observations': len(port['returns']),
                        'portfolio': summary(port), 'by_symbol': {s: summary(v) for s, v in results.items()}})
                    actions += [{'name': name, 'scenario': scenario, 'fold': fold, **v} for v in events]
                    for s, ctx in ctxs.items():
                        ts = attribution(ctx, tf, bases[scenario][s], results[s], opts['delay'])
                        trades += [{'name': name, 'scenario': scenario, 'fold': fold, 'symbol': s, **v} for v in ts]
            print(f'Evaluated {tf} window{fold}', flush=True)
        for arm in candidates:
            name = tf+'_'+arm
            row = {'name': name, 'timeframe': tf, 'arm': arm, 'folds': {sc: records[name, sc] for sc, _ in SCENARIOS}}
            for group, sl in [('development', slice(0, 4)), ('recent', slice(4, 6))]:
                dates = pd.DatetimeIndex(np.concatenate([v.to_numpy() for v in endpoints[sl]]))
                row[group] = {sc: grouped(sequences[name, sc][sl], records[name, sc][sl], dates) for sc, _ in SCENARIOS}
                row[group+'_repair'] = {}
                for sc, _ in SCENARIOS:
                    subset = [v for v in trades if v['name'] == name and v['scenario'] == sc and v['fold'] in range(sl.start, sl.stop)]
                    row[group+'_repair'][sc] = {'all': repair_summary(subset),
                        'natural_only': repair_summary([v for v in subset if not v['left_censored'] and not v['right_censored']])}
            rows.append(row)
            all_daily[name] = [daily_returns(r, t).to_numpy() for r, t in zip(sequences[name, 'base'][:4], endpoints[:4])]
        for group, sl in [('development', slice(0, 4)), ('recent', slice(4, 6))]:
            dates = pd.DatetimeIndex([pd.Timestamp(WINDOWS[sl.start][0], tz='UTC')]+[t for e in endpoints[sl] for t in e])
            curves = {name+'__'+sc: metrics(np.concatenate(parts[sl]))['equity'] for (name, sc), parts in sequences.items()}
            np.savez_compressed(OUT/f'{tf}_{group}_equity.npz', timestamps=dates.as_unit('ns').asi8, **curves)
    for parts in all_daily.values():
        assert [len(p) for p in parts] == [182, 184, 181, 184]
    comparisons = [{'candidate': a, 'reference': b, **paired_interval(all_daily[a], all_daily[b])} for a, b in contrasts()]
    by_name = {r['name']: r for r in rows}
    gates = []
    for tf in FRAMES:
        baseline = by_name[tf+'_alpha']
        for kind in ['ridge', 'boost']:
            for mode in MODES:
                name = f'{tf}_{kind}_{mode}'
                row = by_name[name]
                c = next(v for v in comparisons if v['candidate'] == name and v['reference'] == tf+'_alpha')
                rep = row['development_repair']['base']['all']
                checks = {'net_improvement': row['development']['base']['return'] > baseline['development']['base']['return'],
                    'drawdown_no_worse': row['development']['base']['max_drawdown'] <= baseline['development']['base']['max_drawdown'],
                    'double_cost_no_worse': row['development']['double_cost']['return'] >= baseline['development']['double_cost']['return'],
                    'recent_no_worse': row['recent']['base']['return'] >= baseline['recent']['base']['return'],
                    'repair_20pct': rep['loss_repair_fraction'] is not None and rep['loss_repair_fraction'] >= .20,
                    'retain_90pct': rep['winner_retention'] is not None and rep['winner_retention'] >= .90,
                    'both_intervals_positive': all(lo > 0 for lo, hi in c['intervals'].values())}
                gates.append({'name': name, 'checks': checks, 'all_pass': all(checks.values())})
    write_json(OUT/'trades.json', trades)
    write_json(OUT/'actions.json', actions)
    assert len(rows) == 28 and len(models) == 60
    write_json(OUT/'report.json', {'strategies': rows, 'comparisons': comparisons, 'gates': gates,
        'models': [{k: m[k] for k in ['name', 'kind', 'head', 'cutoff', 'training_count', 'negative_labels', 'last_label_known',
            'mean', 'training_prediction_std', 'training_negative_prediction_fraction', 'training_label_delay_hours', 'unresolved_or_not_mature_count']} for m in models.values()],
        'fit_count': len(models), 'portfolio_windows': len(rows)*18, 'asset_windows': len(rows)*36,
        'elapsed_seconds': time.monotonic()-started, 'labels_sha256': label_hash, 'selected': None})
    print('Study012 complete; no model selected for deployment.', flush=True)


if __name__ == '__main__':
    run()
