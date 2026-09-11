"""Frozen-rule, restartable shadow refresh. No exchange orders or scheduler."""
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import sys
import uuid

import numpy as np
import pandas as pd

from . import adaptive_v13 as old
from .alphatrend import alpha_indicator
from .fast_accounting import audit_simulate
from .indicators_v11 import build_frame, allowed_times, alpha_target
from .repair_v12 import context, signals, policy as alpha_policy, cut, labelled_events, predict
from .shadow_data_v14 import acquire, now, write_json, verify_receipts
from .sources_v8 import load
from .timeframes import aggregate

ROOT = Path('artifacts/shadow_014')
PRIOR = Path('artifacts/experiment_013')
START, ANCHOR = '2026-09-01', '2026-07-01'
ARMS = ['alpha', 'frozen_jan2026', 'monthly', 'weighted', 'guarded']
COSTS = {'base': 7., 'double_cost': 14.}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def hash_file(path):
    return sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def writer(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root/'writer.lock'
    fd = os.open(path, os.O_CREAT | os.O_RDWR)
    file = os.fdopen(fd, 'r+b', buffering=0)
    try:
        if path.stat().st_size == 0:
            file.write(b'0')
        file.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise FileExistsError('Another shadow writer holds the OS lock') from exc
        try:
            file.seek(1); file.truncate()
            file.write(json.dumps({'pid': os.getpid(), 'created': str(now())}).encode())
            yield
        finally:
            file.seek(0)
            if os.name == 'nt':
                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(file.fileno(), fcntl.LOCK_UN)
    finally:
        file.close()  # OS also releases the lock on process death; keep its inode.


def manifest(root):
    return {p.relative_to(root).as_posix(): hash_file(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p != root/'bundle.json'}


def verify_bundle(root, expected=None):
    root = Path(root)
    if expected is not None and hash_file(root/'bundle.json') != expected:
        raise ValueError('Bundle manifest fingerprint mismatch')
    m = read_json(root/'bundle.json')
    if manifest(root) != m['files']:
        raise ValueError('Corrupt, missing or unexpected bundle contents')
    return m


def commit(root, staging, name, parent):
    """Final directory first, current pointer last; no partial state is visible."""
    destination = root/'runs'/name
    if destination.exists():
        raise ValueError('Refuse to overwrite an immutable shadow run')
    write_json(staging/'bundle.json', {'parent': parent, 'files': manifest(staging)})
    staging.rename(destination)
    verify_bundle(destination)
    pointer = {'run': name, 'bundle_sha256': hash_file(destination/'bundle.json')}
    temporary = root/'current.tmp'
    write_json(temporary, pointer)
    temporary.replace(root/'current.json')
    return pointer


def recover_committed_bundle(root, pointer, protocol_hash):
    """Recover a durable final bundle even if the next acquisition is later."""
    while True:
        candidates = []
        for path in (root/'runs').glob('*'):
            if not path.is_dir() or not re.fullmatch(r'\d{8}T\d{6}',path.name):
                continue
            bundle = read_json(path/'bundle.json')
            if bundle['parent'] == pointer:
                candidates.append(path)
        if not candidates:
            return pointer
        if len(candidates) != 1:
            raise ValueError('Ambiguous orphan bundle lineage')
        path = candidates[0]
        verify_bundle(path)
        status = read_json(path/'status.json')
        if status['protocol_sha256'] != protocol_hash or status['run'] != path.name:
            raise ValueError('Orphan bundle has incompatible protocol or endpoint')
        pointer = {'run':path.name,'bundle_sha256':hash_file(path/'bundle.json')}
        write_json(root/'current.tmp',pointer)
        (root/'current.tmp').replace(root/'current.json')


def initialize(root=ROOT):
    root = Path(root)
    if (root/'protocol.json').exists():
        return verify_protocol(root)
    if (root/'bootstrap.pkl').exists():
        raise ValueError('Incomplete initialization requires inspection; no overwrite')
    prior = read_json(PRIOR/'protocol.json')
    for name, expected in {**prior['source_hashes'], **prior['evidence_hashes']}.items():
        if hash_file(name) != expected:
            raise ValueError('Study013 frozen dependency changed: '+name)
    bars = {s: load(s)[0] for s in old.SYMBOLS}
    (root/'bootstrap.pkl').write_bytes(pickle.dumps(bars))
    sources = [*Path('neattrade').glob('*.py'), *Path('tests').glob('test_*.py'),
               Path('reports/study014_method.md'), Path('requirements-study014.txt')]
    hashes = {}
    for source in sources:
        target = root/'source'/source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[source.as_posix()] = hash_file(source)
    evidence = [PRIOR/'protocol.json', PRIOR/'feedback.json', PRIOR/'2h_controller.json',
                PRIOR/'4h_controller.json', PRIOR/'verification.json', PRIOR/'delivery.json',
                Path('artifacts/experiment_012/labels.json'), *sorted((PRIOR/'models').glob('*'))]
    p = {'schema': 1, 'created_at': str(now()), 'source_hashes': hashes,
         'bootstrap_sha256': hash_file(root/'bootstrap.pkl'),
         'evidence_hashes': {str(f): hash_file(f) for f in evidence},
         'timeframes': old.FRAMES, 'symbols': old.SYMBOLS, 'arms': ARMS,
         'costs': COSTS, 'start': START, 'anchor': ANCHOR,
         'python': sys.version, 'packages': {d.metadata['Name']: d.version for d in distributions()},
         'production_selection': None, 'background_scheduler': False}
    write_json(root/'protocol.json', p)
    return p


def verify_protocol(root=ROOT):
    root = Path(root)
    p = read_json(root/'protocol.json')
    runtime = {d.metadata['Name']: d.version for d in distributions()}
    if p['python'] != sys.version or p['packages'] != runtime:
        raise ValueError('Pinned shadow runtime changed')
    for name, expected in {**p['source_hashes'], **p['evidence_hashes']}.items():
        if hash_file(name) != expected:
            raise ValueError('Frozen stream dependency changed: '+name)
    if hash_file(root/'bootstrap.pkl') != p['bootstrap_sha256']:
        raise ValueError('Frozen historical bootstrap changed')
    for name, expected in p['source_hashes'].items():
        if hash_file(root/'source'/name) != expected:
            raise ValueError('Frozen source snapshot changed')
    return p


def catalogs(bars, endpoint):
    labels, feedback, contexts = {}, {}, {}
    for tf in old.FRAMES:
        labels[tf], feedback[tf], contexts[tf] = [], [], {}
        for symbol in old.SYMBOLS:
            b = bars[symbol]
            f = build_frame(b, tf)
            atr = alpha_indicator(aggregate(b, tf)).atr_sma.shift(1).reindex(b.index, method='ffill')
            ctx = context(symbol, b, f, atr, old.TRAIN_START, str(old.utc(endpoint).tz_localize(None)))
            lab, _ = labelled_events(ctx, tf)
            labels[tf].extend(lab)
            m = ctx['market']
            episodes, events = signals(ctx, tf)
            checkpoints = {v['episode']: v for v in events if v['head'] == 'exit'}
            p, allowed, _ = alpha_policy(ctx, tf, 'alpha')
            for i, ep in enumerate(episodes):
                start = ep['entry']+1
                end = ep['exit']+1 if ep['exit'] is not None else len(m.x)
                event = checkpoints.get(i)
                r = {'id': f'{symbol}_{m.times[ep["entry"]].isoformat()}', 'symbol': symbol,
                     'entry': str(m.times[ep['entry']]), 'checkpoint': str(m.times[event['decision']]) if event else None,
                     'known': str(m.times[end]) if ep['exit'] is not None else None,
                     'left_censored': ep['left_censored'], 'x': event['x'].tolist() if event else None,
                     'base': {}, 'early': {}}
                # Unresolved episode payoffs remain provisional diagnostics, not labels.
                for scenario, cost in COSTS.items():
                    close = event['decision']+1 if event else end
                    r['base'][scenario] = float(audit_simulate(cut(m, start, end), p[start:end], allowed[start:end], cost)['return'])
                    r['early'][scenario] = float(audit_simulate(cut(m, start, close), p[start:close], allowed[start:close], cost)['return'])
                feedback[tf].append(r)
            contexts[tf][symbol] = context(symbol, b, f, atr, ANCHOR, str(old.utc(endpoint).tz_localize(None)))
    return labels, feedback, contexts


def assert_catalog_prefix(previous, current):
    """Completed observations are immutable; censoring may only resolve forward."""
    from .verify_v13 import equal
    for tf, rows in previous.items():
        index = {r['id']: r for r in current[tf]}
        if len(index) != len(current[tf]):
            raise ValueError('Duplicate catalog identity')
        for row in rows:
            if row['id'] not in index:
                raise ValueError('Prior event disappeared')
            following = index[row['id']]
            if row['known'] is not None:
                equal(row, following, 2e-10)
            else:
                for k in ['id', 'head', 'decision', 'entry', 'left_censored']:
                    if k in row and row[k] != following[k]:
                        raise ValueError('Unresolved event identity changed')
                if row.get('x') is not None:
                    np.testing.assert_allclose(row['x'], following['x'], atol=2e-10, rtol=0)


def selection(tf, arm, date, schedule):
    if arm == 'alpha':
        return None
    if arm == 'guarded':
        return schedule[str(old.utc(date).normalize().replace(day=1))]
    if arm == 'frozen_jan2026':
        return old.version_id(tf, 'expanding', '2026-01-01')
    if arm not in ['monthly', 'weighted']:
        raise ValueError('Unregistered shadow arm')
    return old.version_id(tf, 'weighted' if arm == 'weighted' else 'expanding', date)


def targets(ctx, tf, arm, models, schedule):
    m = ctx['market']
    allowed = allowed_times(m, tf)
    episodes, events = signals(ctx, tf)
    at = {e['decision']: e for e in events}
    values, actions, blocked, version = np.zeros(len(m.x)), [], False, None
    for t in np.flatnonzero(allowed):
        if t+1 >= len(m.x):
            raise ValueError('Decision lacks execution data')
        base = float(alpha_target(m.x[t:t+1])[0])
        if base == 0:
            blocked = False
        event = at.get(int(t))
        if event is not None:
            protected = episodes[event['episode']]['entry'] == 0
            forecast, reason = None, ''
            if event['head'] == 'entry':
                blocked = False
                version = None if protected else selection(tf, arm, m.times[t], schedule)
                if version is not None and old.utc(models[version]['available']) > m.times[t]:
                    raise ValueError('Position assigned a future model')
                reason = 'boundary_keep' if protected else 'latch'
            elif version is not None and not blocked:
                forecast = float(predict(models[version], event['x'])[0])
                if forecast < 0:
                    blocked, reason = True, 'early_exit'
            actions.append({'id': f'{tf}_{arm}_{m.symbol}_{m.times[t].isoformat()}_{event["head"]}',
                'timeframe': tf, 'arm': arm, 'symbol': m.symbol, 'episode': event['episode'],
                'head': event['head'], 'time': str(m.times[t]), 'execution': str(m.times[t+1]),
                'entry_time': str(m.times[episodes[event['episode']]['entry']]),
                'version': version, 'prediction': forecast, 'blocked': bool(blocked),
                'reason': reason, 'boundary_protected': protected})
        values[t] = 0. if blocked else base
    anchors = np.maximum.accumulate(np.where(allowed, np.arange(len(m.x)), 0))
    return np.r_[0., values[anchors][:-1]], np.r_[False, allowed[:-1]], actions


def account(m, positions, allowed, cost):
    """Continuous fixed-unit account; no artificial liquidation on refresh."""
    p, allowed = np.asarray(positions), np.asarray(allowed)
    if len(p) != len(m.x) or len(allowed) != len(p) or not np.isfinite(p).all() or (np.abs(p) > .5000001).any():
        raise ValueError('Invalid account targets')
    if not np.isfinite(cost) or cost < 0:
        raise ValueError('Invalid cost')
    n, rate = len(p), cost/10000
    equity, units, fees, funding = np.ones(n+1), np.zeros(n+1), np.zeros(n+1), np.zeros(n+1)
    orders = np.zeros(n+1, dtype=np.int64)
    e, q = 1., 0.
    for i in range(n):
        before = e
        wanted = p[i]*e/m.prices[i] if allowed[i] else q
        if q*wanted > 0 and abs(wanted-q)*m.prices[i]/e < .05:
            wanted = q
        traded = abs(wanted-q)*m.prices[i]
        fee = traded*rate
        e -= fee
        q = wanted
        adverse = m.lows[i] if q >= 0 else m.highs[i]
        if e+q*(adverse-m.prices[i]) <= .05*before:
            raise ValueError('Intrabar equity floor breached')
        payment = q*m.marks[i+1]*m.funding[i+1]
        e += q*(m.prices[i+1]-m.prices[i])
        e -= payment
        if not np.isfinite(e) or e <= 0:
            raise ValueError('Insolvent shadow account')
        equity[i+1], units[i+1] = e, q
        fees[i+1], funding[i+1] = fees[i]+fee, funding[i]+payment
        orders[i+1] = orders[i]+int(traded > 1e-10)
    return {'equity': equity, 'units': units, 'fees': fees, 'funding': funding, 'orders': orders,
            'liquidation_equity': float(e-abs(q)*m.prices[-1]*rate)}


def journal(actions, prior, created, recorded):
    lookup = {a['id']: a for a in prior}
    result = []
    for a in actions:
        if a['id'] in lookup:
            saved = lookup[a['id']]
            if any(saved[k] != v for k, v in a.items()):
                raise ValueError('Previously journaled decision changed')
            result.append(saved)
        else:
            fitted = created[a['version']] if a['version'] is not None else None
            entry_execution = old.utc(a['entry_time'])+pd.Timedelta(minutes=15)
            timely = old.utc(recorded) < old.utc(a['execution']) and (fitted is None or old.utc(fitted) < entry_execution)
            if a['head'] != 'entry':
                entry_id = f'{a["timeframe"]}_{a["arm"]}_{a["symbol"]}_{old.utc(a["entry_time"]).isoformat()}_entry'
                entry = next((r for r in result if r['id'] == entry_id), None)
                timely = timely and entry is not None and entry['evidence_class'] == 'timely_shadow'
            result.append({**a, 'recorded_at': str(recorded), 'model_created_at': str(fitted) if fitted is not None else None,
                           'evidence_class': 'timely_shadow' if timely else 'delayed_replay'})
    if set(lookup)-{a['id'] for a in result}:
        raise ValueError('Previously journaled decision disappeared')
    return result


def evaluate(bars, endpoint, models, controllers, created, prior_actions=()):
    labels, feedback, contexts = catalogs(bars, endpoint)
    updates = []
    for tf in old.FRAMES:
        c = controllers[tf]
        boundary = old.utc(c.state['last_boundary'])+pd.offsets.MonthBegin(1)
        while boundary <= old.utc(endpoint):
            for method in ['expanding', 'weighted']:
                name = old.version_id(tf, method, boundary)
                if name in models:
                    raise ValueError('Unexpected preexisting uncommitted model version')
                models[name] = old.train_version(labels[tf], tf, method, boundary)
                created[name] = str(now())
            update = c.step(boundary, feedback[tf], models)
            updates.append({'timeframe': tf, **update})
            boundary += pd.offsets.MonthBegin(1)
    arrays, metrics, actions = {}, [], []
    for tf in old.FRAMES:
        schedule = {h['boundary']: h['active'] for h in controllers[tf].state['history']}
        for symbol, ctx in contexts[tf].items():
            m = ctx['market']
            ix = m.times.get_loc(old.utc(START))
            for arm in ARMS:
                p, allowed, rows = targets(ctx, tf, arm, models, schedule)
                actions.extend(rows)
                for scenario, cost in COSTS.items():
                    result = account(m, p, allowed, cost)
                    key = f'{tf}__{arm}__{scenario}__{symbol}'
                    for field in ['equity', 'units', 'fees', 'funding', 'orders']:
                        arrays[key+'__'+field] = result[field]
                    arrays[key+'__targets'], arrays[key+'__allowed'] = p, allowed
                    e = result['equity']; initial = e[ix]
                    metrics.append({'timeframe': tf, 'symbol': symbol, 'arm': arm, 'scenario': scenario,
                        'start': START, 'end': str(endpoint), 'return': float(e[-1]/initial-1),
                        'max_drawdown': float(np.max(1-e[ix:]/np.maximum.accumulate(e[ix:]))),
                        'liquidation_return': result['liquidation_equity']/initial-1,
                        'fees': float((result['fees'][-1]-result['fees'][ix])/initial),
                        'funding': float((result['funding'][-1]-result['funding'][ix])/initial),
                        'orders': int(result['orders'][-1]-result['orders'][ix]),
                        'open_units': float(result['units'][-1])})
    actions = journal(actions, prior_actions, created, str(now()))
    return {'labels': labels, 'feedback': feedback, 'contexts': contexts, 'updates': updates,
            'arrays': arrays, 'metrics': metrics, 'actions': actions}


def refresh(root=ROOT, acquisition=acquire):
    root = Path(root)
    with writer(root):
        protocol = initialize(root)
        protocol_hash = hash_file(root/'protocol.json')
        pointer = read_json(root/'current.json') if (root/'current.json').exists() else None
        if pointer:
            if not re.fullmatch(r'\d{8}T\d{6}',pointer['run']):
                raise ValueError('Invalid current run identity')
            verify_bundle(root/'runs'/pointer['run'],pointer['bundle_sha256'])
        pointer = recover_committed_bundle(root,pointer,protocol_hash)
        previous_run = root/'runs'/pointer['run'] if pointer else None
        if previous_run:
            verify_bundle(previous_run, pointer['bundle_sha256'])
        models = {p.stem: pickle.loads(p.read_bytes()) for p in (PRIOR/'models').glob('*.pkl')}
        created = {name: read_json(PRIOR/'delivery.json')['created_utc'] for name in models}
        controllers = {}
        prior_hash = hash_file(PRIOR/'protocol.json')
        for tf in old.FRAMES:
            path = (previous_run if previous_run else PRIOR)/f'{tf}_controller.json'
            controllers[tf] = old.Controller.load(path, tf, prior_hash)
        if previous_run:
            models.update(pickle.loads((previous_run/'new_models.pkl').read_bytes()))
            created.update(read_json(previous_run/'model_created.json'))
        bars = pickle.loads(((previous_run/'bars.pkl') if previous_run else root/'bootstrap.pkl').read_bytes())
        stamp = now().strftime('%Y%m%dT%H%M%S%f')
        receipt_root = root/'receipts'/stamp
        bars, receipt = acquisition(bars, receipt_root)
        endpoint = old.utc(receipt['usable_endpoint'])
        if endpoint <= old.utc(START):
            raise ValueError('No complete extension accounting window')
        name = endpoint.strftime('%Y%m%dT%H%M%S')
        if pointer and name == pointer['run']:
            return {'status': 'unchanged', 'run': name, 'receipt': str(receipt_root)}
        if pointer and name < pointer['run']:
            raise ValueError('Data endpoint regressed')
        destination = root/'runs'/name
        if destination.exists():
            bundle = verify_bundle(destination)
            if bundle['parent'] != pointer or read_json(destination/'status.json')['protocol_sha256'] != protocol_hash:
                raise ValueError('Orphan bundle has incompatible lineage')
            saved = pickle.loads((destination/'bars.pkl').read_bytes())
            for s in old.SYMBOLS:
                np.testing.assert_allclose(saved[s], bars[s].reindex(saved[s].index), atol=1e-12, rtol=2e-14)
            recovered = {'run': name, 'bundle_sha256': hash_file(destination/'bundle.json')}
            write_json(root/'current.tmp', recovered)
            (root/'current.tmp').replace(root/'current.json')
            return {'status': 'recovered', **recovered}
        previous_actions = read_json(previous_run/'actions.json') if previous_run else []
        result = evaluate(bars, endpoint, models, controllers, created, previous_actions)
        prior_labels = read_json(previous_run/'labels.json') if previous_run else read_json('artifacts/experiment_012/labels.json')
        prior_feedback = read_json(previous_run/'feedback.json') if previous_run else read_json(PRIOR/'feedback.json')
        assert_catalog_prefix(prior_labels, result['labels'])
        assert_catalog_prefix(prior_feedback, result['feedback'])
        if previous_run:
            with np.load(previous_run/'accounts.npz') as previous:
                for key in previous.files:
                    np.testing.assert_array_equal(previous[key], result['arrays'][key][:len(previous[key])])
        staging = root/'runs'/('pending_'+uuid.uuid4().hex)
        staging.mkdir(parents=True)
        (staging/'bars.pkl').write_bytes(pickle.dumps(bars))
        (staging/'new_models.pkl').write_bytes(pickle.dumps({k: v for k, v in models.items() if old.utc(v['available']) >= old.utc(START)}))
        write_json(staging/'model_created.json', created)
        for tf, controller in controllers.items():
            controller.save(staging/f'{tf}_controller.json')
        for key in ['labels', 'feedback', 'updates', 'metrics', 'actions']:
            write_json(staging/(key+'.json'), result[key])
        np.savez_compressed(staging/'accounts.npz', **result['arrays'])
        status = {'status': 'committed', 'run': name, 'endpoint': str(endpoint), 'created_at': str(now()),
            'protocol_sha256': protocol_hash, 'receipt': str(receipt_root),
            'receipt_manifest_sha256': hash_file(receipt_root/'manifest.json'),
            'new_model_count': sum(old.utc(m['available']) >= old.utc(START) for m in models.values()),
            'update_count_this_refresh': len(result['updates']), 'asset_accounts': len(result['metrics']),
            'timely_shadow_decisions': sum(a['evidence_class'] == 'timely_shadow' for a in result['actions']),
            'extension_decisions': sum(old.utc(a['time']) >= old.utc(START) for a in result['actions']),
            'production_selection': None, 'background_scheduler': False}
        write_json(staging/'status.json', status)
        commit(root, staging, name, pointer)
        return status


if __name__ == '__main__':
    if '--init' in sys.argv:
        with writer(ROOT):
            initialize(ROOT)
        print('Frozen shadow protocol initialized; no September market evaluation yet')
    else:
        print(json.dumps(refresh(), indent=2))
