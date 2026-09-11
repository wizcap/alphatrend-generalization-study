"""Independent model, controller, target, account and split-refresh replay."""
from copy import deepcopy
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd

from . import shadow_v14 as s
from .adaptive_v13 import utc, model_digest, train_version, Controller
from .indicators_v11 import build_frame
from .alphatrend import alpha_indicator
from .timeframes import aggregate, TIMEFRAMES
from .repair_v12 import context
from .market import simulate
from .verify_v12 import reference_state, reference_predict
from .verify_v13 import equal, reference_feedback, reference_score
from .shadow_data_v14 import verify_receipts


def controller_step(state, tf, date, rows, models):
    """Separate scalar transition starting from the independently saved prior state."""
    a, activated, q, queued, incumbent = [state[k] for k in ['active','activated','challenger','queued','incumbent']]
    t = utc(date)
    eligible = lambda start: sorted([r for r in rows if not r['left_censored'] and r['known'] is not None
        and utc(r['known']) < t-pd.Timedelta(days=7)
        and utc(r['entry']) >= max(utc(start),t-pd.Timedelta(days=180))], key=lambda r:r['id'])
    v = {'boundary':str(t),'before':a,'actions':[],'health':None,'validation':None,
         'health_status':'inactive','validation_status':'newly_queued'}
    if a:
        health = reference_score(eligible(activated), models[a], None)
        enough = health['checkpoints'] >= 12 and all(n >= 3 for n in health['by_symbol'].values())
        v.update(health=health,health_status='healthy' if enough else 'insufficient_evidence')
        if enough and any(x['alpha_excess'] < 0 for x in health['economics'].values()):
            a = activated = q = queued = incumbent = None
            v['actions'].append('rollback')
            v.update(health_status='rollback',validation_status='cancelled_by_rollback')
    if q:
        value = reference_score(eligible(queued), models[q], models.get(incumbent))
        b,d = value['economics']['base'],value['economics']['double_cost']
        checks = {'sample':value['checkpoints'] >= 20 and all(n >= 4 for n in value['by_symbol'].values()),
            'outcomes':b['losers'] >= 3 and b['winners'] >= 3 and value['early_exits'] >= 2,
            'alpha_base':b['alpha_excess'] > 0,'alpha_double':d['alpha_excess'] > 0,
            'incumbent_base':b['incumbent_excess'] > 0,'incumbent_double':d['incumbent_excess'] >= 0,
            'repair':b['repair'] is not None and b['repair'] >= .2,
            'retention':b['retention'] is not None and b['retention'] >= .9}
        v['validation'] = {'version':q,'queued':queued,'incumbent':incumbent,**value,'checks':checks}
        v['validation_status'] = 'observing'
        if checks['sample'] and checks['outcomes']:
            if all(checks.values()):
                a, activated = q, str(t)
                v['actions'].append('promote'); v['validation_status'] = 'promoted'
            else:
                v['actions'].append('reject'); v['validation_status'] = 'rejected'
            q = queued = incumbent = None
        elif (t-utc(queued)).days >= 180:
            v['actions'].append('expire'); v['validation_status'] = 'expired'
            q = queued = incumbent = None
    if q is None:
        q, queued, incumbent = f'{tf}_weighted_{t:%Y%m}', str(t), a
        v['actions'].append('queue')
    v.update(active=a,activated=activated,challenger=q,queued=queued)
    state.update(active=a,activated=activated,challenger=q,queued=queued,incumbent=incumbent)
    return v


def reference_targets(ctx, tf, arm, models, schedule):
    m = ctx['market']; n = len(m.x); stride = TIMEFRAMES[tf]//15
    p, allowed, values = np.zeros(n), np.zeros(n,bool), np.zeros(n)
    before, entry, blocked, version = False, None, False, None
    forecasts = {}
    for i in range(0,n,stride):
        positive = m.x[i,0] > 0
        if positive and not before:
            entry, blocked = i, False
            t = m.times[i]
            if i == 0 or arm == 'alpha': version = None
            elif arm == 'frozen_jan2026': version = tf+'_expanding_202601'
            elif arm == 'guarded': version = schedule[str(t.normalize().replace(day=1))]
            else: version = f'{tf}_{"weighted" if arm == "weighted" else "expanding"}_{t:%Y%m}'
        elif positive and i-entry == 96 and version is not None:
            x = reference_state(ctx,entry,i,1)
            forecast = float(reference_predict(models[version],x)[0])
            forecasts[str(m.times[i])] = forecast
            blocked = forecast < 0
        elif not positive:
            blocked = False
        values[i:min(i+stride,n)] = min(.5,.2/max(m.x[i,-1],.1)) if positive and not blocked else 0.
        before = positive
        allowed[i+1] = True
    p[1:] = values[:-1]
    return p,allowed,forecasts


def run(root=s.ROOT):
    root = Path(root)
    s.verify_protocol(root)
    pointer = s.read_json(root/'current.json')
    path = root/'runs'/pointer['run']
    s.verify_bundle(path,pointer['bundle_sha256'])
    status = s.read_json(path/'status.json')
    assert s.hash_file(Path(status['receipt'])/'manifest.json') == status['receipt_manifest_sha256']
    verify_receipts(status['receipt'])
    bars = pickle.loads((path/'bars.pkl').read_bytes())
    labels, feedback, contexts = s.catalogs(bars,status['endpoint'])
    equal(labels,s.read_json(path/'labels.json')); equal(feedback,s.read_json(path/'feedback.json'))
    old_models = {p.stem:pickle.loads(p.read_bytes()) for p in (s.PRIOR/'models').glob('*.pkl')}
    new_models = pickle.loads((path/'new_models.pkl').read_bytes())
    models = {**old_models,**new_models}
    for name,m in new_models.items():
        rebuilt = train_version(labels[m['timeframe']],m['timeframe'],m['method'],m['available'])
        assert model_digest(rebuilt) == model_digest(m), name
        x = [r['x'] for r in labels[m['timeframe']] if r['id'] in set(m['training_ids'])]
        np.testing.assert_allclose(reference_predict(m,x),m['training_predictions'],atol=2e-10,rtol=0)
        assert utc(m['last_label_known']) < utc(m['available'])-pd.Timedelta(days=7)
    s.assert_catalog_prefix(s.read_json('artifacts/experiment_012/labels.json'),labels)
    s.assert_catalog_prefix(s.read_json(s.PRIOR/'feedback.json'),feedback)
    reference_rows, feedback_count = {}, 0
    for tf in s.old.FRAMES:
        reference_rows[tf] = []
        for symbol,b in bars.items():
            frame = build_frame(b,tf)
            atr = alpha_indicator(aggregate(b,tf)).atr_sma.shift(1).reindex(b.index,method='ffill')
            ctx = context(symbol,b,frame,atr,s.old.TRAIN_START,str(utc(status['endpoint']).tz_localize(None)))
            reference_rows[tf].extend(reference_feedback(ctx,tf))
        equal(reference_rows[tf],feedback[tf],2e-9)
        feedback_count += len(feedback[tf])
    controllers, update_count, asset_count, predictions = {},0,0,0
    for tf in s.old.FRAMES:
        state = deepcopy(s.read_json(s.PRIOR/f'{tf}_controller.json')['state'])
        controller = Controller.load(path/f'{tf}_controller.json',tf,s.hash_file(s.PRIOR/'protocol.json'))
        controllers[tf] = controller
        for update in controller.state['history'][32:]:
            expected = controller_step(state,tf,update['boundary'],reference_rows[tf],models)
            equal(expected,update,2e-9)
            update_count += 1
    with np.load(path/'accounts.npz') as arrays:
        for tf in s.old.FRAMES:
            schedule = {h['boundary']:h['active'] for h in controllers[tf].state['history']}
            for symbol,ctx in contexts[tf].items():
                m = ctx['market']
                for arm in s.ARMS:
                    p,allowed,forecasts = reference_targets(ctx,tf,arm,models,schedule)
                    for row in s.read_json(path/'actions.json'):
                        if (row['timeframe'],row['arm'],row['symbol']) == (tf,arm,symbol) and row['prediction'] is not None:
                            np.testing.assert_allclose(row['prediction'],forecasts[row['time']],atol=2e-10,rtol=0)
                            predictions += 1
                    for scenario,cost in s.COSTS.items():
                        key = f'{tf}__{arm}__{scenario}__{symbol}'
                        np.testing.assert_allclose(p,arrays[key+'__targets'],atol=1e-15,rtol=0)
                        np.testing.assert_array_equal(allowed,arrays[key+'__allowed'])
                        result = simulate(m,p,cost,0.,trade_allowed=allowed,rebalance_band=.05,trace=True)
                        e = arrays[key+'__equity']; units = arrays[key+'__units']
                        np.testing.assert_allclose(e[:-1],result['equity'][:-1],atol=2e-10,rtol=0)
                        liquidation = e[-1]-abs(units[-1])*m.prices[-1]*cost/10000
                        np.testing.assert_allclose(liquidation,result['equity'][-1],atol=2e-10,rtol=0)
                        np.testing.assert_allclose(arrays[key+'__funding'][-1],result['funding_paid_initial_equity'],atol=2e-10,rtol=0)
                        np.testing.assert_allclose(arrays[key+'__fees'][-1]+result['ledger'][-1]['terminal_exit_cost'],result['cost_paid_initial_equity'],atol=2e-10,rtol=0)
                        np.testing.assert_allclose(units[1:],[r['units'] for r in result['ledger']],atol=2e-10,rtol=0)
                        asset_count += 1
    # A split inside the acquired extension is fixed by elapsed time, not outcomes.
    end = utc(status['endpoint'])
    split = (utc(s.START)+(end-utc(s.START))/2).floor('4h')
    prior_hash = s.hash_file(s.PRIOR/'protocol.json')
    initial_controllers = {tf:Controller.load(s.PRIOR/f'{tf}_controller.json',tf,prior_hash) for tf in s.old.FRAMES}
    fresh = deepcopy(old_models)
    created = {name:s.read_json(s.PRIOR/'delivery.json')['created_utc'] for name in fresh}
    truncated = {symbol:b.loc[:split] for symbol,b in bars.items()}
    first = s.evaluate(truncated,split,fresh,initial_controllers,created)
    second = s.evaluate(bars,end,fresh,initial_controllers,created,first['actions'])
    for name in new_models:
        assert model_digest(fresh[name]) == model_digest(new_models[name])
    with np.load(path/'accounts.npz') as saved:
        for key,value in second['arrays'].items():
            np.testing.assert_array_equal(value,saved[key])
            np.testing.assert_array_equal(first['arrays'][key],value[:len(first['arrays'][key])])
    for tf,c in initial_controllers.items(): equal(c.state,controllers[tf].state)
    equal(second['metrics'],s.read_json(path/'metrics.json'))
    # Repeat endpoint through the full persistence path with the original observed
    # acquisition; no network, fit or pointer mutation is needed.
    original_pointer = (root/'current.json').read_bytes()
    def same_acquisition(previous,receipt_root):
        for symbol in bars: pd.testing.assert_frame_equal(previous[symbol],bars[symbol])
        return previous,{'usable_endpoint':status['endpoint']}
    unchanged = s.refresh(root,acquisition=same_acquisition)
    assert unchanged['status'] == 'unchanged' and (root/'current.json').read_bytes() == original_pointer
    tests = subprocess.run([sys.executable,'-m','pytest','-q'],capture_output=True,text=True)
    Path('reports/study014_tests.txt').write_text(tests.stdout+tests.stderr,encoding='utf-8')
    if tests.returncode: raise AssertionError(tests.stdout+tests.stderr)
    import re
    count = int(re.search(r'(\d+) passed',tests.stdout).group(1))
    result = {'run':pointer['run'],'tests_passed':count,'new_models_refitted':len(new_models),
        'independent_controller_updates':update_count,'independent_feedback_episodes':feedback_count,
        'independent_asset_accounts':asset_count,'manual_predictions':predictions,
        'old_completed_labels_and_feedback_preserved':True,'batch_split_prefix_identical':True,
        'same_endpoint_idempotent':True,'source_and_receipt_hashes_verified':True,
        'production_selection':None}
    s.write_json(root/'verification.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    run()
