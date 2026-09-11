"""Independent book-window integration, neural replay and financial reconciliation."""
from hashlib import sha256
import json
from pathlib import Path
import pickle
import re
import subprocess
import sys

import neat
import numpy as np
import pandas as pd

from .batch import fitness_accounting
from .market import summary
from .orderbook import ROOT, SYMBOLS
from .policies import mlp, CONNECTIONS
from .overlay import incremental_objective
from .research import batch_activate
from .research_v3 import SEEDS, FOLDS
from .research_v5 import model_config
from .research_v6 import write_json
from .research_v9 import (OUT, WINDOWS, SPECS, SCENARIOS, POPULATION, dataset, market, prepare,
    train_markets, base_targets, targets, network_inputs, desired, evaluate, grouped)
from .timeframes import PER_YEAR, portfolio, metrics


def direct_window(available, values, end):
    end=int(end)
    stop = np.searchsorted(available, end, side='left')
    if stop == 0:
        return 0., False, 0.
    first = max(0, np.searchsorted(available, end-14400, side='right')-1)
    ix = np.arange(first,stop)
    next_time = np.r_[available[first+1:stop], end]
    duration = np.maximum(0, np.minimum(np.minimum(next_time, available[ix]+120), end)
                          - np.maximum(available[ix],end-14400))
    duration = np.where(np.isfinite(values[ix]),duration,0)
    total = duration.sum()
    valid = total/14400 >= .95-1e-12 and np.isfinite(values[stop-1]) and end-available[stop-1] <= 120
    value = float(np.dot(duration,np.nan_to_num(values[ix]))/total) if total and valid else 0.
    return value, bool(valid), float(total/14400)


def run():
    protocol=json.loads((OUT/'protocol.json').read_text())
    report=json.loads((OUT/'report.json').read_text())
    for p,h in protocol['source_hashes'].items():
        assert sha256(Path(p).read_bytes()).hexdigest()==h,p
        assert sha256((OUT/'source'/p).read_bytes()).hexdigest()==h,p
    assert sha256((OUT/'manifest.json').read_bytes()).hexdigest()==protocol['manifest_sha256']
    manifests=json.loads((OUT/'manifest.json').read_text())
    raw_count=0
    for item in manifests['market']+manifests['book']:
        if item.get('status')=='missing_404':
            continue
        assert sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
        raw_count+=1
    bars,frames,late,meta=dataset()
    feature_checks=0
    for s in SYMBOLS:
        path=ROOT/f'{s}_snapshots.npz'
        assert sha256(path.read_bytes()).hexdigest()==meta['symbols'][s]['snapshot_sha256']
        with np.load(path) as data:
            t,v=data['times'],data['values']
        available=t+60
        f=pd.read_csv(ROOT/f'{s}_features.csv',index_col=0,parse_dates=True)
        f.index=pd.to_datetime(f.index,utc=True)
        selected=f.loc[(f.index.hour%4==0)&(f.index.minute==0)]
        for stamp,row in selected.iterrows():
            value,valid,coverage=direct_window(available,v,stamp.value/1e9)
            np.testing.assert_allclose([value,coverage],[row.book_imbalance,row.book_coverage],atol=2e-10,rtol=0)
            assert valid==row.book_valid
            feature_checks+=1
        print(f'Independent time-weighted book windows verified: {s}',flush=True)
    models={}
    neural_rows=0
    for fold,start in enumerate([a for a,_ in FOLDS]+['2026-01-01']):
        training=train_markets(bars,frames,start)
        prepared=[prepare(m) for m in training]
        benchmarks=[fitness_accounting(d,base_targets(d.x)[None,:],rebalance_band=.05) for d in prepared]
        for spec in SPECS:
            for seed in SEEDS:
                name=f'fold{fold}_{spec}_{seed}'
                path=OUT/'jobs'/f'{name}.pkl'
                info=json.loads(path.with_suffix('.json').read_text())
                assert sha256(path.read_bytes()).hexdigest()==info['model_sha256']
                assert info['protocol_sha256']==sha256((OUT/'protocol.json').read_bytes()).hexdigest()
                with path.open('rb') as f: member=pickle.load(f)
                assert member['spec']==spec and member['seed']==seed and member['name']==name
                assert pd.Timestamp(member['train_end'])==pd.Timestamp(start,tz='UTC')-pd.Timedelta(days=7)
                models[name]=member
                is_cma=member.get('kind')=='cma'
                if is_cma:
                    w=np.asarray(member['weights'])
                    assert w.shape==(29,) and np.max(np.abs(w))<=5+1e-10
                else:
                    config=model_config(POPULATION,5)
                    net=neat.nn.FeedForwardNetwork.create(member['genome'],config)
                history=pd.read_csv(path.with_suffix('.csv'))
                assert len(history)==protocol['generations']
                if is_cma: assert history.candidate_evaluations.sum()==POPULATION*protocol['generations']
                results=[]
                for d in prepared:
                    x=network_inputs(d.x,spec)
                    if is_cma:
                        ref=np.array([np.tanh(sum(np.tanh(sum(row[i]*w[i*4+j] for i in range(5))+w[20+j])*w[24+j]
                                                for j in range(4))+w[28]) for row in x])
                        computed=mlp(w,x)[0]
                    else:
                        ref=np.array([net.activate(row)[0] for row in x])
                        computed=batch_activate(net,x)
                    np.testing.assert_allclose(ref,computed,atol=2e-10,rtol=0)
                    neural_rows+=len(x)
                    r=fitness_accounting(d,targets(ref,d.x)[None,:],rebalance_band=.05)
                    r['years']=(d.market.times[-1]-d.market.times[0]).total_seconds()/(365.25*86400)
                    assert r['years']==len(d.market.x)/PER_YEAR
                    results.append(r)
                penalty=CONNECTIONS if is_cma else len(member['genome'].connections)
                fitness=incremental_objective(results,benchmarks)[0]-.0001*penalty
                np.testing.assert_allclose(fitness,member['fitness'] if is_cma else member['genome'].fitness,atol=2e-10,rtol=0)
        print(f'Replayed selected neural models and training fitness: fold{fold}',flush=True)
    rows={x['name']:x for x in report['strategies']}
    sequences,records,endpoints={}, {},[]
    asset_checks=0
    for fold,(start,end) in enumerate(WINDOWS):
        markets={s:market(s,bars[s],frames[s],start,end) for s in SYMBOLS}
        lagged={s:market(s,bars[s],late[s],start,end) for s in SYMBOLS}
        endpoints.append(markets[SYMBOLS[0]].times[1:])
        candidates={n:n for n in ['alpha','alpha_half','passive_vol','cash']}
        for spec in SPECS:
            members=[models[f'fold{min(fold,4)}_{spec}_{seed}'] for seed in SEEDS]
            candidates[spec]=members
            candidates.update({f'{spec}_seed{seed}':m for seed,m in zip(SEEDS,members)})
        for name,candidate in candidates.items():
            for scenario,options in SCENARIOS:
                opts=dict(options)
                use=lagged if opts.pop('lag_book',False) else markets
                r=evaluate(candidate,use,**opts)
                saved=rows[name]['folds'][scenario][fold]
                for s,m in use.items():
                    p,allowed=desired(candidate,m,opts.get('zero_book',False))
                    delay=opts.get('delay',0)
                    if delay:
                        p=np.r_[np.zeros(delay),p[:-delay]]
                        allowed=np.r_[np.zeros(delay,bool),allowed[:-delay]]
                    d=prepare(m,allowed)
                    fast=fitness_accounting(d,p[d.anchors][None,:],cost_bps=7*opts.get('multiplier',1.),rebalance_band=.05)
                    for field in ['return','max_drawdown','cost_paid_initial_equity','funding_paid_initial_equity']:
                        np.testing.assert_allclose(fast[field][0],r[s][field],atol=2e-10,rtol=0)
                        np.testing.assert_allclose(r[s][field],saved['by_symbol'][s][field],atol=2e-10,rtol=0)
                    asset_checks+=1
                port=portfolio(r)
                sequences.setdefault((name,scenario),[]).append(port['returns'])
                records.setdefault((name,scenario),[]).append({'by_symbol':r,'observations':len(port['returns'])})
                for k in ['return','max_drawdown','sharpe']:
                    np.testing.assert_allclose(port[k],saved['portfolio'][k],atol=2e-10,rtol=0)
        print(f'Independent accounting replays: {asset_checks}',flush=True)
    for group,sl in [('development',slice(0,4)),('recent',slice(4,6))]:
        dates=pd.DatetimeIndex(np.concatenate([v.to_numpy() for v in endpoints[sl]]))
        curves=pd.read_csv(OUT/f'{group}_equity.csv',index_col=0)
        for (name,scenario),parts in sequences.items():
            replay=grouped(parts[sl],records[name,scenario][sl],dates)
            saved=rows[name][group][scenario]
            for k in ['return','max_drawdown','sharpe','daily_sharpe']:
                np.testing.assert_allclose(replay[k],saved[k],atol=2e-10,rtol=0)
            for y,value in replay['annual_returns'].items():
                np.testing.assert_allclose(value,saved['annual_returns'][y],atol=2e-10,rtol=0)
            np.testing.assert_allclose(curves[name+'__'+scenario],metrics(np.concatenate(parts[sl]))['equity'],atol=2e-10,rtol=0)
    previous=json.loads(Path('artifacts/experiment_008/report.json').read_text())
    alpha=next(r for r in previous['strategies'] if r['name']=='4h_long')
    for group in ['development','recent']:
        for scenario in ['base','double_cost','delay15m']:
            for k in ['return','max_drawdown']:
                np.testing.assert_allclose(rows['alpha'][group][scenario][k],alpha[group][scenario][k],atol=2e-10,rtol=0)
    tests=subprocess.run([sys.executable,'-m','pytest','-q'],capture_output=True,text=True,check=True)
    (OUT/'pytest-output.txt').write_text(tests.stdout,encoding='utf-8')
    result={'tests_passed':int(re.search(r'(\d+) passed',tests.stdout)[1]),'raw_archives_verified':raw_count,
        'independent_book_window_checks':feature_checks,'selected_models_replayed':len(models),
        'reference_neural_activation_rows':neural_rows,'training_fitness_replayed':True,
        'independent_asset_accounting_checks':asset_checks,'full_curves_and_annual_returns_verified':True,
        'study008_alpha_control_reproduced':True,'frozen_sources_verified':True}
    write_json(OUT/'verification.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    run()
