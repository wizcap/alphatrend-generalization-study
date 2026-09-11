"""Study004: incremental NEAT trend gating with controlled turnover."""
import argparse
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

from .batch import fitness_accounting,prepare
from .data import download,load
from .market import make_market,simulate,summary
from .overlay import baseline_targets,incremental_objective,overlay_targets
from .research import assert_aligned,batch_activate,bootstrap_ci,config,portfolio
from .research_v3 import FOLDS,SEEDS,Trace,daily_schedule,development_gate,snapshot,stitched_metrics,training_markets

MODES=['plain','hysteretic']
BAND=.05


def desired(candidate,m,c):
    d=prepare(m)
    kind=candidate['kind']
    if kind=='baseline':
        daily=baseline_targets(d.x)*candidate.get('scale',1.)
    elif kind=='gate':
        raw=batch_activate(neat.nn.FeedForwardNetwork.create(candidate['genome'],c),d.x)
        daily=overlay_targets(raw[None,:],d.x,candidate['mode'])[0]
    elif kind=='ensemble':
        return np.mean([desired(member,m,c) for member in candidate['members']],axis=0)
    elif kind=='cash':
        daily=np.zeros(len(d.x))
    else:
        raise ValueError(kind)
    # Forward-fill past decisions for logging only; execution is restricted to anchors.
    anchors=np.maximum.accumulate(np.where(daily_schedule(m),np.arange(len(m.x)),0))
    full=np.zeros(len(m.x));full[d.anchors]=daily
    return full[anchors]


def evaluate(candidate,markets,c,multiplier=1.,delay=0,band=None):
    assert_aligned(markets)
    band=candidate.get('band',BAND) if band is None else band
    result={}
    for m in markets:
        p,allowed=desired(candidate,m,c),daily_schedule(m)
        if delay:
            p=np.r_[np.zeros(delay),p[:-delay]]
            allowed=np.r_[np.zeros(delay,dtype=bool),allowed[:-delay]]
        r=simulate(m,p,5*multiplier,2*multiplier,trade_allowed=allowed,rebalance_band=band)
        r['decision_flat_fraction']=float(np.mean(p[allowed]==0))
        r['decision_long_fraction']=float(np.mean(p[allowed]>0))
        baseline=baseline_targets(prepare(m).x)
        r['baseline_long_episodes']=int(np.sum((baseline>0)&~np.r_[False,baseline[:-1]>0]))
        result[m.symbol]=r
    return result


def train_one(job):
    markets,population,generations,seed,mode,name,output=job
    path=Path(output)/'jobs'/f'{name}.pkl'
    prepared=[prepare(m) for m in markets]
    benchmarks=[fitness_accounting(d,baseline_targets(d.x)[None,:],rebalance_band=BAND) for d in prepared]
    c=config(population)
    p=neat.Population(c,seed=seed)
    trace=Trace(name);p.add_reporter(trace)

    def fitness(genomes,conf):
        nets=[neat.nn.FeedForwardNetwork.create(g,conf) for _,g in genomes]
        results=[];cash=np.ones(len(genomes),bool)
        for d in prepared:
            raw=np.stack([batch_activate(net,d.x) for net in nets])
            targets=overlay_targets(raw,d.x,mode)
            cash&=np.all(targets==0,axis=1)
            results.append(fitness_accounting(d,targets,rebalance_band=BAND))
        values=incremental_objective(results,benchmarks)
        for i,(_,g) in enumerate(genomes):
            g.fitness=float(values[i]-.0001*len(g.connections));g.research_cash=bool(cash[i])
    winner=p.run(fitness,generations)
    candidate={'name':name,'kind':'gate','mode':mode,'genome':winner,'seed':seed,
               'train_cash':winner.research_cash,'train_end':str(max(m.times[-1] for m in markets))}
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:pickle.dump(candidate,f)
    tmp.replace(path)
    pd.DataFrame(trace.rows).to_csv(path.with_suffix('.csv'),index=False)
    return candidate


def controls():
    return [{'name':'cash','kind':'cash'},
            {'name':'baseline_unbanded','kind':'baseline','band':0.},
            {'name':'baseline_banded','kind':'baseline','band':BAND},
            {'name':'baseline_half_banded','kind':'baseline','scale':.5,'band':BAND}]


def relative_gates(row,baseline):
    paired=[a['portfolio']['return']-b['portfolio']['return'] for a,b in zip(row['folds'],baseline['folds'])]
    return {'beats_baseline_total':row['scenarios']['base']['return']>baseline['scenarios']['base']['return'],
            'beats_baseline_median_paired_fold':float(np.median(paired))>0,
            'beats_baseline_three_folds':sum(v>0 for v in paired)>=3,
            'beats_baseline_double_cost':row['scenarios']['double_cost']['return']>baseline['scenarios']['double_cost']['return'],
            'beats_baseline_delayed':row['scenarios']['delay_4h']['return']>baseline['scenarios']['delay_4h']['return']}


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():raise ValueError('Use a fresh study directory')
    bars={s:load(s,args.data) for s in ['BTCUSDT','ETHUSDT']}
    protocol={'study':'004','steps':['frozen003 execution audit','causal confidence and execution-band implementation',
             'paired rolling development','stress, controls, known recent development and conditional transfer'],
        'modes':MODES,'primary_mode':'plain','secondary_mode':'hysteretic; if selected it is explicitly a best-of-two result',
        'seeds':SEEDS,'population':args.population,'generations':args.generations,'folds':FOLDS,'embargo_days':7,
        'plain_thresholds':[-.35,.35],'hysteresis_extreme_thresholds':[-.5,.5],'hysteresis_transition_thresholds':[-.15,.15],
        'trend_rule':'lagged180bar EMA distance >0; identical long-only baseline rule; trend off resets confidence to0',
        'state':'reset flat at each training/validation window; indicator warmup uses causal past, confidence never carries across folds',
        'band':BAND,'band_semantics':'absolute equity-fraction notional gap; entry/full exit/flip bypass; small nonzero trims may be suppressed; initial target capped0.5 but exposure drifts',
        'fitness':'duration-weighted annual excess log net growth over banded baseline minus0.5 cross-block excess dispersion and drawdown beyond20%; complexity penalty0.0001/connection',
        'cash_yield':0.,'fee_bps_per_side':5,'slippage_bps_per_side':2,
        'selection':'NEAT only: all inherited absolute development gates AND positive total/medianpaired/3of4fold excess versus banded baseline AND excess under doubled cost and4h delay; best medianpaired excess; else no candidate',
        'known_data':'All BTC/ETH through August2026 is development already inspected; current2026 replay is context, never fresh validation',
        'recent_context':['2026-01-01','2026-08-31 20:00'],
        'transfer':{'symbols':['SOLUSDT','BNBUSDT'],'period':['2026-01-01','2026-08-31 20:00'],
                    'rule':'only selected incremental NEAT candidate; baseline alone never unlocks transfer',
                    'limit':'same observed calendar regime, correlated assets; no independent-time profitability claim'},
        'stopping':'do not alter thresholds/reselect using recent context or transfer; failed NEAT remains unqualified',
        'source_hashes':snapshot(out),'manifest_hash':sha256(Path(args.data,'manifest.json').read_bytes()).hexdigest(),
        'neat_version':neat.__version__,'numpy_version':np.__version__,'pandas_version':pd.__version__}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    shutil.copyfile(Path(args.data,'manifest.json'),out/'data-manifest.json')
    start=time.monotonic()
    jobs=[(training_markets(bars,a),args.population,args.generations,seed,mode,f'fold{fold}_{mode}_{seed}',str(out))
          for fold,(a,z) in enumerate(FOLDS) for mode in MODES for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:winners=list(pool.map(train_one,jobs))
    models={x['name']:x for x in winners};c=config(args.population)
    sequences={};asset_sequences={};logs={};seed_rows=[];dates=[]
    scenarios=[('base',{}),('cost_1_5',{'multiplier':1.5}),('double_cost',{'multiplier':2}),('delay_4h',{'delay':1})]
    for fold,(a,z) in enumerate(FOLDS):
        markets=[make_market(s,b,a,z) for s,b in bars.items()];dates.extend(markets[0].times[1:])
        candidates=controls()+[{'name':f'neat_{mode}','kind':'ensemble','mode':mode,
                               'members':[models[f'fold{fold}_{mode}_{s}'] for s in SEEDS]} for mode in MODES]
        for candidate in candidates:
            name=candidate['name']
            for scenario,opts in scenarios:
                r=evaluate(candidate,markets,c,**opts);port=portfolio(r)
                sequences.setdefault((name,scenario),[]).append(port['returns'])
                if scenario=='base':
                    logs.setdefault(name,[]).append({'period':[a,z],'portfolio':summary(port),'by_symbol':{s:summary(v) for s,v in r.items()}})
                    for s,v in r.items():asset_sequences.setdefault((name,s),[]).append(v['returns'])
        for mode in MODES:
            for seed in SEEDS:
                member=models[f'fold{fold}_{mode}_{seed}'];r=evaluate(member,markets,c)
                seed_rows.append({'fold':fold,'mode':mode,'seed':seed,'train_cash':member['train_cash'],
                                  'nodes':len(member['genome'].nodes),'connections':len(member['genome'].connections),
                                  'portfolio':summary(portfolio(r)),'by_symbol':{s:summary(v) for s,v in r.items()}})
    curves=pd.DataFrame(index=pd.DatetimeIndex([pd.Timestamp(FOLDS[0][0],tz='UTC')]+dates));rows=[]
    for name,folds in logs.items():
        row={'name':name,'folds':folds,'scenarios':{s:summary(stitched_metrics(sequences[name,s])) for s,_ in scenarios},
             'asset_compounded_returns':{s:stitched_metrics(asset_sequences[name,s])['return'] for s in bars}}
        row['absolute_gates']=development_gate(row);row['absolute_pass']=all(row['absolute_gates'].values())
        row['bootstrap_ci95_descriptive']=bootstrap_ci(stitched_metrics(sequences[name,'base'])['returns'])
        rows.append(row);curves[name]=stitched_metrics(sequences[name,'base'])['equity']
    baseline=next(r for r in rows if r['name']=='baseline_banded')
    for row in rows:
        row['relative_gates']=relative_gates(row,baseline)
        row['incremental_neat_pass']=row['name'].startswith('neat_') and row['absolute_pass'] and all(row['relative_gates'].values())
        row['median_paired_excess']=float(np.median([a['portfolio']['return']-b['portfolio']['return'] for a,b in zip(row['folds'],baseline['folds'])]))
    eligible=[r for r in rows if r['incremental_neat_pass']]
    selected=max(eligible,key=lambda r:r['median_paired_excess'])['name'] if eligible else None
    selection={'selected':selected,'development':rows,'per_seed':seed_rows,'candidate_count':len(rows),
               'population_evaluation_budget':len(jobs)*args.population*args.generations,'development_only':True}
    (out/'selection.json').write_text(json.dumps(selection,indent=2));curves.to_csv(out/'development_equity.csv')
    print(f'Frozen incremental NEAT selection: {selected}',flush=True)
    # Fourth step: refit the two predeclared modes for already-known 2026 context.
    final_jobs=[(training_markets(bars,'2026-01-01'),args.population,args.generations,seed,mode,f'final_{mode}_{seed}',str(out)) for mode in MODES for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:final_models=list(pool.map(train_one,final_jobs))
    final_candidates=[{'name':f'neat_{mode}','kind':'ensemble','mode':mode,'members':[m for m in final_models if m['mode']==mode]} for mode in MODES]
    with (out/'frozen_final_models.pkl').open('wb') as f:pickle.dump({'config':c,'candidates':final_candidates,'selected':selected},f)
    recent=[make_market(s,b,*protocol['recent_context']) for s,b in bars.items()]
    recent_curves=pd.DataFrame(index=recent[0].times);context={}
    for candidate in controls()+final_candidates:
        r=evaluate(candidate,recent,c);port=portfolio(r)
        context[candidate['name']]={'portfolio':summary(port),'by_symbol':{s:summary(v) for s,v in r.items()}}
        recent_curves[candidate['name']]=port['equity']
    recent_curves.to_csv(out/'known2026_equity.csv')
    report={'selected':selected,'incremental_development_pass':bool(eligible),'known2026_context':context,
            'transfer_accessed':False,'research_gate_pass':False,'final_refit_evaluation_budget':len(final_jobs)*args.population*args.generations}
    if selected is not None:
        candidate=next(x for x in final_candidates if x['name']==selected)
        transfer_root=out/'transfer_data'
        download(str(transfer_root),start='2025-11',end='2026-08',symbols=protocol['transfer']['symbols'])
        newbars={s:load(s,str(transfer_root)) for s in protocol['transfer']['symbols']}
        markets=[make_market(s,b,*protocol['transfer']['period']) for s,b in newbars.items()]
        report['transfer_accessed']=True;report['transfer']={};report['transfer_baseline']={}
        for scenario,opts in scenarios:
            r=evaluate(candidate,markets,c,**opts);b=evaluate({'kind':'baseline'},markets,c,**opts)
            report['transfer'][scenario]={'portfolio':summary(portfolio(r)),'by_symbol':{s:summary(v) for s,v in r.items()}}
            report['transfer_baseline'][scenario]={'portfolio':summary(portfolio(b)),'by_symbol':{s:summary(v) for s,v in b.items()}}
        report['research_gate_pass']=(all(v['return']>0 for r in report['transfer'].values() for v in r['by_symbol'].values())
              and report['transfer']['base']['portfolio']['max_drawdown']<=.20
              and all(report['transfer'][s]['portfolio']['return']>report['transfer_baseline'][s]['portfolio']['return'] for s in ['base','double_cost','delay_4h']))
    report['elapsed_seconds']=time.monotonic()-start
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data');p.add_argument('--output',default='artifacts/experiment_004')
    p.add_argument('--population',type=int,default=64);p.add_argument('--generations',type=int,default=30);p.add_argument('--workers',type=int,default=3)
    args=p.parse_args()
    if min(args.generations,args.workers)<1 or args.population<2:p.error('Invalid training parameters')
    run(args)
