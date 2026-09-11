"""Public reproduction of study015, without private operational bookkeeping."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import pickle
import sys
import urllib.request
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
E=ROOT/'evidence/study015'
SCENARIOS={'base':(7.,1),'double_cost':(14.,1),'delay30m':(7.,2)}


def same(a,b):
    np.testing.assert_allclose(a,b,atol=2e-9,rtol=0)


def check_manifest():
    manifest=json.loads((ROOT/'PUBLIC_MANIFEST.json').read_text())
    for name,h in manifest['files'].items():
        p=ROOT/name
        assert p.is_file() and sha256(p.read_bytes()).hexdigest()==h,name
    return len(manifest['files'])


def days(returns,times):
    index=(pd.DatetimeIndex(times).tz_localize(None)-pd.Timedelta(nanoseconds=1)).to_period('D')
    return pd.Series(1+returns,index=index).groupby(level=0).prod()-1


def boot(a,b):
    ds=[np.log1p(x)-np.log1p(y) for x,y in zip(a,b)]
    result={}
    for width in [7,28]:
        rng=np.random.default_rng(1015); values=np.zeros(100000)
        for d in ds:
            n=len(d); count=(n+width-1)//width
            for first in range(0,100000,2000):
                starts=rng.integers(0,n,(2000,count))
                index=((starts[:,:,None]+np.arange(width))%n).reshape(2000,-1)[:,:n]
                values[first:first+2000]+=d[index].sum(axis=1)
        result[str(width)]=np.quantile(values,[.025/40,1-.025/40]).tolist()
    return result


def numerical(report,intervals=False):
    rows={r['name']:r for r in report['strategies']}; daily={}; checked=0
    for tf in ['2h','4h']:
        arms=[r['arm'] for r in report['strategies'] if r['timeframe']==tf]
        seq={}; sleeves={}; endpoints=[]
        for fold in range(6):
            window=rows[tf+'_alpha']['folds']['base'][fold]['window']
            times=pd.date_range(window[0],window[1],freq='15min',tz='UTC')[1:]
            endpoints.append(times)
            with np.load(E/'accounts'/f'{tf}_{fold}.npz',allow_pickle=False) as f:
                for arm in arms:
                    for scenario in SCENARIOS:
                        expected=rows[tf+'_'+arm]['folds'][scenario][fold]
                        eqs=[]
                        for symbol in ['BTCUSDT','ETHUSDT']:
                            pre=f'{arm}__{scenario}__{symbol}__'
                            eq=f[pre+'equity']; p=f[pre+'positions']; allowed=f[pre+'allowed']
                            assert len(eq)==len(times)+1 and len(p)==len(times)==len(allowed)
                            assert np.isfinite(eq).all() and np.min(eq)>0 and eq[0]==1
                            assert np.isfinite(p).all() and np.min(p)>=0 and np.max(p)<=.50000001
                            actual=expected['by_symbol'][symbol]
                            same(eq[-1]-1,actual['return'])
                            same(np.max(1-eq/np.maximum.accumulate(eq)),actual['max_drawdown'])
                            sleeves.setdefault((arm,scenario,symbol),[]).append(eq[-1]-1)
                            eqs.append(eq); checked+=1
                        combined=(eqs[0]+eqs[1])/2
                        returns=combined[1:]/combined[:-1]-1
                        same(combined[-1]-1,expected['portfolio']['return'])
                        same(np.max(1-combined/np.maximum.accumulate(combined)),expected['portfolio']['max_drawdown'])
                        seq.setdefault((arm,scenario),[]).append(returns)
        for group,sl in [('development',slice(0,4)),('recent',slice(4,6))]:
            times=pd.DatetimeIndex([v for part in endpoints[sl] for v in part])
            with np.load(E/f'{tf}_{group}_equity.npz',allow_pickle=False) as f:
                np.testing.assert_array_equal(f['timestamps'][1:],times.as_unit('ns').asi8)
                for (arm,sc),parts in seq.items():
                    r=np.concatenate(parts[sl]); eq=np.r_[1.,np.cumprod(1+r)]
                    name=tf+'_'+arm; expected=rows[name][group][sc]
                    same(eq,f[name+'__'+sc]); same(eq[-1]-1,expected['return'])
                    same(np.max(1-eq/np.maximum.accumulate(eq)),expected['max_drawdown'])
                    d=days(r,times)
                    for year,v in d.groupby(d.index.year):
                        same(np.prod(1+v)-1,expected['annual_returns'][str(year)])
                    for s in ['BTCUSDT','ETHUSDT']:
                        same(np.prod(1+np.array(sleeves[arm,sc,s][sl]))-1,expected['by_symbol_return'][s])
        for arm in arms:
            daily[tf+'_'+arm]=[days(r,t).to_numpy() for r,t in zip(seq[arm,'base'][:4],endpoints[:4])]
    for c in report['comparisons']:
        a,b=daily[c['candidate']],daily[c['reference']]
        same(sum((np.log1p(x)-np.log1p(y)).sum() for x,y in zip(a,b)),c['log_wealth_excess'])
        if intervals:
            actual=boot(a,b)
            for width in ['7','28']: same(actual[width],c['intervals'][width])
    assert checked==720
    return {'derived_asset_curves':checked,'group_curves':120,'contrast_point_estimates':20,'bootstrap_intervals':40 if intervals else 0}


def download():
    manifest=json.loads((E/'raw_manifest.json').read_text())
    def one(row):
        relative=Path(row['path'].replace('\\','/'))
        assert not relative.is_absolute() and relative.parts[0]=='data' and '..' not in relative.parts
        target=ROOT/relative; target.parent.mkdir(parents=True,exist_ok=True)
        url=archive_url(row)
        payload=target.read_bytes() if target.exists() else urllib.request.urlopen(url,timeout=90).read()
        assert sha256(payload).hexdigest()==row['sha256'],str(relative)
        if not target.exists(): target.write_bytes(payload)
        target.with_suffix('.zip.CHECKSUM').write_text(row['sha256']+'  '+target.name,encoding='utf-8')
    with ThreadPoolExecutor(max_workers=6) as pool: list(pool.map(one,manifest))
    dest=ROOT/'data/study008/manifest.json'; dest.parent.mkdir(parents=True,exist_ok=True)
    # The legacy loader expects URLs on every row; preserve the frozen evidence
    # and normalize only this ignored working manifest.
    dest.write_text(json.dumps([dict(row,url=archive_url(row)) for row in manifest],indent=2),encoding='utf-8')
    return len(manifest)


def archive_url(row):
    """Retain recorded URLs; recover legacy funding URLs from the frozen name."""
    if 'url' in row:
        url=row['url']
    else:
        assert row.get('role')=='existing_official_funding'
        name=Path(row['path'].replace('\\','/')).name
        symbol=name.split('-')[0]
        assert symbol in ['BTCUSDT','ETHUSDT'] and '-fundingRate-' in name and name.endswith('.zip')
        url=f'https://data.binance.vision/data/futures/um/monthly/fundingRate/{symbol}/{name}'
    assert url.startswith('https://data.binance.vision/data/futures/um/')
    return url


def load_models():
    # Pickles are publisher-generated research models; validate PUBLIC_MANIFEST before loading.
    result={}
    for folder in ['models','reference_models']:
        for path in (E/folder).glob('*.pkl'):
            m=pickle.loads(path.read_bytes()); assert path.stem==m['name']; result[m['name']]=m
    assert len(result)==768
    return result


def refit(models):
    from neattrade.verify_v15 import audit_model
    labels=json.loads((E/'labels.json').read_text()); count=0
    for m in models.values():
        if 'family' not in m: continue
        assert audit_model(m,labels[m['timeframe']]); count+=1
        if count%88==0: print('Independent model refits',count,'/ 704',flush=True)
    assert count==704
    return count


def market_replay(models):
    import os
    os.chdir(ROOT)
    # Fetch is an explicit separate operation. Refuse missing/corrupt input here.
    manifest=json.loads((E/'raw_manifest.json').read_text())
    for item in manifest:
        p=ROOT/item['path'].replace('\\','/')
        assert p.exists() and sha256(p.read_bytes()).hexdigest()==item['sha256'],str(p)
    from neattrade.repair_v12 import dataset,context,TRAIN_START,END
    from neattrade.verify_v12 import reference_labels
    from neattrade.verify_v15 import reference_policy
    from neattrade.fast_accounting import audit_simulate
    from neattrade.research_v12 import WINDOWS
    bars,frames,atr=dataset()
    saved_labels=json.loads((E/'labels.json').read_text()); label_count=0; count=0
    for tf in ['2h','4h']:
        rebuilt=[]
        for s in ['BTCUSDT','ETHUSDT']:
            rebuilt+=reference_labels(context(s,bars[s],frames[tf][s],atr[tf][s],TRAIN_START,END),tf)
        expected=saved_labels[tf]; assert len(rebuilt)==len(expected)
        for a,b in zip(rebuilt,expected):
            assert a.keys()==b.keys()
            for k in a:
                if k in ['x','y'] and a[k] is not None: same(a[k],b[k])
                else: assert a[k]==b[k],k
        label_count+=len(rebuilt)
        for fold,(start,end) in enumerate(WINDOWS):
            contexts={s:context(s,bars[s],frames[tf][s],atr[tf][s],start,end) for s in ['BTCUSDT','ETHUSDT']}
            with np.load(E/'accounts'/f'{tf}_{fold}.npz',allow_pickle=False) as curves:
                for arm in ['alpha','current','compact8','robust8','temporal_mean','temporal_guard','ridge8','transfer_current','transfer_compact','transfer_robust']:
                    for sc,(cost,delay) in SCENARIOS.items():
                        for s,ctx in contexts.items():
                            p,a,_=reference_policy(ctx,tf,arm,models,delay)
                            prefix=f'{arm}__{sc}__{s}__'
                            same(p,curves[prefix+'positions']); np.testing.assert_array_equal(a,curves[prefix+'allowed'])
                            account=audit_simulate(ctx['market'],p,a,cost)
                            same(account['equity'],curves[prefix+'equity'])
                            count+=1
            print('Raw-data market accounts',count,'/ 720',flush=True)
    assert count==720
    return {'raw_archive_hashes':len(manifest),'rebuilt_labels':label_count,'market_accounts':count}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--intervals',action='store_true',help='Recompute all 40 bootstrap intervals')
    p.add_argument('--models',action='store_true',help='Independently refit the 704 new models')
    p.add_argument('--download',action='store_true',help='Retrieve and checksum the 355 public Binance archives')
    p.add_argument('--market',action='store_true',help='Rebuild labels and replay 720 accounts from already downloaded archives')
    args=p.parse_args(); checks={'public_hashes':check_manifest()}
    report=json.loads((E/'report.json').read_text()); checks.update(numerical(report,args.intervals))
    if args.download: checks['downloaded_archives']=download()
    if args.models or args.market:
        models=load_models()
        if args.models: checks['refitted_models']=refit(models)
        if args.market: checks.update(market_replay(models))
    output=ROOT/'reproduction'; output.mkdir(exist_ok=True)
    name='public_'+('_'.join(k for k in ['intervals','models','download','market'] if getattr(args,k)) or 'summary')+'.json'
    (output/name).write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))


if __name__=='__main__': main()
