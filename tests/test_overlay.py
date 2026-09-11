import numpy as np
import pandas as pd
import pytest

from neattrade.batch import fitness_accounting,prepare
from neattrade.market import Market,simulate
from neattrade.overlay import baseline_targets,confidence,incremental_objective,overlay_targets


def test_hysteresis_ignores_threshold_noise_but_exits_on_strong_signal():
    raw=np.array([[.6,.2,.1,-.1,.1,-.6,.2,.6]])
    out=confidence(raw,np.ones(8,bool),'hysteretic')
    np.testing.assert_array_equal(out,[[1,1,1,1,1,0,.5,1]])


def test_trend_exit_resets_memory_and_uses_no_future_values():
    raw=np.array([[.7,.1,.1,.7]])
    trend=np.array([True,False,True,True])
    np.testing.assert_array_equal(confidence(raw,trend,'hysteretic'),[[1,0,0,1]])
    modified=raw.copy();modified[:,2:]=-1
    np.testing.assert_array_equal(confidence(raw,trend,'hysteretic')[:,:2],confidence(modified,trend,'hysteretic')[:,:2])


def test_overlay_can_only_reduce_long_baseline_and_allows_cash():
    x=np.zeros((10,10));x[:,5]=1;x[:,6]=.8
    raw=np.random.default_rng(3).uniform(-1,1,(20,10))
    for mode in ['plain','hysteretic']:
        targets=overlay_targets(raw,x,mode)
        assert np.all(targets>=0)
        assert np.all(targets<=baseline_targets(x))
        assert not overlay_targets(np.full((1,10),-1.),x,mode).any()


def test_band_never_blocks_exit_entry_or_flip():
    prices=np.array([100.,100.,100.,100.])
    m=Market('X',pd.date_range('2024-01-01',periods=4,freq='4h',tz='UTC'),np.zeros((3,10)),prices,prices,np.zeros(4),prices[:-1],prices[:-1])
    r=simulate(m,[.02,-.02,0],rebalance_band=.05,trace=True)
    assert [v['order_category'] for v in r['ledger']]==['entry','flip','exit']
    small=simulate(m,[.3,.31,.32],fee_bps=0,slippage_bps=0,rebalance_band=.05,trace=True)
    assert [v['units'] for v in small['ledger']]==[.003,.003,.003]


@pytest.mark.parametrize('band',[0,.02,.05,.10])
def test_population_band_matches_reference(band):
    rng=np.random.default_rng(25);n=200
    price=100*np.exp(np.r_[0.,np.cumsum(rng.normal(0,.01,n))])
    m=Market('X',pd.date_range('2024-01-01',periods=n+1,freq='4h',tz='UTC'),np.zeros((n,10)),
             price,price*1.001,rng.normal(0,.0002,n+1),np.minimum(price[:-1],price[1:]),np.maximum(price[:-1],price[1:]))
    d=prepare(m)
    targets=rng.choice([-.5,-.48,0,.2,.22,.4],size=(10,len(d.anchors)))
    actual=fitness_accounting(d,targets,rebalance_band=band)
    allowed=np.zeros(n,bool);allowed[d.anchors]=True
    for i in range(len(targets)):
        full=np.zeros(n);full[d.anchors]=targets[i]
        expected=simulate(m,full,trade_allowed=allowed,rebalance_band=band)
        for key in ['return','max_drawdown','cost_paid_initial_equity','funding_paid_initial_equity']:
            assert actual[key][i]==pytest.approx(expected[key],abs=2e-12)


def test_incremental_fitness_does_not_reward_churn_or_copying():
    b={'return':np.array([.05]),'years':.5,'max_drawdown':np.array([.1]),'ruined':np.array([False])}
    np.testing.assert_allclose(incremental_objective([b],[b]),0)
    worse=dict(b,**{'return':np.array([.04])})
    assert incremental_objective([worse],[b])[0]<0


def test_trace_reconciles_and_invalid_band_fails():
    price=np.array([100.,110.,105.])
    m=Market('X',pd.date_range('2024-01-01',periods=3,freq='4h',tz='UTC'),np.zeros((2,10)),price,price,np.array([0,.001,-.001]),np.array([100,105]),np.array([110,110]))
    r=simulate(m,[.3,.2],trace=True)
    total=sum(v['price_pnl']-v['funding_paid']-v['trade_cost']-v['terminal_exit_cost'] for v in r['ledger'])
    assert total==pytest.approx(r['return'])
    with pytest.raises(ValueError):
        simulate(m,[0,0],rebalance_band=-1)


@pytest.mark.parametrize('mode',['plain','hysteretic'])
def test_overlay_training_and_reference_inference_match(mode):
    import neat
    from neattrade.research import batch_activate,config
    from neattrade.research_v3 import daily_schedule
    from neattrade.research_v4 import desired
    rng=np.random.default_rng(14);n=80
    price=100*np.exp(np.r_[0.,np.cumsum(rng.normal(0,.01,n))])
    x=rng.normal(size=(n,10));x[:,6]=.5
    m=Market('X',pd.date_range('2024-01-01 04:00',periods=n+1,freq='4h',tz='UTC'),x,price,price,np.zeros(n+1),price[:-1]*.95,price[:-1]*1.05)
    c=config(8);g=next(iter(neat.Population(c,seed=42).population.values()))
    net=neat.nn.FeedForwardNetwork.create(g,c);d=prepare(m)
    target=overlay_targets(batch_activate(net,d.x)[None,:],d.x,mode)
    fast=fitness_accounting(d,target,rebalance_band=.05)
    full=desired({'kind':'gate','mode':mode,'genome':g},m,c)
    reference=simulate(m,full,trade_allowed=daily_schedule(m),rebalance_band=.05)
    assert fast['return'][0]==pytest.approx(reference['return'],abs=2e-12)
    assert fast['max_drawdown'][0]==pytest.approx(reference['max_drawdown'],abs=2e-12)
    changed=m.x.copy();changed[2:]*=-1
    from dataclasses import replace
    later=desired({'kind':'gate','mode':mode,'genome':g},replace(m,x=changed),c)
    np.testing.assert_array_equal(full[:5],later[:5])
