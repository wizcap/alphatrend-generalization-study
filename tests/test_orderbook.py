import numpy as np
import pandas as pd
import pytest
import neat

from neattrade.orderbook import snapshots, book_feature, LOOKBACK_SECONDS
from neattrade.research_v9 import (base_targets, targets, network_inputs, prepare,
    allowed_times, desired, train_markets, POPULATION)
from neattrade.research_v5 import model_config
from neattrade.batch import fitness_accounting
from neattrade.market import Market, simulate


def book_rows():
    return pd.DataFrame([
        ['2024-01-01 00:00:00', -1, 2., 200.], ['2024-01-01 00:00:00', 1, 1., 100.],
        ['2024-01-01 00:00:00', -2, 30., 3000.], ['2024-01-01 00:00:00', 2, 10., 1000.]],
        columns=['timestamp', 'percentage', 'depth', 'notional'])


def test_only_paired_one_percent_notional_not_overlapping_depth_bands():
    t, v, audit = snapshots(book_rows(), '2024-01-01')
    assert len(t) == 1 and v[0] == pytest.approx(1/3)
    assert audit['invalid_snapshots'] == 0


def test_missing_ambiguous_and_nonpositive_snapshot_break_state():
    for f in [book_rows().iloc[[0,2,3]], book_rows().iloc[[2,3]],
              pd.concat([book_rows(), pd.DataFrame([['2024-01-01 00:00:00',-1,2.,300.]], columns=book_rows().columns)])]:
        t, v, a = snapshots(f, '2024-01-01')
        assert len(t) == 1 and np.isnan(v[0]) and a['invalid_snapshots'] == 1
    f = book_rows()
    f.loc[0,'depth'] = 0
    assert np.isnan(snapshots(f,'2024-01-01')[1][0])


def test_identical_duplicates_are_audited_but_not_overweighted():
    f = book_rows()
    _, v, a = snapshots(pd.concat([f,f]),'2024-01-01')
    assert v[0] == pytest.approx(1/3)
    assert a['exact_duplicate_rows_removed'] == 2


def test_day_boundaries_and_schema_rejected():
    with pytest.raises(ValueError,match='outside'):
        snapshots(book_rows(),'2024-01-02')
    with pytest.raises(ValueError,match='schema'):
        snapshots(book_rows().drop(columns='notional'),'2024-01-01')


def regular_series():
    start = pd.Timestamp('2024-01-01',tz='UTC').value//10**9
    return np.arange(start,start+8*3600,30), start


def test_constant_signal_fully_covered_after_latency():
    t, start = regular_series()
    when = pd.to_datetime([start+4*3600+60],unit='s',utc=True)
    f = book_feature(t,np.full(len(t),.25),when)
    assert f.book_valid.iloc[0] and f.book_coverage.iloc[0] == pytest.approx(1.)
    assert f.book_imbalance.iloc[0] == pytest.approx(.25)


def test_future_mutation_and_truncation_preserve_earlier_features():
    t,start = regular_series()
    v=np.sin(np.arange(len(t))/20)
    when=pd.to_datetime([start+5*3600],unit='s',utc=True)
    ref=book_feature(t,v,when)
    changed=v.copy()
    changed[t>=start+5*3600-60]=-1
    pd.testing.assert_frame_equal(ref,book_feature(t,changed,when))
    past=t<start+5*3600-60
    pd.testing.assert_frame_equal(ref,book_feature(t[past],v[past],when))


def test_integral_is_weighted_by_time_not_snapshot_count():
    t,start=regular_series()
    # Alternate held durations20s and40s, with values1 and0.
    t=np.ravel(np.column_stack([np.arange(start,start+8*3600,60), np.arange(start+20,start+8*3600,60)]))
    v=np.tile([1.,0.],len(t)//2)
    f=book_feature(t,v,pd.to_datetime([start+4*3600+60],unit='s',utc=True))
    assert f.book_imbalance.iloc[0] == pytest.approx(1/3)


def test_stale_or_invalid_last_snapshot_cannot_be_forward_filled():
    t,start=regular_series()
    when=pd.to_datetime([start+4*3600],unit='s',utc=True)
    keep=t<start+4*3600-200
    assert not book_feature(t[keep],np.ones(keep.sum()),when).book_valid.iloc[0]
    v=np.ones(len(t))
    v[t==start+4*3600-90]=np.nan
    f=book_feature(t,v,when)
    assert not f.book_valid.iloc[0] and f.book_imbalance.iloc[0] == 0


def test_long_gap_fails_coverage_and_neutral_fill_is_explicit():
    t,start=regular_series()
    keep=(t<start+3600)|(t>=start+2*3600)
    f=book_feature(t[keep],np.ones(keep.sum()),pd.to_datetime([start+4*3600],unit='s',utc=True))
    assert f.book_coverage.iloc[0]<.8
    assert not f.book_valid.iloc[0] and f.book_imbalance.iloc[0] == 0


def test_snapshot_validation_and_utc_decision_requirement():
    with pytest.raises(ValueError):
        book_feature([1,1],[0,0],pd.date_range('2024-01-01',periods=1,tz='UTC'))
    with pytest.raises(ValueError):
        book_feature([1,2],[0,2],pd.date_range('2024-01-01',periods=1,tz='UTC'))
    with pytest.raises(ValueError):
        book_feature([1,2],[0,0],pd.date_range('2024-01-01',periods=1))


def test_same_fallback_for_all_raw_outputs_and_zero_column_control():
    x=np.zeros((4,7));x[:,0]=[1,1,-1,1];x[:,5]=[0,1,0,1];x[:,6]=.5;x[:,4]=[-.8,.2,.7,.4]
    p=targets(np.array([[-1,-1,1,1],[1,1,-1,-1]]),x)
    np.testing.assert_allclose(p[:,0],base_targets(x)[0])
    assert (p[:,2]==0).all()
    assert network_inputs(x,'at_zero').shape == network_inputs(x,'at_book').shape == (4,5)
    assert (network_inputs(x,'at_zero')[:,4]==0).all()
    np.testing.assert_array_equal(network_inputs(x,'at_book')[:,:4],network_inputs(x,'at_zero')[:,:4])


def market_fixture():
    dates=pd.date_range('2024-01-01',periods=193,freq='15min',tz='UTC')
    p=100+np.sin(np.arange(193)/11)*3+np.arange(193)/50
    x=np.zeros((192,7));x[:,0]=1;x[:,4]=np.sin(np.arange(192)/8);x[:,5]=1;x[:,6]=.5
    funding=np.where(np.arange(193)%32==0,.001,0.)
    return Market('TEST',dates,x,p,p*1.002,funding,p[:-1]-.3,p[:-1]+.3)


@pytest.mark.parametrize('delay,cost',[(0,7),(0,14),(1,7)])
def test_segmented_financial_accounting_matches_scalar_15m_grid(delay,cost):
    m=market_fixture()
    p,allowed=desired('alpha',m)
    if delay:
        p=np.r_[np.zeros(delay),p[:-delay]]
        allowed=np.r_[np.zeros(delay,bool),allowed[:-delay]]
    d=prepare(m,allowed)
    fast=fitness_accounting(d,p[d.anchors][None,:],cost_bps=cost,rebalance_band=.05)
    ref=simulate(m,p,cost,0,trade_allowed=allowed,rebalance_band=.05)
    for k in ['return','max_drawdown','cost_paid_initial_equity','funding_paid_initial_equity']:
        assert fast[k][0] == pytest.approx(ref[k],abs=2e-12)


def test_five_input_network_can_use_book_and_mask_still_overrides_it():
    m=market_fixture()
    config=model_config(POPULATION,5)
    g=next(iter(neat.Population(config,seed=909).population.values()))
    for c in g.connections.values(): c.weight=0
    g.connections[(-5,0)].weight=3
    g.nodes[0].bias=0;g.nodes[0].response=1
    model={'genome':g,'spec':'at_book'}
    p,a=desired(model,m)
    zero,_=desired(model,m,zero_book=True)
    assert np.any(np.abs(p[a]-zero[a])>.01)
    m.x[:,5]=0
    p,a=desired(model,m)
    np.testing.assert_allclose(p[a],base_targets(m.x[a]))


def test_expanding_training_bounds_enforce_seven_day_embargo(monkeypatch):
    import neattrade.research_v9 as r
    monkeypatch.setattr(r,'market',lambda s,b,f,a,z:(s,a,z))
    got=train_markets({'BTC':None},{'BTC':None},'2024-07-01')
    assert got[0][1]=='2023-01-02'
    assert got[-1][2]=='2024-06-24'
    assert all(pd.Timestamp(a)<pd.Timestamp(z)<=pd.Timestamp('2024-06-24') for _,a,z in got)


def test_fixed_mlp_matches_independent_scalar_forward():
    from neattrade.policies import mlp
    rng=np.random.default_rng(9)
    weights=rng.normal(size=(3,29));x=rng.normal(size=(7,5))
    reference=[]
    for w in weights:
        reference.append([np.tanh(sum(np.tanh(sum(v[i]*w[4*i+j] for i in range(5))+w[20+j])*w[24+j]
                                      for j in range(4))+w[28]) for v in x])
    np.testing.assert_allclose(mlp(weights,x),reference,atol=1e-14,rtol=0)


def test_cma_adapter_has_reproducible_fixed_budget_and_saved_best():
    from neattrade.evolution import fit_cma
    calls=[]
    def fitness(w):
        assert np.max(np.abs(w))<=5
        value=-np.sum((w-.2)**2,axis=1)
        calls.extend(value.tolist())
        return value
    w,value,trace=fit_cma(fitness,11,8,3)
    assert len(calls)==24 and sum(r['candidate_evaluations'] for r in trace)==24
    assert value==pytest.approx(max(calls))
    assert value==pytest.approx(-np.sum((w-.2)**2))
    other,score,_=fit_cma(lambda v:-np.sum((v-.2)**2,axis=1),11,8,3)
    np.testing.assert_allclose(w,other,atol=1e-12)
    assert score==pytest.approx(value)


def test_cma_policy_uses_same_external_missing_data_fallback():
    m=market_fixture()
    w=np.zeros(29);w[28]=-4
    model={'kind':'cma','spec':'cma_book','weights':w}
    p,a=desired(model,m)
    assert (p==0).all()
    m.x[:,5]=0
    p,a=desired(model,m)
    np.testing.assert_allclose(p[a],base_targets(m.x[a]))


def test_crossed_or_incompatible_band_vwaps_are_invalid():
    for bad in [130.,90.]:
        frame=book_rows()
        frame.loc[1,'notional']=bad
        _,v,a=snapshots(frame,'2024-01-01')
        assert np.isnan(v[0]) and a['inconsistent_band_vwap']==1


def test_side_staleness_is_causal_and_crosses_day_boundaries():
    from neattrade.orderbook import stale_sides
    t=np.arange(0,300,30)+pd.Timestamp('2024-01-01 23:59:00',tz='UTC').value//10**9
    sides=np.column_stack([np.ones(10),np.ones(10)*100,np.arange(10)+1,np.arange(10)+100])
    invalid=stale_sides(t,sides)
    assert not invalid[:5].any() and invalid[5:].all()
    changed=sides.copy();changed[7:,0]=2
    np.testing.assert_array_equal(stale_sides(t,changed)[:7],invalid[:7])
    assert not stale_sides(t,changed)[7:].any()


def test_alpha_bridge_keeps_unfinished_four_hour_candles_out():
    from neattrade.research_v9 import alpha_inputs
    n=96*7
    ix=pd.date_range('2024-01-01',periods=n,freq='15min',tz='UTC')
    p=100+np.sin(np.arange(n)/10)*3
    b=pd.DataFrame({'open':p,'high':p+1,'low':p-1,'close':p+.2,'volume':np.arange(n)+100.},index=ix)
    cut=ix[96*5]
    original=alpha_inputs(b)
    assert original.loc[cut].notna().all()
    future=b.copy();future.loc[cut:,:]*=3
    pd.testing.assert_frame_equal(original.loc[:cut],alpha_inputs(future).loc[:cut])


def test_shared_objective_handles_ruin_but_rejects_other_nonfinite_values():
    from neattrade.research_v9 import population_scores
    reference={'return':np.array([.1]),'years':1.}
    result={'return':np.array([0.,.2]),'max_drawdown':np.array([.1,.3]),
            'ruined':np.array([True,False]),'years':1.}
    score=population_scores([result],[reference],np.array([5,24]))
    assert np.isfinite(score).all() and score[0]<-99
    result['return'][1]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):
        population_scores([result],[reference],24)
