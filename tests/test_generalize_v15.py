from copy import deepcopy
import pickle
import numpy as np
import pandas as pd
import pytest

pytest.importorskip('sklearn')
from neattrade import generalize_v15 as g
from neattrade.market import Market
from neattrade.repair_v12 import predict


def rows():
    result = []
    for i in range(80):
        t = g.utc('2023-01-01')+pd.Timedelta(days=i*3)
        s = g.SYMBOLS[int(i>=60)]
        result.append(dict(id=f'{s}_{t.isoformat()}_exit', head='exit', decision=str(t),
            known=str(t+pd.Timedelta(days=2)), left_censored=False,
            x=(np.sin(np.arange(18)+i)+i/40).tolist(), y=float(np.cos(i)*.02)))
    return result


@pytest.mark.parametrize('family', ['current', 'compact', 'robust'])
def test_held_asset_and_future_payload_cannot_change_fit(family):
    data = rows()
    a = g.train(data, '4h', family, None, 'ETHUSDT', '2024-01-01')
    other = deepcopy(data)
    for r in other:
        if r['id'].startswith('ETH'):
            r.update(x=[np.nan]*18, y=1e12)
    future = deepcopy(data[0]); future.update(id='BTCUSDT_future_exit', known='2024-01-01', x=[np.nan]*18, y=1e12)
    b = g.train(other+[future], '4h', family, None, 'ETHUSDT', '2024-01-01')
    assert a['training_ids']==b['training_ids'] and a['clip_limits']==b['clip_limits']
    for k in ['raw_y', 'fit_y', 'weights', 'training_predictions']:
        np.testing.assert_array_equal(a[k], b[k])
    np.testing.assert_array_equal(a['scaler'].mean_, b['scaler'].mean_)


def test_cutoff_censoring_and_window_boundary():
    data = rows()[:4]; cutoff = g.utc('2023-12-25')
    data[0]['known'] = str(cutoff)
    data[1]['known'] = None
    data[2]['left_censored'] = True
    assert len(g.cohort(data, '4h', '2024-01-01'))==1
    for i, r in enumerate(data):
        r.update(left_censored=False, decision=str(cutoff-pd.Timedelta(days=365, seconds=i)), known='2023-06-01')
    assert len(g.cohort(data, '4h', '2024-01-01', 365))==1


@pytest.mark.parametrize('family', ['robust', 'ridge'])
def test_training_only_winsor_weights_and_scaler(family):
    data = rows(); data[-1]['y'] = 100.
    m = g.train(data, '4h', family, None, None, '2024-01-01')
    assert len(m['training_ids'])==80
    np.testing.assert_allclose(m['clip_limits'], np.quantile([r['y'] for r in data], [.025, .975]))
    w = np.array([1/np.sqrt(m['groups'].count(c)) for c in m['groups']]); w /= w.mean()
    np.testing.assert_allclose(w, m['weights'])
    x = np.array([r['x'] for r in data])[:, g.INDICES]
    np.testing.assert_allclose(m['scaler'].mean_, np.average(x, weights=w, axis=0))
    assert m['fit_y'][-1]<100 and m['raw_y'][-1]==100
    assert m['effective_weight_count']<80


@pytest.mark.parametrize('family', ['compact', 'robust', 'ridge'])
def test_actual_model_shape_and_pickle_roundtrip(family):
    m = g.train(rows(), '4h', family, None, None, '2024-01-01')
    assert m['features']==g.FEATURES and m['scaler'].n_features_in_==8
    if family!='ridge':
        assert m['estimator'].get_params()['early_stopping'] is False
        assert all(t[0].nodes['is_leaf'].sum()<=3 for t in m['estimator']._predictors)
        assert all(max(t[0].nodes['depth'])<=2 for t in m['estimator']._predictors)
    np.testing.assert_array_equal(m['training_predictions'], predict(pickle.loads(pickle.dumps(m)), np.array([r['x'] for r in rows()])[:, g.INDICES]))


@pytest.mark.parametrize('n', [0, 19, 20])
def test_insufficient_cohort_is_explicit_abstention(n):
    m = g.train(rows()[:n], '4h', 'robust', None, None, '2024-01-01')
    assert m['fitted']==(n>=20) and m['training_count']==n
    if n<20:
        assert 'scaler' not in m and m['mean']==0


def constant(date, value):
    return dict(kind='mean', features=list(range(18)), mean=value, available=str(g.utc(date)), fitted=True)


@pytest.mark.parametrize('values,expected', [([-.1, -.2, -.3], True), ([-.1, -.1, 0.], False), ([-.1, -.1, .1], False)])
def test_guard_requires_every_component_strictly_negative(values, expected):
    models = {str(i): constant('2024-01-01', v) for i, v in enumerate(values)}
    score, _, _ = g.forecast(models, list(models), np.zeros(18), 'temporal_guard')
    assert (score<0)==expected
    models['0']['fitted'] = False
    score, values, reason = g.forecast(models, list(models), np.zeros(18), 'temporal_mean')
    assert (score, values, reason)==(0., [], 'insufficient')


@pytest.mark.parametrize('delay', [1, 2])
@pytest.mark.parametrize('arm', ['robust8', 'temporal_guard', 'transfer_robust'])
def test_version_pinning_natural_exit_and_boundary_protection(delay, arm):
    n = 640; ix = pd.date_range('2024-01-30', periods=n+1, freq='15min', tz='UTC')
    x = np.zeros((n, 13)); x[:, 0] = -1; x[:, -1] = .8
    x[:128, 0] = 1; x[160:304, 0] = 1; x[384:576, 0] = 1
    prices = np.full(n+1, 100.)
    ctx = dict(market=Market('BTCUSDT', ix, x, prices, prices.copy(), np.zeros(n+1), prices[:-1], prices[:-1]), atr=np.ones(n), closes=prices[:-1])
    models = {v: constant(date, value) for date, value in [('2024-01-01', .1), ('2024-02-01', -.1)]
              for v in g.components('4h', arm, date, 'BTCUSDT')}
    p, a, events = g.policy(ctx, '4h', arm, models, delay)
    exits = [e for e in events if e['head']=='exit']
    assert exits[0]['boundary_protected'] and not exits[0]['versions']
    assert exits[1]['versions']==g.components('4h', arm, '2024-01-01', 'BTCUSDT')
    assert not exits[1]['blocked'] and exits[2]['blocked']
    assert p[96+delay]>0 and p[256+delay]>0 and p[480+delay]==0
    assert np.all(p[480+delay:576+delay]==0) and a[480+delay]
    # A natural exit exactly at the checkpoint has priority and no model event.
    ctx['market'].x[480:, 0] = -1
    _, _, ev = g.policy(ctx, '4h', arm, models, delay)
    assert not any(e['decision']==480 for e in ev)


@pytest.mark.parametrize('change', ['duplicate', 'timeframe', 'payload', 'clock'])
def test_invalid_eligible_data_rejected(change):
    data = rows()
    if change=='duplicate': data.append(deepcopy(data[0]))
    if change=='timeframe': data[0]['timeframe']='2h'
    if change=='payload': data[0]['x']=[0.]*17
    if change=='clock': data[0]['known']=data[0]['decision']
    with pytest.raises(ValueError): g.train(data, '4h', 'robust', None, None, '2024-01-01')


def test_diagnostics_do_not_score_abstention_or_guard_maximum():
    from neattrade.research_v15 import diagnostics
    a = dict(name='4h_temporal_guard', scenario='base', head='exit', symbol='BTCUSDT',
        entry='2024-01-02', fold=0, boundary_protected=False, prediction=.1,
        component_predictions=[-.1, -.1, .1], blocked=False)
    r = dict(id='BTCUSDT_2024-01-02T00:00:00+00:00_exit', head='exit', left_censored=False, known='2024-01-10', y=-.02)
    d = diagnostics([a], {'4h': [r]})[0]
    assert d['payoff_prediction']==pytest.approx(-.1/3) and d['regret']==.02
    a.update(component_predictions=[], prediction=0.)
    d = diagnostics([a], {'4h': [r]})[0]
    assert d['absolute_error'] is None and d['regret']==.02
    r['known']=None; a['blocked']=True
    assert diagnostics([a], {'4h': [r]})[0]['regret'] is None


@pytest.mark.parametrize('family', ['current', 'compact', 'robust', 'ridge'])
def test_independent_fit_and_arithmetic_replay(family):
    from neattrade.verify_v15 import audit_model
    data = rows(); held = 'ETHUSDT' if family=='current' else None
    m = g.train(data, '4h', family, None, held, '2024-01-01')
    assert audit_model(m, data)
    corrupt = deepcopy(m); corrupt['fit_y'][0] += .1
    with pytest.raises(AssertionError): audit_model(corrupt, data)


def test_independent_block_indices_reproduce_prefix_arithmetic():
    from neattrade.verify_v15 import direct_interval
    from neattrade.research_v15 import interval
    from neattrade.verify_v13 import equal
    x = [np.sin(np.arange(31))*.01, np.arange(35)*.0002]
    y = [np.zeros(31), np.full(35, .0001)]
    equal(interval(x, y), direct_interval(x, y), 2e-12)
