from dataclasses import replace

import neat
import numpy as np
import pandas as pd
import pytest

from neattrade.alphatrend import (ALPHA_FEATURES, FEATURE_NAMES, STATE, alpha_features,
    alpha_indicator, alpha_targets, cross_state, gated_targets, select_inputs, specifications)
from neattrade.batch import fitness_accounting, prepare
from neattrade.market import Market, simulate
from neattrade.research import batch_activate
from neattrade.research_v3 import daily_schedule
from neattrade.research_v5 import model_config
from neattrade.research_v7 import desired, evaluate, schedule


def bars():
    rng = np.random.default_rng(707)
    n = 220
    close = 100*np.exp(np.cumsum(rng.normal(0, .025, n)))
    return pd.DataFrame({'open': close, 'high': close+1, 'low': close-1, 'close': close,
                         'volume': rng.uniform(1, 20, n)},
                        index=pd.date_range('2023-01-01', periods=n, freq='4h', tz='UTC'))


def test_hand_calculated_sma_atr_mfi_and_alpha_recurrence():
    close = np.array([10., 11., 12., 13., 12., 10., 9., 10., 11., 12.])
    b = pd.DataFrame({'high': close+1, 'low': close-1, 'close': close, 'volume': 1.})
    a = alpha_indicator(b, period=2)
    assert a.atr_sma.iloc[:2].isna().all()
    np.testing.assert_allclose(a.atr_sma.iloc[2:], [2, 2, 2, 2.5, 2.5, 2, 2, 2])
    np.testing.assert_allclose(a.mfi.iloc[2:], [100, 100, 52, 0, 0, 1000/19, 100, 100])
    np.testing.assert_allclose(a.line.iloc[2:], [9, 10, 10, 10, 10, 10, 10, 10])
    # A Wilder smoother would produce a different ATR after the gap.
    assert a.atr_sma.iloc[6] == 2.5


def test_crosses_hold_through_equal_plateaus_and_require_a_real_cross():
    line = pd.Series([np.nan, np.nan, 10, 10, 10, 11, 11, 11, 10, 10, 10, 11])
    buy, sell, state = cross_state(line)
    assert np.flatnonzero(buy).tolist() == [5, 11]
    assert np.flatnonzero(sell).tolist() == [8]
    np.testing.assert_array_equal(state, [0, 0, 0, 0, 0, 1, 1, 1, -1, -1, -1, 1])


def test_flat_zero_volume_has_no_invented_bull_signal():
    b = bars()
    b[['open', 'high', 'low', 'close']] = 100.
    b.volume = 0.
    a = alpha_indicator(b)
    assert a.mfi.isna().all()
    assert not a.buy.any() and not a.sell.any()
    assert not a.state.any()
    f = alpha_features(b).iloc[30:]
    assert (f[['alpha_state', 'alpha_slope', 'alpha_mfi']] == 0.).all().all()
    # The original nz bootstrap keeps the line at zero until a bullish branch.
    assert (f.alpha_distance == 5.).all()


def test_causal_prefix_and_current_bar_is_never_used_for_its_open():
    b = bars()
    base = alpha_features(b)
    for cut in [60, 137, 200]:
        shortened = alpha_features(b.iloc[:cut])
        pd.testing.assert_frame_equal(base.iloc[:cut], shortened)
        future = b.copy()
        future.iloc[cut:, :4] *= 3
        future.iloc[cut:, 4] *= 20
        changed = alpha_features(future)
        pd.testing.assert_frame_equal(base.iloc[:cut+1], changed.iloc[:cut+1])


def market():
    b = bars()
    n = len(b)-1
    x = np.zeros((n, len(FEATURE_NAMES)))
    x[:, 6] = .6
    x[:, 5] = -1.  # EMA always negative; AlphaTrend must still trade.
    x[:, -4:] = alpha_features(b).ffill().fillna(0.).to_numpy()[:-1]
    return Market('TEST', pd.date_range('2024-01-01', periods=n+1, freq='4h', tz='UTC'), x,
        b.close.to_numpy(), b.close.to_numpy()*1.001, np.full(n+1, .0001),
        np.minimum(b.close.to_numpy()[:-1], b.close.to_numpy()[1:])-1,
        np.maximum(b.close.to_numpy()[:-1], b.close.to_numpy()[1:])+1)


def test_standalone_alpha_is_independent_of_ema_and_has_signed_diagnostic():
    m = market()
    p = desired({'kind': 'alpha'}, m)
    assert (p > 0).any() and not (p < 0).any()
    changed = m.x.copy()
    changed[:, 5] = 100.
    np.testing.assert_array_equal(p, desired({'kind': 'alpha'}, replace(m, x=changed)))
    short = desired({'kind': 'alpha', 'long_short': True}, m)
    assert (short < 0).any() and (short > 0).any()
    np.testing.assert_array_equal(np.maximum(short, 0), p)


@pytest.mark.parametrize('name', ['at_constant', 'at_neat', 'at_momentum', 'at_funding', 'at_all'])
def test_neat_reference_and_independent_training_accounting(name):
    m = market()
    spec = next(s for s in specifications() if s['name'] == name)
    d = prepare(m)
    x = select_inputs(d.x, spec)
    c = model_config(8, inputs=x.shape[1])
    g = next(iter(neat.Population(c, seed=11).population.values()))
    net = neat.nn.FeedForwardNetwork.create(g, c)
    raw = batch_activate(net, x)
    np.testing.assert_allclose(raw, [net.activate(v)[0] for v in x], atol=1e-12)
    fast = fitness_accounting(d, gated_targets(raw[None, :], d.x), rebalance_band=.05)
    reference = evaluate({'kind': 'gate', 'spec': spec, 'genome': g}, [m])[m.symbol]
    for key in ['return', 'max_drawdown', 'cost_paid_initial_equity', 'funding_paid_initial_equity']:
        assert fast[key][0] == pytest.approx(reference[key], abs=2e-12)


def test_native_schedule_and_delay_shift_orders_as_well_as_targets():
    m = market()
    candidate = {'kind': 'alpha', 'frequency': '4h', 'long_short': True}
    assert schedule(candidate, m).all()
    p = alpha_targets(m.x, long_short=True)
    np.testing.assert_array_equal(desired(candidate, m), p)
    shifted = simulate(m, np.r_[0, p[:-1]], trade_allowed=np.r_[False, np.ones(len(p)-1, bool)], rebalance_band=.05)
    reference = evaluate(candidate, [m], delay=1)[m.symbol]
    np.testing.assert_array_equal(reference['equity'], shifted['equity'])


def test_all_groups_add_to_the_same_alpha_features():
    specs = specifications()
    assert len(specs) == 10
    for spec in specs[2:-1]:
        assert spec['features'][:4] == ALPHA_FEATURES
        assert len(set(spec['features'])) == len(spec['features'])
    assert specs[-1]['features'] == FEATURE_NAMES


def test_closed_bar_observation_held_until_next_daily_decision():
    m = market()
    altered = m.x.copy()
    altered[1:6, STATE] *= -1
    np.testing.assert_array_equal(desired({'kind': 'alpha'}, m)[:6],
                                  desired({'kind': 'alpha'}, replace(m, x=altered))[:6])
    assert daily_schedule(m)[6]


def test_external_source_stress_does_not_shift_appended_alpha(monkeypatch):
    from neattrade import research_v7
    b = bars()
    original = pd.DataFrame(np.arange(len(b)*26, dtype=float).reshape(len(b), 26), index=b.index,
                             columns=FEATURE_NAMES[:26])
    monkeypatch.setattr(research_v7, 'load_dataset', lambda *args: ({'TEST': b}, {'TEST': original}, {}))
    _, frame, late, _ = research_v7.dataset()
    pd.testing.assert_frame_equal(frame['TEST'][ALPHA_FEATURES], late['TEST'][ALPHA_FEATURES])
    pd.testing.assert_frame_equal(frame['TEST'].iloc[:, :15], late['TEST'].iloc[:, :15])
    pd.testing.assert_frame_equal(frame['TEST'].iloc[:, 15:26].shift(), late['TEST'].iloc[:, 15:26])
