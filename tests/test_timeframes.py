import numpy as np
import pandas as pd
import pytest

from neattrade.alphatrend import alpha_indicator
from neattrade.batch import DailyMarket, fitness_accounting
from neattrade.timeframes import (TIMEFRAMES, PER_YEAR, aggregate, signals, common_volatility,
    inputs, make_market, orders, evaluate, metrics, daily_returns)


def history(days=130):
    rng = np.random.default_rng(808)
    n = days*96
    p = 100*np.exp(np.cumsum(rng.normal(0, .003, n)))
    o = np.r_[p[0], p[:-1]]
    return pd.DataFrame({'open': o, 'high': np.maximum(o, p)+.1, 'low': np.minimum(o, p)-.1,
                         'close': p, 'volume': rng.uniform(1, 10, n), 'mark': o*1.001,
                         'funding': np.where(np.arange(n) % 32 == 0, .0001, 0.)},
                        index=pd.date_range('2022-01-01', periods=n, freq='15min', tz='UTC'))


def independent_segments(m, target, allowed, cost=7):
    anchors = np.unique(np.r_[0, np.flatnonzero(allowed)])
    # The first anchor before a delayed initial order must remain flat.
    p = target[anchors].copy()
    if not allowed[0]:
        p[0] = 0.
    segments = []
    for a, z in zip(anchors, np.r_[anchors[1:], len(target)]):
        cumulative = m.prices[a+1:z+1]-m.prices[a]-np.cumsum(m.marks[a+1:z+1]*m.funding[a+1:z+1])
        segments.append((int(a), int(z), cumulative))
    prepared = DailyMarket(m, anchors, m.x[anchors], segments)
    return fitness_accounting(prepared, p[None, :], cost_bps=cost, rebalance_band=.05)


def test_ohlcv_aggregation_uses_only_complete_bins_and_exact_values():
    b = history(2)
    a = aggregate(b, '1h')
    assert len(a) == 48
    first = b.iloc[:4]
    assert a.open.iloc[0] == first.open.iloc[0]
    assert a.close.iloc[0] == first.close.iloc[-1]
    assert a.high.iloc[0] == first.high.max()
    assert a.low.iloc[0] == first.low.min()
    assert a.volume.iloc[0] == pytest.approx(first.volume.sum())
    truncated = aggregate(b.iloc[1:-1], '1h')
    assert len(truncated) == 46
    with pytest.raises(ValueError, match='Incomplete source'):
        aggregate(b.drop(b.index[17]), '1h')


def test_weekly_bucket_is_monday_utc_with_no_partial_first_or_last_week():
    a = aggregate(history(130), '1w')
    assert (a.index.dayofweek == 0).all() and (a.index.hour == 0).all()
    assert str(a.index[0]) == '2022-01-03 00:00:00+00:00'
    b = history(130)
    for t in a.index:
        week = b.loc[(b.index >= t) & (b.index < t+pd.Timedelta(days=7))]
        assert len(week) == 672
        assert a.loc[t, 'volume'] == pytest.approx(week.volume.sum())


@pytest.mark.parametrize('tf', ['15m', '1h', '4h', '1d', '1w'])
def test_signal_available_at_close_not_inside_its_own_bucket(tf):
    b = history(130)
    aligned, availability, _ = signals(b, tf)
    native = aggregate(b, tf)
    state = alpha_indicator(native).state
    delta = pd.Timedelta(minutes=TIMEFRAMES[tf])
    for i in range(2, len(native)-1):
        end = native.index[i]+delta
        assert aligned.loc[end] == state.iloc[i]
        assert aligned.loc[end-pd.Timedelta(minutes=15)] == state.iloc[i-1]
    assert (availability == native.index+delta).all()


@pytest.mark.parametrize('tf', ['15m', '4h', '1d', '1w'])
def test_future_mutation_and_mid_bucket_truncation_are_causal(tf):
    b = history(130)
    full, _, _ = signals(b, tf)
    cut = 112*96+13
    truncated, _, _ = signals(b.iloc[:cut], tf)
    pd.testing.assert_series_equal(full.iloc[:cut], truncated)
    altered = b.copy()
    altered.iloc[cut:, :4] *= 3
    altered.iloc[cut:, 4] *= 20
    changed, _, _ = signals(altered, tf)
    pd.testing.assert_series_equal(full.iloc[:cut+1], changed.iloc[:cut+1])
    pd.testing.assert_series_equal(common_volatility(b).iloc[:cut+1], common_volatility(altered).iloc[:cut+1])


@pytest.mark.parametrize('tf,long_short,delay,cost', [('15m', True, 0, 7), ('1h', False, 1, 14),
    ('4h', True, 16, 7), ('1d', False, 96, 7), ('1w', True, 672, 7)])
def test_common_clock_funding_delays_and_segment_accounting(tf, long_short, delay, cost):
    b = history(150)
    f = inputs(b)
    m = make_market('TEST', b, f, '2022-05-02', '2022-05-20')
    target, allowed = orders(m, f, tf, long_short, delay)
    if delay:
        assert not allowed[:delay].any() and not target[:delay].any()
    reference = evaluate(m, f, tf, long_short, multiplier=cost/7, delay=delay)
    fast = independent_segments(m, target, allowed, cost)
    for field in ['return', 'max_drawdown', 'cost_paid_initial_equity', 'funding_paid_initial_equity']:
        assert fast[field][0] == pytest.approx(reference[field], abs=3e-12)


def test_sharpe_uses_15m_annualization_and_month_day_attribution():
    r = np.array([.01, -.005, .003, -.004])
    assert metrics(r)['sharpe'] == pytest.approx(r.mean()/r.std()*np.sqrt(PER_YEAR))
    times = pd.date_range('2024-01-01 23:45', periods=4, freq='15min', tz='UTC')
    daily = daily_returns(r, times)
    assert daily.iloc[0] == pytest.approx((1+r[0])*(1+r[1])-1)
    assert daily.iloc[1] == pytest.approx((1+r[2])*(1+r[3])-1)


def test_own_timeframe_only_controls_rebalance_times():
    b = history(150)
    f = inputs(b)
    m = make_market('TEST', b, f, '2022-05-02', '2022-05-20')
    for tf in TIMEFRAMES:
        p, a = orders(m, f, tf, True)
        assert a[0]
        np.testing.assert_array_equal(a[1:], m.times[1:-1].isin(f['closes'][tf]))
        assert np.all(np.diff(p)[~a[1:]] == 0.)


def test_passive_controls_ignore_alpha_direction_and_use_declared_sizing():
    b = history(150)
    f = inputs(b)
    m = make_market('TEST', b, f, '2022-05-02', '2022-05-20')
    f['states']['4h'].loc[:] = -1.
    vol, a = orders(m, f, '4h', False, passive='vol')
    half, _ = orders(m, f, '4h', False, passive='half')
    assert (vol > 0).all()
    np.testing.assert_array_equal(half, np.full(len(m.x), .5))
    np.testing.assert_allclose(vol[a], np.minimum(.5, .20/np.maximum(m.x[a, 6], .10)))
    assert not orders(m, f, '4h', False)[0].any()
