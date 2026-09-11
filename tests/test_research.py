from dataclasses import replace
import random

import neat
import numpy as np
import pandas as pd
import pytest

from neattrade.market import Market, features, make_market, simulate
from neattrade.research import batch_activate, config, positions


def market(prices=(100, 100, 100), funding=(0, 0, 0), marks=None):
    a = np.array(prices, dtype=float)
    return Market('TEST', pd.date_range('2024-01-01', periods=len(a), freq='4h', tz='UTC'),
                  np.zeros((len(a)-1, 10)), a, a if marks is None else np.array(marks),
                  np.array(funding, dtype=float), np.minimum(a[:-1], a[1:]), np.maximum(a[:-1], a[1:]))


def test_cash_has_no_cost_or_funding():
    r = simulate(market(funding=(0.1, 0.1, 0.1)), [0, 0])
    assert r['return'] == 0
    assert r['turnover'] == 0


def test_entry_and_exit_costs():
    r = simulate(market(prices=(100, 100), funding=(0, 0)), [0.5], fee_bps=10, slippage_bps=0)
    assert r['return'] == pytest.approx(-0.001)
    assert r['order_count'] == 2


def test_round_trip_direction_and_compounding():
    m = market(prices=(100, 110, 121))
    assert simulate(m, [0.5, 0.5], 0, 0)['return'] == pytest.approx(1.05**2 - 1)
    assert simulate(m, [-0.5, -0.5], 0, 0)['return'] == pytest.approx(0.95**2 - 1)


def test_funding_sign_mark_notional_and_incumbent_timing():
    m = market(funding=(0.50, 0.01, 0), marks=(100, 120, 100))
    assert simulate(m, [0.5, 0], 0, 0)['return'] == pytest.approx(-0.006)
    assert simulate(m, [-0.5, 0], 0, 0)['return'] == pytest.approx(0.006)
    assert simulate(m, [0, 0.5], 0, 0)['return'] == 0


def test_flip_charges_both_legs():
    r = simulate(market(), [0.5, -0.5], fee_bps=10, slippage_bps=0)
    # Entry cost .0005; flip .00099975; exit .00049975.
    assert r['cost_paid_initial_equity'] == pytest.approx(0.0019995)
    assert r['return'] == pytest.approx(-0.0019995)


def test_equity_drawdown_includes_initial_capital():
    r = simulate(market(prices=(100, 90, 90)), [0.5, 0.5], 0, 0)
    assert r['max_drawdown'] == pytest.approx(0.05)


def test_position_and_insolvency_guards():
    with pytest.raises(ValueError):
        simulate(market(), [0.6, 0])
    with pytest.raises(ValueError):
        simulate(market(), [np.nan, 0])
    with pytest.raises(ValueError):
        simulate(market(), [0, 0], -1, 0)
    with pytest.raises(ValueError):
        simulate(market(prices=(100, 400, 400)), [-0.5, 0])


def bars():
    rng = np.random.default_rng(13)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    return pd.DataFrame({'open': c, 'close': c, 'high': c*1.01, 'low': c*.99,
                         'volume': rng.uniform(100, 200, 400), 'buy_volume': 60., 'mark': c, 'funding': 0.},
                        index=pd.date_range('2024-01-01', periods=400, freq='4h', tz='UTC'))


def test_features_cannot_see_current_or_future_bars():
    b = bars()
    expected = features(b)
    b.iloc[250:, :4] *= 10
    b.iloc[250:, 4:6] *= 20
    actual = features(b)
    pd.testing.assert_frame_equal(expected.iloc[:251], actual.iloc[:251])
    assert not expected.iloc[251:].equals(actual.iloc[251:])


def test_split_excludes_future_returns_and_retains_warmup():
    b = bars()
    m = make_market('TEST', b, '2024-02-10', '2024-02-20')
    assert m.times[0] == pd.Timestamp('2024-02-10', tz='UTC')
    assert m.times[-1] == pd.Timestamp('2024-02-20', tz='UTC')
    assert len(m.prices) == len(m.x) + 1


def test_batch_network_matches_neat_reference_after_mutation():
    c = config(10)
    random.seed(10)
    g = next(iter(neat.Population(c, seed=10).population.values()))
    x = np.random.default_rng(7).normal(size=(50, 10))
    for _ in range(15):
        g.mutate(c.genome_config)
        net = neat.nn.FeedForwardNetwork.create(g, c)
        expected = np.array([net.activate(row)[0] for row in x])
        np.testing.assert_allclose(batch_activate(net, x), expected, atol=1e-12)


def test_more_cost_does_not_improve_fixed_flat_market_strategy():
    m = market()
    assert simulate(m, [0.5, -0.5], 10, 4)['return'] < simulate(m, [0.5, -0.5], 5, 2)['return']


def test_neat_decisions_are_causal_and_daily():
    c = config(10)
    g = next(iter(neat.Population(c, seed=10).population.values()))
    b = bars()
    m = make_market('TEST', b, '2024-02-10', '2024-02-20')
    candidate = {'kind': 'neat', 'genome': g}
    p = positions(candidate, m, c)
    modified = m.x.copy()
    modified[2:] *= -10
    q = positions(candidate, replace(m, x=modified), c)
    np.testing.assert_array_equal(p[:6], q[:6])
    assert np.max(np.abs(p)) <= 0.5


def test_fixed_quantity_holding_without_maintenance_orders():
    m = market(prices=(100, 110, 121))
    r = simulate(m, [0.5, -0.5], 0, 0, trade_allowed=[True, False])
    assert r['return'] == pytest.approx(0.105)
    assert r['order_count'] == 2
    assert r['mean_abs_exposure'] == pytest.approx((0.5 + 0.55/1.05)/2)


def test_disallowed_entry_has_no_funding_or_cost():
    m = market(funding=(.01, .01, .01))
    assert simulate(m, [.5, .5], trade_allowed=[False, False])['return'] == 0
    with pytest.raises(ValueError):
        simulate(m, [.5, .5], trade_allowed=[True])


def test_weekly_schedule_and_delayed_execution():
    from neattrade.research_v2 import evaluate, schedule
    b = bars()
    m = make_market('TEST', b, '2024-02-10', '2024-02-20')
    a = schedule(m)
    expected = [0, 12, 54]  # Saturday initial open and the two following Mondays.
    assert np.flatnonzero(a).tolist() == expected
    candidate = {'kind': 'long'}
    direct = evaluate(candidate, [m], config())['TEST']
    delayed = evaluate(candidate, [m], config(), delay=1)['TEST']
    assert delayed['equity'][1] == 1
    assert direct['order_count'] <= len(expected) + 1
    assert delayed['order_count'] <= len(expected) + 1


def test_market_rejects_real_timestamp_gap():
    b = bars().drop(bars().index[270])
    with pytest.raises(ValueError, match='4h grid'):
        make_market('TEST', b, '2024-02-10', '2024-02-20')


def test_portfolio_rejects_misaligned_assets():
    from neattrade.research import evaluate
    m = market()
    other = replace(m, symbol='OTHER', times=m.times + pd.Timedelta(hours=4))
    with pytest.raises(ValueError, match='identical execution timestamps'):
        evaluate({'kind': 'cash'}, [m, other], config())


def test_batch_evaluation_rejects_wrong_input_count():
    c = config(6)
    g = next(iter(neat.Population(c, seed=10).population.values()))
    net = neat.nn.FeedForwardNetwork.create(g, c)
    with pytest.raises(ValueError, match='Feature count'):
        batch_activate(net, np.zeros((2, 9)))


@pytest.mark.parametrize('header', [True, False])
def test_archive_header_versions_and_checksum(tmp_path, header):
    from hashlib import sha256
    import zipfile
    from neattrade.data import COLS, read_archive
    p = tmp_path/'bars.zip'
    data = (','.join(COLS) + '\n' if header else '') + '1640995200000,100,110,90,105,100,1641009599999,10000,50,60,6000,0\n'
    with zipfile.ZipFile(p, 'w') as z:
        z.writestr('bars.csv', data)
    p.with_suffix('.zip.CHECKSUM').write_text(sha256(p.read_bytes()).hexdigest() + ' bars.zip')
    frame = read_archive(p)
    assert len(frame) == 1
    assert frame.open.iloc[0] == 100
    p.write_bytes(p.read_bytes() + b'corruption')
    with pytest.raises(ValueError, match='checksum'):
        read_archive(p)
