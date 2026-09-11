import numpy as np
import pandas as pd
import pytest

from neattrade.batch import fitness_accounting, growth_objective, legacy_objective, prepare
from neattrade.market import Market, simulate


def random_market(seed=42, n=137):
    rng = np.random.default_rng(seed)
    prices = 100*np.exp(np.r_[0., np.cumsum(rng.normal(0, .025, n))])
    marks = prices*rng.uniform(.99, 1.01, n+1)
    funding = rng.normal(.0001, .0004, n+1)
    funding[1::2] = 0
    return Market('RANDOM', pd.date_range('2024-01-01 04:00', periods=n+1, freq='4h', tz='UTC'),
                  rng.normal(size=(n, 10)), prices, marks, funding,
                  np.minimum(prices[:-1], prices[1:])*.97,
                  np.maximum(prices[:-1], prices[1:])*1.03)


@pytest.mark.parametrize('cost', [0, 7, 14])
def test_population_simulator_matches_reference(cost):
    m = random_market()
    d = prepare(m)
    p = np.random.default_rng(8).uniform(-.5, .5, size=(16, len(d.anchors)))
    p[0] = 0
    actual = fitness_accounting(d, p, cost)
    for i in range(len(p)):
        full = np.zeros(len(m.x))
        full[d.anchors] = p[i]
        allowed = np.zeros(len(m.x), dtype=bool)
        allowed[d.anchors] = True
        expected = simulate(m, full, fee_bps=cost, slippage_bps=0, trade_allowed=allowed)
        for key in ['return', 'max_drawdown', 'cost_paid_initial_equity', 'funding_paid_initial_equity']:
            assert actual[key][i] == pytest.approx(expected[key], abs=2e-12)
        assert not actual['ruined'][i]


def test_new_objective_does_not_reward_cash_or_churn():
    d = prepare(random_market())
    result = fitness_accounting(d, np.zeros((1, len(d.anchors))))
    assert growth_objective([result])[0] == 0
    assert legacy_objective([result])[0] == 0


def test_valid_growth_beats_cash_without_activity_bonus():
    result = {'return': np.array([0., .06, -.02]), 'years': .5,
              'max_drawdown': np.array([0., .12, .12]), 'ruined': np.zeros(3, bool)}
    new, old = growth_objective([result]), legacy_objective([result])
    assert old[1] < old[0]
    assert new[1] > new[0] > new[2]


def test_annual_growth_is_consistent_across_period_lengths():
    a = {'return': np.array([.1]), 'years': 1., 'max_drawdown': np.array([.1]), 'ruined': np.array([False])}
    b = dict(a, years=.5, **{'return': np.array([np.sqrt(1.1)-1])})
    np.testing.assert_allclose(growth_objective([a]), growth_objective([b]))


def test_population_ruin_is_rejected():
    m = random_market(n=10)
    m.highs[0] = 400
    d = prepare(m)
    result = fitness_accounting(d, np.full((1, len(d.anchors)), -.5))
    assert result['ruined'][0]
    assert growth_objective([result])[0] <= -100


def test_daily_anchors_follow_clock_and_partial_start():
    d = prepare(random_market(n=20))
    assert d.anchors.tolist() == [0, 5, 11, 17]
