from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from neattrade.market import Market
from neattrade.research import config
from neattrade.research_v3 import daily_schedule, development_gate, evaluate, target_from_output, training_markets


def test_discrete_policy_can_still_abstain_and_has_no_forced_activity():
    x = np.zeros((5, 10))
    x[:, 6] = .8
    p = target_from_output(np.array([-1, -.2, 0, .2, 1]), x)
    np.testing.assert_array_equal(p, [-.25, 0, 0, 0, .25])


def test_development_gate_rejects_cash_and_inconsistent_fold_luck():
    row = {'scenarios': {k: {'return': 0., 'max_drawdown': 0., 'sharpe': 0.} for k in ['base', 'double_cost', 'delay_4h']},
           'asset_compounded_returns': {'A': 0., 'B': 0.},
           'folds': [{'portfolio': {'return': 0.}} for _ in range(4)]}
    assert not all(development_gate(row).values())
    for k in row['scenarios']:
        row['scenarios'][k].update({'return': .4, 'sharpe': 1.})
    row['asset_compounded_returns'] = {'A': .4, 'B': .4}
    row['folds'][-1]['portfolio']['return'] = .4
    assert not development_gate(row)['three_positive_folds']


def test_no_intra_day_rebalance_and_delayed_entry():
    n = 12
    prices = np.linspace(100, 112, n+1)
    m = Market('TEST', pd.date_range('2024-01-01', periods=n+1, freq='4h', tz='UTC'),
               np.zeros((n, 10)), prices, prices, np.zeros(n+1), prices[:-1], prices[1:])
    assert np.flatnonzero(daily_schedule(m)).tolist() == [0, 6]
    r = evaluate({'kind': 'long'}, [m], config())['TEST']
    assert r['order_count'] == 3
    assert r['decision_long_fraction'] == 1
    delayed = evaluate({'kind': 'long'}, [m], config(), delay=1)['TEST']
    assert delayed['equity'][1] == 1


def test_training_embargo_and_future_data_isolation():
    index = pd.date_range('2022-01-01', '2024-07-01', freq='4h', tz='UTC')
    rng = np.random.default_rng(10)
    prices = 100*np.exp(np.cumsum(rng.normal(0, .005, len(index))))
    b = pd.DataFrame({'open': prices, 'close': prices, 'high': prices*1.01, 'low': prices*.99,
                      'volume': 100+rng.random(len(index)), 'buy_volume': 50., 'mark': prices, 'funding': 0.}, index=index)
    initial = training_markets({'TEST': b}, '2024-01-01')
    assert max(m.times[-1] for m in initial) == pd.Timestamp('2023-12-25', tz='UTC')
    b.loc['2023-12-26':, 'close'] *= 5
    altered = training_markets({'TEST': b}, '2024-01-01')
    for a, z in zip(initial, altered):
        np.testing.assert_array_equal(a.x, z.x)
        np.testing.assert_array_equal(a.prices, z.prices)
