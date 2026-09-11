from dataclasses import replace
import json

import neat
import numpy as np
import pandas as pd
import pytest

from neattrade.batch import fitness_accounting, prepare
from neattrade.features_v5 import ALL_FEATURES
from neattrade.indicator_types import TYPES, fixed_risk_scale, select_inputs, specifications
from neattrade.market import Market, simulate
from neattrade.overlay import overlay_targets
from neattrade.research import batch_activate
from neattrade.research_v3 import daily_schedule
from neattrade.research_v5 import model_config, simple_scale
from neattrade.research_v6 import desired, validate_resume, write_json


def market():
    rng = np.random.default_rng(606)
    n = 200
    p = 100*np.exp(np.r_[0., np.cumsum(rng.normal(0, .01, n))])
    x = rng.normal(0, 1, (n, 26))
    x[:, 6] = .6
    return Market('TEST', pd.date_range('2024-01-01', periods=n+1, freq='4h', tz='UTC'), x,
                  p, p*1.001, rng.normal(0, .0001, n+1), np.minimum(p[:-1], p[1:]), np.maximum(p[:-1], p[1:]))


def test_types_are_an_exact_partition_and_removals_are_complements():
    flat = [n for names in TYPES.values() for n in names]
    assert len(flat) == len(set(flat)) == len(ALL_FEATURES)
    assert set(flat) == set(ALL_FEATURES)
    specs = {s['name']: s for s in specifications()}
    assert len(specs) == 16
    for group, names in TYPES.items():
        only, without = specs['only_'+group], specs['without_'+group]
        assert set(only['features']) == set(names)
        assert not set(only['features']) & set(without['features'])
        assert set(only['features']) | set(without['features']) == set(ALL_FEATURES)


@pytest.mark.parametrize('name', ['all', 'intercept', 'only_funding', 'only_momentum', 'without_flow'])
def test_compact_inference_matches_neat_and_training_accounting(name):
    spec = next(s for s in specifications() if s['name'] == name)
    m = market()
    d = prepare(m)
    x = select_inputs(d.x, spec)
    c = model_config(8, inputs=x.shape[1])
    g = next(iter(neat.Population(c, seed=11).population.values()))
    net = neat.nn.FeedForwardNetwork.create(g, c)
    raw = batch_activate(net, x)
    np.testing.assert_allclose(raw, [net.activate(row)[0] for row in x], atol=1e-12)
    train = fitness_accounting(d, overlay_targets(raw[None, :], d.x, 'plain'), rebalance_band=.05)
    member = {'kind': 'gate', 'spec': spec, 'genome': g}
    reference = simulate(m, desired(member, m), trade_allowed=daily_schedule(m), rebalance_band=.05)
    for key in ['return', 'max_drawdown', 'cost_paid_initial_equity', 'funding_paid_initial_equity']:
        assert train[key][0] == pytest.approx(reference[key], abs=2e-12)
    changed = m.x.copy()
    changed[2:] *= -1
    np.testing.assert_array_equal(desired(member, m)[:6], desired(member, replace(m, x=changed))[:6])


def test_excluded_inputs_do_not_enter_gate_but_common_risk_inputs_remain():
    spec = next(s for s in specifications() if s['name'] == 'only_funding')
    m = market()
    mutated = m.x.copy()
    muted = [i for i, n in enumerate(ALL_FEATURES) if n not in spec['features'] and i not in [5, 6]]
    mutated[:, muted] += 10
    np.testing.assert_array_equal(select_inputs(m.x, spec), select_inputs(mutated, spec))
    c = model_config(8, inputs=2)
    g = next(iter(neat.Population(c, seed=29).population.values()))
    member = {'kind': 'gate', 'spec': spec, 'genome': g}
    np.testing.assert_array_equal(desired(member, m), desired(member, replace(m, x=mutated)))
    # With identical funding inputs, fixed trend eligibility still forces exits.
    mutated[:, 5] = -1
    assert not desired(member, replace(m, x=mutated)).any()


def test_rule_split_is_exact_union_of_previously_declared_conditions():
    x = np.zeros((4, 26))
    x[:, ALL_FEATURES.index('funding_last_bps')] = [0, 4, 0, 4]
    x[:, ALL_FEATURES.index('premium1d_bps')] = [0, 0, 5, 5]
    np.testing.assert_array_equal(fixed_risk_scale(x, 'funding'), [1, .5, 1, .5])
    np.testing.assert_array_equal(fixed_risk_scale(x, 'premium'), [1, 1, .5, .5])
    np.testing.assert_array_equal(fixed_risk_scale(x, 'both'), [1, .5, .5, .5])
    np.testing.assert_array_equal(fixed_risk_scale(x, 'both'), simple_scale(x, 'derivatives'))


def test_json_supports_numpy_result_scalars_without_hiding_unknown_values(tmp_path):
    path = tmp_path/'result.json'
    write_json(path, {'pass': np.bool_(True), 'count': np.int64(3), 'returns': np.array([.1, .2])})
    assert json.loads(path.read_text()) == {'pass': True, 'count': 3, 'returns': [.1, .2]}
    with pytest.raises(TypeError):
        write_json(path, {'invalid': object()})


def test_resume_rejects_changed_experiment_budget(tmp_path):
    from types import SimpleNamespace
    write_json(tmp_path/'protocol.json', {'population': 64, 'generations': 30, 'specifications': specifications()})
    with pytest.raises(ValueError, match='parameters differ'):
        validate_resume(tmp_path, SimpleNamespace(population=65, generations=30), specifications())
