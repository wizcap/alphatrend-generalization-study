from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('sklearn', reason='Study012 supervised models have their own pinned environment')

from neattrade.market import Market, simulate
from neattrade import repair_v12 as r
from neattrade.timeframes import metrics


def fixture_context(n=480):
    ix = pd.date_range('2023-01-01', periods=n+1, freq='15min', tz='UTC')
    prices = 100+2*np.sin(np.arange(n+1)/21)+np.arange(n+1)*.003
    x = np.zeros((n, 13))
    x[:, 0], x[:, -1] = -1., .8
    x[:16, 0], x[32:192, 0], x[240:, 0] = 1., 1., 1.
    funding = np.where((ix.hour%8 == 0) & (ix.minute == 0), .0003, 0.)
    m = Market('test', ix, x, prices, prices*1.001, funding, prices[:-1]-.4, prices[:-1]+.4)
    return {'market': m, 'atr': np.ones(n), 'closes': prices[1:]-.001}


def constant(head, value):
    return {'kind': 'mean', 'head': head, 'features': r.ENTRY_FEATURES if head == 'entry' else r.EXIT_FEATURES, 'mean': value}


@pytest.mark.parametrize('tf', r.FRAMES)
def test_labels_match_scalar_branches_and_censoring(tf):
    ctx = fixture_context()
    m = ctx['market']
    labels, diag = r.labelled_events(ctx, tf)
    ep, _ = r.signals(ctx, tf)
    p, allowed, _ = r.policy(ctx, tf, 'alpha')
    assert len(ep) == 3 and len(labels) == 5
    assert all(v['y'] is None and v['known'] is None for v in labels if pd.Timestamp(v['decision']) >= m.times[240])
    assert labels[0]['left_censored']
    target = [v for v in labels if not v['left_censored'] and v['known'] is not None]
    a, z, c = 33, 193, 129
    whole = simulate(r.cut(m, a, z), p[a:z], trade_allowed=allowed[a:z], rebalance_band=.05)
    early = simulate(r.cut(m, a, c), p[a:c], trade_allowed=allowed[a:c], rebalance_band=.05)
    assert target[0]['head'] == 'entry' and target[1]['head'] == 'exit'
    np.testing.assert_allclose(target[0]['y'], whole['return'], atol=1e-12)
    np.testing.assert_allclose(target[1]['y'], whole['return']-early['return'], atol=1e-12)
    assert target[1]['known'] == str(m.times[z])


def test_checkpoint_current_candle_and_future_are_not_features():
    ctx = fixture_context()
    original = r.exit_features(ctx, 32, 128, 1)
    changed = deepcopy(ctx)
    for key in ['prices', 'marks', 'lows', 'highs', 'funding']:
        getattr(changed['market'], key)[128:] *= 9
    changed['closes'][128:] *= 9
    changed['atr'][128:] *= 9
    changed['market'].x[129:] *= 9
    np.testing.assert_array_equal(original, r.exit_features(changed, 32, 128, 1))
    shortened = {'market': r.cut(ctx['market'], 0, 129), 'atr': ctx['atr'][:129], 'closes': ctx['closes'][:129]}
    before = [v for v in r.signals(ctx, '2h')[1] if v['decision'] <= 128]
    after = r.signals(shortened, '2h')[1]
    assert len(before) == len(after)
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a['x'], b['x'])
        assert a['decision'] == b['decision'] and a['head'] == b['head']


def test_same_timestamp_funding_is_excluded():
    ctx = fixture_context()
    ctx['market'].funding *= .1
    x = r.exit_features(ctx, 32, 128, 1)
    ctx['market'].funding[128] += .01
    np.testing.assert_array_equal(x, r.exit_features(ctx, 32, 128, 1))
    ctx['market'].funding[127] += .0001
    assert r.exit_features(ctx, 32, 128, 1)[-1] == pytest.approx(x[-1]+.1)


@pytest.mark.parametrize('mode', r.MODES)
@pytest.mark.parametrize('delay', [1, 2])
def test_suppression_priority_and_latency(mode, delay):
    ctx = fixture_context()
    candidate = {'mode': mode, 'entry': constant('entry', -1), 'exit': constant('exit', -1)}
    p, allowed, actions = r.policy(ctx, '2h', candidate, delay)
    base, _, _ = r.policy(ctx, '2h', 'alpha', delay)
    assert np.all(p <= base) and np.all(p >= 0)
    assert np.all(p[:delay] == 0)
    if mode in ['entry', 'both']:
        np.testing.assert_array_equal(p[:16+delay], base[:16+delay])
        assert not p[16+delay:].any()
    else:
        assert p[32+delay] > 0 and p[128+delay] == 0
        assert np.all(p[128+delay:192+delay] == 0)
        assert p[240+delay] > 0  # Only a new Alpha episode can re-enter.
    # The short first Alpha episode exits before the24h check; no exit event.
    assert not any(a['episode'] == 0 and a['head'] == 'exit' for a in actions)
    assert all((pd.Timestamp(a['execution'])-pd.Timestamp(a['time'])).total_seconds() == delay*900 for a in actions)


@pytest.mark.parametrize('tf', r.FRAMES)
def test_zero_predictions_reproduce_alpha(tf):
    ctx = fixture_context()
    candidate = {'mode': 'both', 'entry': constant('entry', 0), 'exit': constant('exit', 0)}
    np.testing.assert_array_equal(r.policy(ctx, tf, candidate)[0], r.policy(ctx, tf, 'alpha')[0])


def test_execution_boundary_is_explicit():
    with pytest.raises(ValueError, match='lacks execution'):
        r.policy(fixture_context(481), '2h', 'alpha', 1)


@pytest.mark.parametrize('delay', [1, 2])
@pytest.mark.parametrize('cost', [7, 14])
def test_exact_episode_decomposition(delay, cost):
    ctx = fixture_context()
    m = ctx['market']
    def evaluate(c):
        p, a, _ = r.policy(ctx, '2h', c, delay)
        return simulate(m, p, cost, 0, trade_allowed=a, rebalance_band=.05)
    baseline = evaluate('alpha')
    for candidate in ['cash', 'alpha_half', 'exit24h', 'loss1atr24h']:
        result = evaluate(candidate)
        records = r.attribution(ctx, '2h', baseline, result, delay)
        assert len(records) == 3
        np.testing.assert_allclose(sum(v['delta_log'] for v in records), np.log1p(result['return'])-np.log1p(baseline['return']), atol=1e-12)


def training_rows(head):
    n = len(r.ENTRY_FEATURES if head == 'entry' else r.EXIT_FEATURES)
    rng = np.random.default_rng(12)
    x = rng.normal(size=(80, n))
    y = .01*np.sin(x[:, 0])-.02*x[:, 2]
    return [{'id': str(i), 'head': head, 'x': v.tolist(), 'y': float(y[i]),
             'known': '2023-06-01 00:00:00+00:00', 'decision': '2023-05-01 00:00:00+00:00',
             'left_censored': False} for i, v in enumerate(x)]


@pytest.mark.parametrize('head', ['entry', 'exit'])
@pytest.mark.parametrize('kind', r.KINDS)
def test_cutoff_maturity_and_fit_invariance(head, kind):
    rows = training_rows(head)
    future = deepcopy(rows[:3])
    future[0].update({'id': 'future', 'known': '2024-01-01 00:00:00+00:00', 'y': 100})
    future[1].update({'id': 'censored', 'known': None, 'y': None})
    future[2].update({'id': 'left', 'left_censored': True, 'y': 100})
    a = r.fit_model(rows, head, kind, '2024-01-01')
    b = r.fit_model(rows+future, head, kind, '2024-01-01')
    assert a['training_count'] == b['training_count'] == 80
    np.testing.assert_array_equal(a['training_predictions'], b['training_predictions'])
    with pytest.raises(ValueError, match='Insufficient'):
        r.fit_model(future, head, kind, '2024-01-01')


@pytest.mark.parametrize('head', ['entry', 'exit'])
@pytest.mark.parametrize('kind', r.KINDS)
def test_independent_predictions(head, kind):
    from neattrade.verify_v12 import reference_predict
    rows = training_rows(head)
    model = r.fit_model(rows, head, kind, '2024-01-01')
    x = np.stack([v['x'] for v in rows])
    reference_predict(model, x)
    reference_predict(model, x*1.2-.1)


@pytest.mark.parametrize('delay', [1, 2])
def test_independent_policy_and_full_account_label_replay(delay):
    from neattrade.verify_v12 import reference_policy, reference_labels, compare_labels
    ctx = fixture_context()
    candidates = r.CONTROLS+[{'mode': mode, 'entry': constant('entry', value), 'exit': constant('exit', -value)}
                             for mode in r.MODES for value in [-1, 0, 1]]
    for c in candidates:
        p, a, actions = r.policy(ctx, '2h', c, delay)
        rp, ra, reference_actions = reference_policy(ctx, '2h', c, delay)
        np.testing.assert_array_equal(p, rp)
        np.testing.assert_array_equal(a, ra)
        assert actions == reference_actions
    compare_labels(r.labelled_events(ctx, '2h')[0], reference_labels(ctx, '2h'))


def test_independent_bootstrap_nonzero():
    from neattrade.research_v12 import paired_interval, contrasts
    from neattrade.verify_v12 import direct_intervals
    rng = np.random.default_rng(24)
    a = [rng.normal(0, .002, n) for n in [31, 35, 37, 42]]
    b = [rng.normal(0, .002, len(p)) for p in a]
    fast, slow = paired_interval(a, b)['intervals'], direct_intervals(a, b)
    assert len(contrasts()) == 32
    for block in ['7', '28']:
        np.testing.assert_allclose(fast[block], slow[block], atol=1e-12, rtol=0)
