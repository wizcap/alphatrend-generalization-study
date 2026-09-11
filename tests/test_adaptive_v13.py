from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('sklearn', reason='Study013 uses the pinned supervised runtime')
from neattrade import adaptive_v13 as a
from neattrade.market import Market
from neattrade.repair_v12 import predict


def constant(date, value=-.1):
    return {'kind': 'mean', 'features': list(range(18)), 'mean': value, 'available': str(a.utc(date)),
            'training_ids': ['past'], 'training_predictions': np.asarray([value])}


def registry():
    return {a.version_id('4h', 'weighted', date): constant(date) for date in a.MONTHS}


def episodes(start='2024-01-02', count=20, harmful=False):
    result = []
    for i in range(count):
        date = a.utc(start)+pd.Timedelta(days=i)
        b = .03 if harmful or i%4 == 0 else -.02
        early = -.02 if harmful else max(b, 0.)
        result.append({'id': f'{date}_{i}', 'symbol': a.SYMBOLS[i%2], 'entry': str(date),
            'checkpoint': str(date+pd.Timedelta(days=1)), 'known': str(date+pd.Timedelta(days=2)),
            'left_censored': False, 'x': [0.]*18, 'base': {'base': b, 'double_cost': b-.001},
            'early': {'base': early, 'double_cost': early-.001}})
    return result


def training_rows():
    return [{'id': str(i), 'head': 'exit', 'known': str(a.utc('2023-02-01')+pd.Timedelta(days=i)),
             'decision': str(a.utc('2023-01-25')+pd.Timedelta(days=i)), 'left_censored': False,
             'x': (np.sin(np.arange(18)+i)+i/40).tolist(), 'y': float(np.cos(i)*.02)} for i in range(40)]


@pytest.mark.parametrize('method', ['expanding', 'weighted'])
def test_training_future_mutation_invariance(method):
    rows = training_rows()
    first = a.train_version(rows, '4h', method, '2024-01-01')
    future = deepcopy(rows[0])
    future.update(id='future', known='2024-01-01 00:00:00+00:00', y=1e8, x=[np.nan]*18)
    second = a.train_version(rows+[future], '4h', method, '2024-01-01')
    np.testing.assert_array_equal(first['training_predictions'], second['training_predictions'])
    assert first['training_ids'] == second['training_ids']


def test_weighted_scaler_and_weight_clock_are_training_only():
    rows = training_rows()
    m = a.train_version(rows, '4h', 'weighted', '2024-01-01')
    ages = np.asarray([(a.utc('2023-12-25')-a.utc(r['decision'])).days for r in rows])
    w = 2.**(-ages/180); w *= len(w)/sum(w)
    np.testing.assert_allclose(m['weights'], w, atol=1e-12)
    np.testing.assert_allclose(m['scaler'].mean_, np.average([r['x'] for r in rows], axis=0, weights=w), atol=1e-12)
    assert m['effective_weight_count'] <= len(rows)


@pytest.mark.parametrize('method', ['expanding', 'weighted'])
def test_training_rejects_insufficient_matured_rows(method):
    with pytest.raises(ValueError):
        a.train_version(training_rows()[:19], '4h', method, '2024-01-01')


def test_strict_maturity_and_censored_feedback():
    rows = episodes()
    rows[0]['known'] = str(a.utc('2024-02-01')-pd.Timedelta(days=7))
    rows[1]['known'], rows[2]['left_censored'] = None, True
    got = a.matured(rows, '2024-02-01', '2024-01-01')
    assert len(got) == 17 and all(r['id'] not in {v['id'] for v in rows[:3]} for r in got)


def test_promote_then_rollback_uses_only_post_activation_outcomes():
    c, models = a.Controller('4h', 'frozen'), registry()
    good = episodes()
    jan = c.step('2024-01-01', good, models)
    assert jan['actions'] == ['queue'] and jan['active'] is None
    feb = c.step('2024-02-01', good, models)
    assert feb['actions'] == ['promote', 'queue']
    assert feb['active'] == a.version_id('4h', 'weighted', '2024-01-01')
    mar = c.step('2024-03-01', good+episodes('2024-02-02', 12, True), models)
    assert mar['actions'] == ['rollback', 'queue'] and mar['active'] is None
    assert len(mar['health']['ids']) == 12


def test_rejection_and_winner_retention():
    c, models = a.Controller('4h', 'frozen'), registry()
    rows = episodes()
    for r in rows:
        r['early'] = {'base': -.05, 'double_cost': -.051}
    c.step('2024-01-01', rows, models)
    value = c.step('2024-02-01', rows, models)
    assert value['actions'] == ['reject', 'queue'] and not value['validation']['checks']['retention']


def test_insufficient_checkpoints_expire_without_changing_alpha():
    c, models = a.Controller('4h', 'frozen'), registry()
    for date in a.MONTHS[:7]:
        last = c.step(date, [], models)
    assert last['actions'] == ['expire', 'queue'] and last['active'] is None


def test_controller_prefix_future_payoff_invariance():
    one, two, models = a.Controller('4h', 'p'), a.Controller('4h', 'p'), registry()
    known = episodes()
    unknown = deepcopy(known[0]); unknown.update(id='future', known='2099-01-01', x=[np.nan]*18)
    for date in a.MONTHS[:2]:
        assert one.step(date, known, models) == two.step(date, known+[unknown], models)
    assert one.state == two.state


def test_atomic_resume_idempotency_and_changed_input(tmp_path):
    models, rows = registry(), episodes()
    c = a.Controller('4h', 'hash')
    first = c.step('2024-01-01', rows, models)
    path = tmp_path/'state.json'; c.save(path)
    recovered = a.Controller.load(path, '4h', 'hash')
    assert recovered.step('2024-01-01', rows, models) == first
    assert recovered.step('2024-02-01', rows, models) == c.step('2024-02-01', rows, models)
    changed = deepcopy(rows); changed[0]['base']['base'] += .01
    with pytest.raises(ValueError, match='Changed input'):
        recovered.step('2024-02-01', changed, models)
    with pytest.raises(ValueError, match='fingerprint'):
        a.Controller.load(path, '4h', 'other')
    payload = json.loads(path.read_text()); payload['state']['active'] = 'fake'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='Corrupt'):
        a.Controller.load(path, '4h', 'hash')


@pytest.mark.parametrize('date', ['2023-12-01', '2024-03-01', '2024-02-02'])
def test_out_of_order_skipped_or_nonmonthly_updates_rejected(date):
    c = a.Controller('4h', 'x'); models = registry()
    c.step('2024-01-01', [], models)
    with pytest.raises(ValueError):
        c.step(date, [], models)


def test_shadow_model_cannot_validate_before_creation():
    with pytest.raises(ValueError):
        a.screen(episodes(), constant('2024-02-01'))


@pytest.mark.parametrize('delay', [1, 2])
def test_model_version_is_latched_across_month_change(delay):
    n = 640
    ix = pd.date_range('2024-01-30', periods=n+1, freq='15min', tz='UTC')
    x = np.zeros((n, 13)); x[:, 0] = -1; x[:, -1] = .8
    x[160:304, 0] = 1; x[384:576, 0] = 1
    prices = np.full(n+1, 100.)
    ctx = {'market': Market('BTCUSDT', ix, x, prices, prices.copy(), np.zeros(n+1), prices[:-1], prices[:-1]),
           'atr': np.ones(n), 'closes': prices[:-1]}
    jan, feb = a.version_id('4h', 'weighted', '2024-01-01'), a.version_id('4h', 'weighted', '2024-02-01')
    models = {jan: constant('2024-01-01', .1), feb: constant('2024-02-01', -.1)}
    p, allowed, events = a.policy(ctx, '4h', 'weighted', models, {}, delay)
    exits = [e for e in events if e['head'] == 'exit']
    assert exits[0]['version'] == jan and not exits[0]['blocked']
    assert exits[1]['version'] == feb and exits[1]['blocked']
    assert p[256+delay] > 0 and p[480+delay] == 0
    assert np.all(p[480+delay:576+delay] == 0)
    assert allowed[480+delay] and not np.any(p[:160+delay])


def test_semiannual_schedule_keeps_2026_january_version():
    assert a.selected_version('4h', 'semiannual', '2026-08-16', {}) == '4h_expanding_202601'
    assert a.selected_version('4h', 'frozen', '2026-08-16', {}) == '4h_expanding_202401'


def test_failed_transition_leaves_state_unchanged():
    c, models, rows = a.Controller('4h', 'p'), registry(), episodes()
    c.step('2024-01-01', rows, models)
    before = deepcopy(c.state)
    bad = deepcopy(rows); bad[0]['x'] = [0.]*17
    with pytest.raises(ValueError):
        c.step('2024-02-01', bad, models)
    assert c.state == before


def test_idempotency_tracks_old_candidate_and_ignores_record_order():
    c, models, rows = a.Controller('4h', 'p'), registry(), episodes()
    c.step('2024-01-01', rows, models)
    value = c.step('2024-02-01', rows, models)
    assert c.step('2024-02-01', list(reversed(rows)), models) == value
    models['4h_weighted_202401']['mean'] += .01
    with pytest.raises(ValueError, match='Changed input'):
        c.step('2024-02-01', rows, models)


def test_model_fingerprint_is_stable_across_pickle_and_prediction():
    import pickle
    rows = training_rows()
    weighted = a.train_version(rows, '4h', 'weighted', '2024-01-01')
    expanding = a.train_version(rows, '4h', 'expanding', '2024-01-01')
    assert weighted['training_ids'] == expanding['training_ids']
    assert weighted['estimator'].get_params() == expanding['estimator'].get_params()
    fingerprint = a.model_digest(weighted)
    predict(weighted, [r['x'] for r in rows])
    assert fingerprint == a.model_digest(pickle.loads(pickle.dumps(weighted)))


def test_registered_date_and_timeframe_guards():
    with pytest.raises(ValueError, match='evaluation dates'):
        a.selected_version('4h', 'monthly', '2026-09-01', {})
    rows = training_rows(); rows[0]['timeframe'] = '2h'
    with pytest.raises(ValueError, match='timeframe'):
        a.train_version(rows, '4h', 'weighted', '2024-01-01')


def test_independent_controller_transitions_match_synthetic_branches():
    from neattrade.verify_v13 import reference_controller, equal
    models = registry(); rows = episodes()+episodes('2024-02-02', 12, True)
    c = a.Controller('4h', 'p')
    for date in a.MONTHS:
        c.step(date, rows, models)
    equal(c.state['history'], reference_controller(rows, models, '4h'))


def test_bootstrap_direct_and_prefix_implementations_agree():
    from neattrade.research_v13 import interval
    x = [np.linspace(-.01, .02, 31), np.sin(np.arange(37))*.003]
    y = [np.full(31, .0001), np.zeros(37)]
    from neattrade.verify_v13 import equal
    equal(interval(x, y), interval(x, y, direct=True), 1e-12)
