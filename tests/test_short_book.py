import numpy as np
import pandas as pd
import pytest

pytest.importorskip('numba', reason='Study010 uses its separately pinned runtime')

from neattrade.fast_accounting import population_accounting, audit_simulate
from neattrade.market import Market, simulate, InsolventSimulation
from neattrade.orderbook import book_feature
from neattrade.research_v10 import allowed_times, prepare, desired, paired_interval
from neattrade.research_v9 import targets, base_targets
from neattrade.short_book import book_window, alpha_inputs
from neattrade.timeframes import common_volatility


def fixture():
    ix = pd.date_range('2024-01-01', periods=385, freq='15min', tz='UTC')
    price = 100+np.sin(np.arange(385)/8)*4+np.arange(385)/40
    x = np.zeros((384, 7))
    x[:, 0] = 1
    x[:, 4] = np.sin(np.arange(384)/9)
    x[:, 5] = np.arange(384) % 13 != 0
    x[:, 6] = .55
    return Market('TEST', ix, x, price, price*1.03,
                  np.where(np.arange(385)%32 == 0, .007, 0.), price[:-1]-2, price[:-1]+2)


@pytest.mark.parametrize('tf,step', [('15m', 1), ('30m', 2), ('1h', 4), ('4h', 16)])
@pytest.mark.parametrize('cost,delay', [(0., 0), (7., 0), (14., 1)])
def test_compiled_accounting_matches_scalar_full_paths(tf, step, cost, delay):
    m = fixture()
    a = allowed_times(m, tf)
    np.testing.assert_array_equal(np.flatnonzero(a), np.arange(0, len(m.x), step))
    # Long, flat, short and resizing; funding and a terminal nonzero position.
    chosen = np.resize([.5, .5, .48, 0., -.5, -.24, .25], a.sum())
    p = np.repeat(chosen, step)[:len(m.x)]
    if delay:
        p = np.r_[0., p[:-1]]
        a = np.r_[False, a[:-1]]
    ref = simulate(m, p, cost, 0., trade_allowed=a, rebalance_band=.05)
    got = audit_simulate(m, p, a, cost)
    for key in ['return', 'max_drawdown', 'cost_paid_initial_equity', 'funding_paid_initial_equity',
                'turnover', 'mean_abs_exposure', 'equity', 'returns', 'order_count']:
        np.testing.assert_allclose(got[key], ref[key], atol=2e-12, rtol=0)
    if not delay:
        training = population_accounting(prepare(m, tf), chosen[None, :], cost)
        for key in ['return', 'max_drawdown', 'cost_paid_initial_equity', 'funding_paid_initial_equity']:
            assert training[key][0] == pytest.approx(ref[key], abs=2e-12)


def test_compiled_fitness_excludes_ruined_candidates_without_nan():
    m = fixture()
    m.lows[0] = -200
    d = prepare(m, '15m')
    p = np.full((2, len(d.anchors)), .5)
    p[1] = 0.
    got = population_accounting(d, p)
    assert got['ruined'].tolist() == [True, False]
    assert np.isfinite(got['return']).all() and got['return'][1] == 0
    with pytest.raises(InsolventSimulation):
        simulate(m, p[0])
    with pytest.raises(ValueError):
        population_accounting(d, np.full_like(p, .51))
    with pytest.raises(ValueError):
        population_accounting(d, p, cost_bps=-1)


@pytest.mark.parametrize('window', [900, 1800, 3600, 14400])
def test_short_book_window_is_causal_and_duration_weighted(window):
    start = pd.Timestamp('2024-01-01', tz='UTC').value//10**9
    t = np.ravel(np.column_stack([np.arange(start, start+28800, 60), np.arange(start+20, start+28800, 60)]))
    v = np.tile([1., 0.], len(t)//2)
    when = pd.to_datetime([start+21660], unit='s', utc=True)
    ref = book_window(t, v, when, window)
    assert ref.book_imbalance.iloc[0] == pytest.approx(1/3)
    assert ref.book_coverage.iloc[0] == 1 and ref.book_valid.iloc[0]
    changed = v.copy()
    changed[t >= start+21600] = -.9
    pd.testing.assert_frame_equal(ref, book_window(t, changed, when, window))
    mask = t < start+21600
    pd.testing.assert_frame_equal(ref, book_window(t[mask], v[mask], when, window))
    if window == 14400:
        pd.testing.assert_frame_equal(ref, book_feature(t, v, when))


def test_missing_latest_or_local_gap_breaks_short_book():
    t = np.arange(0, 7200, 30)+pd.Timestamp('2024-01-01', tz='UTC').value//10**9
    v = np.full(len(t), .4)
    end = t[0]+3600
    when = pd.to_datetime([end], unit='s', utc=True)
    v[t == end-90] = np.nan
    assert not book_window(t, v, when, 900).book_valid.iloc[0]
    keep = (t < end-700) | (t >= end-300)
    f = book_window(t[keep], np.full(keep.sum(), .4), when, 900)
    assert not f.book_valid.iloc[0] and f.book_coverage.iloc[0] < .95


@pytest.mark.parametrize('tf', ['15m', '30m', '1h'])
def test_short_alpha_and_common_volatility_exclude_unfinished_bars(tf):
    ix = pd.date_range('2024-01-01', periods=96*12, freq='15min', tz='UTC')
    p = 100+np.sin(np.arange(len(ix))/7)*3
    b = pd.DataFrame({'open': p, 'close': p+.2, 'high': p+1, 'low': p-1,
                      'volume': np.arange(len(ix))+100.}, index=ix)
    cut = ix[96*10+3]
    future = b.copy()
    future.loc[cut:, :] *= 3
    original = alpha_inputs(b, tf)
    assert original.loc[cut].notna().all()
    pd.testing.assert_frame_equal(original.loc[:cut], alpha_inputs(future, tf).loc[:cut])
    pd.testing.assert_series_equal(common_volatility(b).loc[:cut], common_volatility(future).loc[:cut])


@pytest.mark.parametrize('tf', ['15m', '30m', '1h'])
def test_shared_missing_fallback_and_book_input_can_change_short_targets(tf):
    m = fixture()
    weights = np.zeros(29)
    weights[16] = 5.
    weights[24] = 5.
    candidate = {'kind': 'cma', 'spec': 'cma_book', 'weights': weights}
    p, a = desired(candidate, m, tf)
    off, _ = desired(candidate, m, tf, True)
    assert np.any(np.abs(p[a]-off[a]) > .1)
    missing = a & ~m.x[:, 5].astype(bool)
    np.testing.assert_allclose(p[missing], base_targets(m.x[missing]))
    raw = np.stack([np.full(a.sum(), -1), np.full(a.sum(), 1)])
    gated = targets(raw, m.x[a])
    np.testing.assert_allclose(gated[:, ~m.x[a, 5].astype(bool)], base_targets(m.x[a])[~m.x[a, 5].astype(bool)][None, :].repeat(2, axis=0))


def test_zero_difference_bootstrap_is_zero_with_predeclared_family():
    result = paired_interval([np.zeros(3)], [np.zeros(3)])
    assert result['adjusted_intervals'] == {'7': [0., 0.], '28': [0., 0.]}
    assert result['intervals_corrected'] == 42


@pytest.mark.parametrize('spec', ['at_book', 'cma_book'])
def test_training_adapter_saves_fixed_budget_and_resumes_identical_model(tmp_path, monkeypatch, spec):
    import neattrade.research_v10 as r
    from neattrade.overlay import incremental_objective
    monkeypatch.setattr(r, 'OUT', tmp_path)
    monkeypatch.setattr(r, 'POPULATION', 8)
    monkeypatch.setattr(r, 'GENERATIONS', 3)
    m = fixture()
    job = ([m], '15m', spec, 11, 'fixture_'+spec, 'synthetic-test')
    model = r.train_one(job)
    again = r.train_one(job)
    assert again['training_seconds'] == model['training_seconds']
    trace = pd.read_csv(tmp_path/'jobs'/('fixture_'+spec+'.csv'))
    assert len(trace) == 3
    p, a = r.desired(model, m, '15m')
    ref = simulate(m, p, trade_allowed=a, rebalance_band=.05)
    baseline, _ = r.desired('alpha', m, '15m')
    b = simulate(m, baseline, trade_allowed=a, rebalance_band=.05)
    result = {'return': np.array([ref['return']]), 'max_drawdown': np.array([ref['max_drawdown']]),
              'ruined': np.array([False]), 'years': len(m.x)/(365.25*96)}
    connections = 24 if spec.startswith('cma') else len(model['genome'].connections)
    fitness = incremental_objective([result], [{'return': np.array([b['return']])}])[0]-.0001*connections
    stored = model['fitness'] if spec.startswith('cma') else model['genome'].fitness
    assert fitness == pytest.approx(stored, abs=2e-10)
    assert trace.candidate_evaluations.sum() == model['candidate_evaluations'] == 24


def test_reference_block_sums_reproduce_resampling_with_partial_last_block():
    from neattrade.verify_v10 import reference_interval
    rng = np.random.default_rng(10)
    a = [rng.normal(0, .002, 65), rng.normal(0, .003, 181)]
    b = [rng.normal(0, .002, len(x)) for x in a]
    expected = paired_interval(a, b)['adjusted_intervals']
    got = reference_interval(a, b)
    for block in ['7', '28']:
        np.testing.assert_allclose(got[block], expected[block], atol=2e-12, rtol=0)


def test_checkpoint_commit_marker_is_last_and_history_hash_checked(tmp_path, monkeypatch):
    from pathlib import Path
    import neattrade.research_v10 as r
    monkeypatch.setattr(r, 'OUT', tmp_path)
    monkeypatch.setattr(r, 'POPULATION', 8)
    monkeypatch.setattr(r, 'GENERATIONS', 1)
    job = ([fixture()], '15m', 'cma_book', 11, 'interrupted', 'fixture')
    original = Path.replace
    def fail_final(self, destination):
        if str(destination).endswith('interrupted.pkl'):
            raise OSError('Synthetic interruption before commit')
        return original(self, destination)
    monkeypatch.setattr(Path, 'replace', fail_final)
    with pytest.raises(OSError, match='Synthetic'):
        r.train_one(job)
    assert not (tmp_path/'jobs/interrupted.pkl').exists()
    assert (tmp_path/'jobs/interrupted.json').exists()
    monkeypatch.setattr(Path, 'replace', original)
    r.train_one(job)
    (tmp_path/'jobs/interrupted.csv').write_text('changed')
    with pytest.raises(AssertionError):
        r.train_one(job)
