import json
from hashlib import sha256
import pickle

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('numba', reason='Study011 uses the pinned study010/011 runtime')

from neattrade import indicators_v11 as f
from neattrade import policies_v11 as p
from neattrade import research_v11 as r
from neattrade.market import simulate
from neattrade.fast_accounting import population_accounting, audit_simulate


def bars():
    rng = np.random.default_rng(51)
    ix = pd.date_range('2023-01-01', periods=5000, freq='15min', tz='UTC')
    op = 100*np.exp(np.cumsum(rng.normal(0, .001, len(ix))))
    close = op*np.exp(rng.normal(0, .0005, len(ix)))
    volume = rng.uniform(10, 20, len(ix))
    return pd.DataFrame({'open': op, 'high': np.maximum(op, close)*1.003,
        'low': np.minimum(op, close)*.997, 'close': close, 'volume': volume,
        'quote_volume': volume*close, 'buy_volume': volume*.55,
        'buy_quote_volume': volume*close*.55, 'trades': 5,
        'mark': op*1.0001, 'funding': np.where((ix.hour%8 == 0) & (ix.minute == 0), .0001, 0.)}, index=ix)


@pytest.fixture
def market():
    b = bars()
    return f.market('test', b, f.build_frame(b, '1h'), '2023-01-15', '2023-01-25')


@pytest.mark.parametrize('tf', f.FRAMES)
def test_prefix_and_future_candle_causality(tf):
    b = bars()
    full = f.build_frame(b, tf)
    boundary = pd.Timestamp('2023-02-01', tz='UTC')
    # Future/current candle, volume, funding and mark cannot alter today's input.
    changed = b.copy()
    changed.loc[boundary:, ['open', 'high', 'low', 'close', 'volume', 'buy_volume']] *= 1.3
    perturbed = f.build_frame(changed, tf)
    pd.testing.assert_frame_equal(full.loc[:boundary], perturbed.loc[:boundary])
    prefix = f.build_frame(b.loc[:boundary+pd.Timedelta(hours=4)-pd.Timedelta(minutes=15)], tf)
    pd.testing.assert_frame_equal(full.loc[:boundary], prefix.loc[:boundary])


@pytest.mark.parametrize('pack,count', [('alpha', 4), ('momentum', 8), ('volflow', 8), ('all', 12)])
def test_true_pack_dimensions_and_risk_excluded(pack, count):
    x = np.arange(3*13).reshape(3, 13).astype(float)
    selected = f.inputs(x, pack)
    assert selected.shape == (3, count)
    np.testing.assert_array_equal(selected, x[:, [f.FEATURES.index(n) for n in f.PACKS[pack]]])
    assert len(set(f.COLUMNS)) == 13
    assert all('book' not in name and 'valid' not in name for name in f.COLUMNS)


def test_missing_required_indicator_fails(market):
    b = bars()
    frame = f.build_frame(b, '1h')
    frame.loc['2023-01-20', 'rsi14'] = np.nan
    with pytest.raises(ValueError, match='Missing indicator'):
        f.market('test', b, frame, '2023-01-15', '2023-01-25')
    frame = frame.fillna(0)
    frame['book_valid'] = True
    with pytest.raises(ValueError, match='schema'):
        f.market('test', b, frame, '2023-01-15', '2023-01-25')


def test_no_alpha_direction_gate_or_fallback():
    x = np.zeros((5, 13))
    x[:, 0], x[:, -1] = -1., .8
    np.testing.assert_array_equal(f.alpha_target(x), np.zeros(5))
    np.testing.assert_allclose(f.learned_target(np.array([-1, -.35, 0, .35, 1]), x), [0, .125, .125, .125, .25])
    with pytest.raises(ValueError, match='Nonfinite'):
        f.learned_target(np.full(5, np.nan), x)


@pytest.mark.parametrize('kind', ['linear', 'mlp'])
@pytest.mark.parametrize('n', [4, 8, 12])
def test_policy_matches_scalar(kind, n):
    rng = np.random.default_rng(n)
    w = rng.normal(size=(3, p.dimensions(kind, n)[0]))
    x = rng.normal(size=(7, n))
    reference = []
    for weight in w:
        values = []
        for row in x:
            if kind == 'linear':
                val = sum(weight[i]*row[i] for i in range(n))+weight[-1]
            else:
                hidden = [np.tanh(sum(row[i]*weight[4*i+j] for i in range(n))+weight[4*n+j]) for j in range(4)]
                val = sum(hidden[j]*weight[4*n+4+j] for j in range(4))+weight[-1]
            values.append(np.tanh(val))
        reference.append(values)
    np.testing.assert_allclose(p.forward(w, x, kind), reference, atol=2e-14, rtol=0)
    with pytest.raises(ValueError, match='dimension'):
        p.forward(w[:, :-1], x, kind)


@pytest.mark.parametrize('n', [5, 9, 13, 25, 41, 57])
def test_cma_fixed_budget_and_reproducibility(n):
    evaluated = []
    def evaluate(w):
        evaluated.extend(w.copy())
        return -(w*w).sum(axis=1)
    a, fit_a, hist = p.fit(evaluate, n, 11, 6, 2)
    b, fit_b, _ = p.fit(evaluate, n, 11, 6, 2)
    np.testing.assert_array_equal(a, b)
    assert fit_a == fit_b and sum(h['candidate_evaluations'] for h in hist) == 12
    assert any(np.array_equal(a, v) for v in evaluated[:12])
    np.testing.assert_allclose(fit_a, max(-sum(v*v) for v in evaluated[:12]), atol=1e-12, rtol=0)


@pytest.mark.parametrize('tf', f.FRAMES)
def test_native_schedule_accounting_and_ensemble(market, tf):
    d = r.prepare(market, tf)
    signal = np.tile([-.6, 0, .6], int(np.ceil(len(d.x)/3)))[:len(d.x)]
    small = f.learned_target(signal, d.x)
    positions = np.repeat(small, np.diff(np.r_[d.anchors, len(market.x)]))
    scalar = simulate(market, positions, trade_allowed=f.allowed_times(market, tf), rebalance_band=.05)
    batch = population_accounting(d, small[None, :])
    audit = audit_simulate(market, positions, f.allowed_times(market, tf))
    for key in ['return', 'max_drawdown']:
        np.testing.assert_allclose(batch[key][0], scalar[key], atol=1e-12)
        np.testing.assert_allclose(audit[key], scalar[key], atol=1e-12)
    np.testing.assert_allclose(audit['equity'], scalar['equity'], atol=1e-12)
    ensemble, allowed = r.desired(['cash', 'alpha', 'passive_vol'], market, tf)
    expected = sum(r.desired(v, market, tf)[0] for v in ['cash', 'alpha', 'passive_vol'])/3
    np.testing.assert_allclose(ensemble, expected)
    assert allowed.sum() == len(d.x)


@pytest.mark.parametrize('kind', ['linear', 'mlp', 'neat'])
def test_job_checkpoint_budget_and_corruption(tmp_path, monkeypatch, market, kind):
    monkeypatch.setattr(r, 'OUT', tmp_path)
    monkeypatch.setattr(r, 'POPULATION', 6)
    monkeypatch.setattr(r, 'GENERATIONS', 2)
    job = ([market], '1h', kind, 'alpha', 11, 'small_'+kind, 'protocol')
    result = r.train_one(job)
    assert result['candidate_evaluations'] == 12 and result['input_count'] == 4
    assert r.train_one(job)['name'] == result['name']
    csv = tmp_path/'jobs'/('small_'+kind+'.csv')
    csv.write_text('corrupted')
    with pytest.raises(AssertionError):
        r.train_one(job)


def test_missing_inputs_do_not_open_depth_sources(monkeypatch):
    seen = []
    b = bars()
    # This synthetic loader is the only data dependency of the new dataset.
    def loader(symbol):
        seen.append(symbol)
        return b, []
    monkeypatch.setattr(f, 'load', loader)
    monkeypatch.setattr(f, 'TRAIN_START', '2023-01-15')
    _, frames, quality = f.dataset()
    assert seen == f.SYMBOLS
    assert all(list(frames[tf][s].columns) == f.COLUMNS for tf in f.FRAMES for s in f.SYMBOLS)


def test_contrast_count_and_cash_intervals():
    assert len(r.contrasts()) == 63
    z = [np.zeros(n) for n in [182, 184, 181, 184]]
    result = r.paired_interval(z, z)
    assert result['adjusted_intervals'] == {'7': [0., 0.], '28': [0., 0.]}


def test_independent_bootstrap_nonzero_sample():
    from neattrade.verify_v11 import reference_interval
    rng = np.random.default_rng(19)
    a = [rng.normal(.0001, .005, n) for n in [31, 34, 35, 37]]
    b = [rng.normal(0, .004, len(x)) for x in a]
    fast, direct = r.paired_interval(a, b)['adjusted_intervals'], reference_interval(a, b)
    for block in ['7', '28']:
        np.testing.assert_allclose(fast[block], direct[block], atol=1e-12, rtol=0)


@pytest.mark.parametrize('kind', ['linear', 'mlp', 'neat'])
def test_independent_target_replay(tmp_path, monkeypatch, market, kind):
    from neattrade.verify_v11 import reference_targets
    monkeypatch.setattr(r, 'OUT', tmp_path)
    monkeypatch.setattr(r, 'POPULATION', 6)
    monkeypatch.setattr(r, 'GENERATIONS', 2)
    member = r.train_one(([market], '1h', kind, 'all', 11, 'reference_'+kind, 'protocol'))
    for candidate in [member, [member, 'alpha', 'cash'], *r.CONTROLS]:
        np.testing.assert_allclose(r.desired(candidate, market, '1h')[0], reference_targets(candidate, market, '1h'), atol=1e-12, rtol=0)
