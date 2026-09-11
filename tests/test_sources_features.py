from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import zipfile

import neat
import numpy as np
import pandas as pd
import pytest

from neattrade.batch import fitness_accounting, prepare
from neattrade.data import COLS, read_archive
from neattrade.features_v5 import (ALL_FEATURES, FEATURES, GROUPS, PACKS, build_features,
                                   pack_matrix, source_lag, with_features)
from neattrade.market import features, simulate
from neattrade.overlay import overlay_targets
from neattrade.research import batch_activate
from neattrade.research_v3 import daily_schedule
from neattrade.research_v5 import desired, model_config, simple_scale
from neattrade.sources_v5 import normalize


def inputs(n=540):
    rng = np.random.default_rng(607)
    index = pd.date_range('2022-01-01', periods=n, freq='4h', tz='UTC')
    c = 100*np.exp(np.cumsum(rng.normal(.0002, .009, n)))
    b = pd.DataFrame({'close': c, 'open': np.r_[c[0], c[:-1]], 'volume': rng.uniform(10, 30, n)}, index=index)
    b['high'] = b[['open', 'close']].max(axis=1)*1.005
    b['low'] = b[['open', 'close']].min(axis=1)*.995
    b['buy_volume'] = b.volume*rng.uniform(.3, .7, n)
    b['quote_volume'] = b.volume*c
    b['mark'] = b.open*1.0001
    events = np.arange(n)%2 == 0
    b['funding'] = np.where(events, .0001, 0.)
    spot = b.copy()
    spot['close'] *= np.exp(rng.normal(0, .0001, n))
    premium = pd.DataFrame({'close': rng.normal(0, .0002, n)}, index=index)
    peer = b.copy()
    peer['close'] *= np.exp(np.cumsum(rng.normal(0, .005, n)))
    return b, spot, premium, peer, events


def test_whole_feature_pipeline_prefix_invariant_and_core_unchanged():
    b, s, p, peer, events = inputs()
    full = build_features(b, s, p, peer, events)
    pd.testing.assert_frame_equal(full[FEATURES], features(b))
    for n in [182, 201, 299, 401, 500]:
        prefix = build_features(b.iloc[:n], s.iloc[:n], p.iloc[:n], peer.iloc[:n], events[:n])
        pd.testing.assert_frame_equal(full.iloc[:n], prefix)
    # Change the current and all future candles/events in every source.
    n = 300
    mutated = [v.copy() for v in [b, s, p, peer]]
    for v in mutated:
        v.iloc[n:] *= 1.2
    later = build_features(*mutated, events)
    pd.testing.assert_frame_equal(full.iloc[:n+1], later.iloc[:n+1])


def test_funding_zero_event_replaces_old_rate_and_current_event_is_not_input():
    b, s, p, peer, events = inputs()
    b.loc[b.index[300], 'funding'] = 0.
    f = build_features(b, s, p, peer, events)
    assert f.funding_last_bps.iloc[300] == 1.
    assert f.funding_last_bps.iloc[301] == 0.
    assert f.funding_last_bps.iloc[302] == 0.
    assert f.funding_last_bps.iloc[303] == 1.


def test_source_lag_does_not_delay_core_or_execution_prices():
    f = build_features(*inputs())
    lagged = source_lag(f)
    pd.testing.assert_frame_equal(lagged.iloc[:, :15], f.iloc[:, :15])
    pd.testing.assert_frame_equal(lagged.iloc[1:, 15:].reset_index(drop=True), f.iloc[:-1, 15:].reset_index(drop=True))
    with pytest.raises(ValueError):
        source_lag(f, -1)


def test_cross_join_missing_or_misaligned_timestamps_fail():
    b, s, p, peer, events = inputs()
    with pytest.raises(ValueError, match='match exactly'):
        build_features(b, s.iloc[1:], p, peer, events)


@pytest.mark.parametrize('pack', PACKS)
def test_pack_masks_preserve_core_and_only_enable_declared_inputs(pack):
    x = np.ones((5, 26))
    masked = pack_matrix(x, pack)
    np.testing.assert_array_equal(masked[:, :10], 1.)
    names = set(ALL_FEATURES if pack == 'combined' else FEATURES+GROUPS[pack])
    for i, name in enumerate(ALL_FEATURES):
        np.testing.assert_array_equal(masked[:, i], float(name in names))


def test_generalized_network_matches_reference_and_training_accounting():
    b, s, p, peer, events = inputs()
    f = build_features(b, s, p, peer, events)
    m = with_features('TEST', b, f, '2022-02-01', '2022-03-01')
    c = model_config(8)
    g = next(iter(neat.Population(c, seed=11).population.values()))
    net = neat.nn.FeedForwardNetwork.create(g, c)
    d = prepare(m)
    x = pack_matrix(d.x, 'combined')
    raw = batch_activate(net, x)
    np.testing.assert_allclose(raw, [net.activate(v)[0] for v in x], atol=1e-12)
    targets = overlay_targets(raw[None, :], d.x, 'plain')
    fast = fitness_accounting(d, targets, rebalance_band=.05)
    full = desired({'kind': 'gate', 'genome': g, 'pack': 'combined'}, m, c)
    reference = simulate(m, full, trade_allowed=daily_schedule(m), rebalance_band=.05)
    for key in ['return', 'max_drawdown', 'funding_paid_initial_equity', 'cost_paid_initial_equity']:
        assert fast[key][0] == pytest.approx(reference[key], abs=2e-12)
    changed = m.x.copy()
    changed[2:] *= -1
    future = desired({'kind': 'gate', 'genome': g, 'pack': 'combined'}, replace(m, x=changed), c)
    np.testing.assert_array_equal(full[:6], future[:6])


def test_simple_rule_scales_have_fixed_thresholds_and_clipped_extreme():
    x = np.zeros((3, 26))
    x[:, ALL_FEATURES.index('funding_last_bps')] = [0., 4., 0.]
    x[:, ALL_FEATURES.index('premium1d_bps')] = [0., 0., 5.]
    np.testing.assert_array_equal(simple_scale(x, 'derivatives'), [1., .5, .5])
    for group in ['technical', 'derivatives', 'cross', 'combined']:
        assert np.all((simple_scale(x, group) >= .5) & (simple_scale(x, group) <= 1.))


def test_spot_timestamp_unit_transition_and_invalid_close_rejected():
    stamps = pd.DatetimeIndex(['2024-12-31 20:00', '2025-01-01 00:00'], tz='UTC')
    ms = stamps.as_unit('ms').asi8
    frame = pd.DataFrame([[ms[0], 1, 2, .5, 1, 10, ms[0]+14_400_000-1, 10, 5, 5, 5, 0],
                          [ms[1]*1000, 1, 2, .5, 1, 10, (ms[1]+14_400_000)*1000-1, 10, 5, 5, 5, 0]], columns=COLS)
    out = normalize(frame)
    pd.testing.assert_index_equal(out.index, stamps.as_unit('ms').rename('time'))
    broken = frame.copy()
    broken.loc[0, 'close_time'] += 1000
    with pytest.raises(ValueError, match='close timestamp'):
        normalize(broken)
    with pytest.raises(ValueError, match='Duplicate'):
        normalize(pd.concat([frame, frame]))


def test_signed_premium_candles_are_valid_but_spot_must_be_positive():
    start = 1640995200000
    frame = pd.DataFrame([[start, -.001, 0., -.002, -.001, 0., start+14_400_000-1, 0., 0, 0., 0., 0]], columns=COLS)
    assert normalize(frame, positive=False).close.iloc[0] == -.001
    with pytest.raises(ValueError, match='Invalid source price'):
        normalize(frame, positive=True)


def test_archive_content_tampering_is_rejected(tmp_path):
    path = tmp_path/'example.zip'
    stream = BytesIO()
    with zipfile.ZipFile(stream, 'w') as z:
        z.writestr('example.csv', '1,1,1,1,1,1,1,1,1,1,1,1\n')
    path.write_bytes(stream.getvalue())
    path.with_suffix('.zip.CHECKSUM').write_text(sha256(path.read_bytes()).hexdigest()+' example.zip')
    assert len(read_archive(path)) == 1
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='checksum'):
        read_archive(path)
