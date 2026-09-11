"""Causal, grouped indicators with identical observation windows across packs."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load, read_archive
from .market import FEATURES, features, make_market
from .sources_v5 import load_extra

TECHNICAL = ['rsi14', 'macd_hist', 'adx14', 'bollinger42', 'atr14_relative']
DERIVATIVES = ['funding_last_bps', 'funding7d', 'premium1d_bps', 'premium_z', 'mark_basis_bps']
CROSS = ['spot_basis1d_bps', 'spot_basis_z', 'spot_buy1d', 'spot_relative_volume',
         'relative_momentum7d', 'peer_trend180']
ALL_FEATURES = FEATURES + TECHNICAL + DERIVATIVES + CROSS
PACKS = ['core', 'technical', 'derivatives', 'cross', 'combined']
GROUPS = {'core': FEATURES, 'technical': TECHNICAL, 'derivatives': DERIVATIVES, 'cross': CROSS}


def rolling_z(x, n=42, floor=1e-6):
    return (x - x.rolling(n).mean()) / x.rolling(n).std().clip(lower=floor)


def build_features(bars, spot, premium, peer, events):
    for source in [spot, premium, peer]:
        if not source.index.equals(bars.index):
            raise ValueError('All source timestamps must match exactly')
    events = np.asarray(events, bool)
    if events.shape != (len(bars),):
        raise ValueError('Funding event mask shape mismatch')
    c = bars.close
    vol = np.log(c).diff().rolling(42).std().clip(lower=.001)
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    relative_up = up / (up + down).replace(0, np.nan)
    relative_up = relative_up.where((up + down) != 0, .5)
    tr = pd.concat([bars.high - bars.low, (bars.high - c.shift()).abs(),
                    (bars.low - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    high_change, low_change = bars.high.diff(), -bars.low.diff()
    plus = high_change.where((high_change > low_change) & (high_change > 0), 0.)
    minus = low_change.where((low_change > high_change) & (low_change > 0), 0.)
    plus = plus.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr.clip(lower=1e-12)
    minus = minus.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr.clip(lower=1e-12)
    dx = (plus - minus).abs() / (plus + minus).clip(lower=1e-12)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f = pd.DataFrame(index=bars.index)
    f['rsi14'] = 2 * relative_up - 1
    f['macd_hist'] = (macd - macd.ewm(span=9, adjust=False).mean()) / c / vol
    f['adx14'] = 2 * dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean() - 1
    f['bollinger42'] = (c - c.rolling(42).mean()) / (2*c.rolling(42).std().clip(lower=1e-12))
    f['atr14_relative'] = atr / c / vol
    # A zero-rate settlement is still an event. Never infer events from nonzero rates.
    f['funding_last_bps'] = bars.funding.where(events).ffill(limit=2) * 10000
    f['funding7d'] = bars.funding.rolling(42).sum() * 1000
    f['premium1d_bps'] = premium.close.rolling(6).mean() * 10000
    f['premium_z'] = rolling_z(premium.close)
    f['mark_basis_bps'] = np.log(bars.mark / bars.open) * 10000
    basis = np.log(c / spot.close)
    f['spot_basis1d_bps'] = basis.rolling(6).mean() * 10000
    f['spot_basis_z'] = rolling_z(basis)
    volume = spot.volume.rolling(6).sum()
    f['spot_buy1d'] = (spot.buy_volume.rolling(6).sum() / volume.replace(0, np.nan) - .5) * 4
    f['spot_buy1d'] = f.spot_buy1d.where(volume != 0, 0.)
    f['spot_relative_volume'] = rolling_z(np.log1p(spot.quote_volume), floor=.01) - rolling_z(np.log1p(bars.quote_volume), floor=.01)
    f['relative_momentum7d'] = (np.log(c / c.shift(42)) - np.log(peer.close / peer.close.shift(42))) / (vol*np.sqrt(42))
    peer_vol = np.log(peer.close).diff().rolling(42).std().clip(lower=.001)
    f['peer_trend180'] = np.log(peer.close / peer.close.ewm(span=180, adjust=False).mean()) / (peer_vol*np.sqrt(180))
    # Every supplementary value is from at least the previous 4h observation.
    return pd.concat([features(bars), f.clip(-5, 5).shift(1)], axis=1)[ALL_FEATURES]


def pack_matrix(x, pack):
    if pack not in PACKS or x.shape[1] != len(ALL_FEATURES):
        raise ValueError('Unknown feature pack or shape')
    enabled = set(ALL_FEATURES if pack == 'combined' else FEATURES + GROUPS.get(pack, []))
    return np.where(np.array([name in enabled for name in ALL_FEATURES]), x, 0.)


def source_lag(x, bars=1):
    """Age external inputs only; keep the market, core/technical signals unchanged."""
    if bars < 0:
        raise ValueError('Negative source lag')
    y = x.copy()
    external = len(FEATURES) + len(TECHNICAL)
    y.iloc[:, external:] = x.iloc[:, external:].shift(bars)
    return y


def with_features(symbol, bars, frame, start, end):
    m = make_market(symbol, bars, start, end)
    x = frame.reindex(m.times[:-1]).to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError('Missing features inside common evaluation window')
    return replace(m, x=x)


def load_dataset(data='data', extra='data/study005', audit_path=None):
    bars = {s: load(s, data).loc[:'2026-08-31 20:00'] for s in ['BTCUSDT', 'ETHUSDT']}
    frames, audit = {}, {}
    for symbol, b in bars.items():
        spot, sr = load_extra(symbol, 'spot', b.index, extra)
        premium, pr = load_extra(symbol, 'premium', b.index, extra)
        raw = pd.concat([read_archive(p, funding=True) for p in sorted(Path(data, 'fundingRate').glob(f'{symbol}-*.zip'))])
        times = pd.to_datetime((raw.time / 14_400_000).round().astype('int64') * 14_400_000, unit='ms', utc=True)
        events = b.index.isin(times)
        peer = bars['ETHUSDT' if symbol == 'BTCUSDT' else 'BTCUSDT']
        frame = build_features(b, spot, premium, peer, events)
        frames[symbol] = frame
        valid = frame.dropna()
        start = features(b).dropna().index[0]
        if not np.isfinite(frame.loc[start:].to_numpy()).all():
            raise ValueError(f'Additional sources shrink the common sample: {symbol}')
        audit[symbol] = {'bars': len(b), 'first': str(b.index[0]), 'last': str(b.index[-1]),
                         'first_all_features_valid': str(valid.index[0]), 'feature_count': len(ALL_FEATURES),
                         'spot_daily_repairs': sr, 'premium_daily_repairs': pr,
                         'funding_events': int(events.sum()), 'source_missing_after_warmup': 0,
                         'spot_zero_volume_bars': int((spot.volume == 0).sum()),
                         'features': {n: {'min': float(valid[n].min()), 'max': float(valid[n].max()),
                                          'clip_fraction': float((valid[n].abs() >= 5).mean())} for n in ALL_FEATURES}}
    if audit_path:
        Path(audit_path).parent.mkdir(parents=True, exist_ok=True)
        Path(audit_path).write_text(json.dumps(audit, indent=2))
    return bars, frames, audit
