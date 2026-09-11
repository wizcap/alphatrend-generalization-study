"""Study011: native-candle indicators, with no depth input or availability gate."""
import numpy as np
import pandas as pd

from .alphatrend import ALPHA_FEATURES, alpha_features
from .market import Market
from .sources_v8 import load
from .timeframes import aggregate, common_volatility, TIMEFRAMES

SYMBOLS = ['BTCUSDT', 'ETHUSDT']
FRAMES = ['1h', '2h', '4h']
MOMENTUM = ['rsi14', 'macd_hist', 'ema12_26', 'ret6']
VOLFLOW = ['bollinger42', 'atr14_relative', 'volume_z42', 'taker_fraction']
FEATURES = list(ALPHA_FEATURES)+MOMENTUM+VOLFLOW
PACKS = {'alpha': list(ALPHA_FEATURES), 'momentum': list(ALPHA_FEATURES)+MOMENTUM,
         'volflow': list(ALPHA_FEATURES)+VOLFLOW, 'all': FEATURES.copy()}
COLUMNS = FEATURES+['risk_vol']
TRAIN_START = '2023-01-02'


def native_features(bars):
    c = bars.close
    vol = np.log(c).diff().rolling(42).std().clip(lower=.001)
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    relative = (up/(up+down).replace(0, np.nan)).where((up+down) != 0, .5)
    tr = pd.concat([bars.high-bars.low, (bars.high-c.shift()).abs(),
                    (bars.low-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    macd = c.ewm(span=12, adjust=False).mean()-c.ewm(span=26, adjust=False).mean()
    lv = np.log1p(bars.volume)
    f = pd.DataFrame({'rsi14': 2*relative-1,
        'macd_hist': (macd-macd.ewm(span=9, adjust=False).mean())/c/vol,
        'ema12_26': macd/c/vol,
        'ret6': np.log(c/c.shift(6))/(vol*np.sqrt(6)),
        'bollinger42': (c-c.rolling(42).mean())/(2*c.rolling(42).std().clip(lower=1e-12)),
        'atr14_relative': atr/c/vol,
        'volume_z42': (lv-lv.rolling(42).mean())/lv.rolling(42).std().clip(lower=.01),
        'taker_fraction': ((bars.buy_volume/bars.volume.replace(0, np.nan)-.5)*4).where(bars.volume != 0, 0.)}, index=bars.index)
    return pd.concat([alpha_features(bars), f.clip(-5, 5).shift(1)], axis=1)[FEATURES]


def build_frame(bars, tf):
    if tf not in FRAMES:
        raise ValueError('Unregistered timeframe')
    f = native_features(aggregate(bars, tf)).reindex(bars.index, method='ffill')
    f['risk_vol'] = common_volatility(bars)
    return f[COLUMNS]


def dataset():
    bars, frames, quality = {}, {tf: {} for tf in FRAMES}, {}
    for s in SYMBOLS:
        b, repairs = load(s)
        bars[s] = b
        quality[s] = {'rows': len(b), 'repairs': repairs, 'timeframes': {}}
        for tf in FRAMES:
            f = build_frame(b, tf)
            used = f.loc[TRAIN_START:'2026-08-31 20:00']
            if not np.isfinite(used.to_numpy()).all():
                raise ValueError('Missing indicator after fixed warmup; cannot drop observations')
            frames[tf][s] = f
            quality[s]['timeframes'][tf] = {'columns': COLUMNS, 'rows_after_warmup': len(used),
                'first_finite': str(f.dropna().index[0]), 'missing_after_warmup': 0,
                'clip_fraction': {n: float((used[n].abs() >= 5).mean()) for n in FEATURES}}
    return bars, frames, quality


def market(symbol, b, f, start, end):
    start, end = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')
    ix = b.index[(b.index >= start) & (b.index <= end)]
    if len(ix) < 2 or ix[0] != start or ix[-1] != end or np.any(np.diff(ix.as_unit('ns').asi8) != pd.Timedelta('15min').value):
        raise ValueError('Exact regular market endpoints required')
    if list(f.columns) != COLUMNS:
        raise ValueError('Indicator schema mismatch')
    x = f.reindex(ix[:-1]).to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError('Missing indicator within fixed market window')
    block = b.reindex(ix)
    return Market(symbol, ix, x, block.open.to_numpy(), block.mark.to_numpy(),
                  block.funding.to_numpy(), block.low.to_numpy()[:-1], block.high.to_numpy()[:-1])


def train_markets(bars, frames, test_start):
    end = pd.Timestamp(test_start)-pd.Timedelta(days=7)
    bounds = [pd.Timestamp(TRAIN_START)]+list(pd.date_range('2023-07-01', end, freq='6MS'))+[end]
    return [market(f'{s}_{a.date()}', b, frames[s], str(a.date()), str(z.date()))
            for s, b in bars.items() for a, z in zip(bounds[:-1], bounds[1:]) if z > a]


def inputs(x, pack):
    if x.ndim != 2 or x.shape[1] != len(COLUMNS) or pack not in PACKS:
        raise ValueError('Unknown pack or schema')
    return x[:, [FEATURES.index(n) for n in PACKS[pack]]]


def risk(x):
    return np.minimum(.5, .20/np.maximum(x[:, -1], .10))


def alpha_target(x):
    return np.maximum(x[:, 0], 0.)*risk(x)


def learned_target(raw, x):
    # Standalone long/flat policy: AlphaTrend state is an input, not an external gate.
    raw = np.asarray(raw)
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite policy output')
    return np.where(raw < -.35, 0., np.where(raw > .35, 1., .5))*risk(x)


def allowed_times(m, tf):
    ix = m.times[:-1]
    allowed = np.asarray((ix.hour*60+ix.minute) % TIMEFRAMES[tf] == 0)
    if not allowed[0]:
        raise ValueError('Market must begin at native decision boundary')
    return allowed
