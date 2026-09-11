"""Short-horizon features from the frozen study009 snapshot quality decisions."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .alphatrend import alpha_features, ALPHA_FEATURES
from .orderbook import ROOT as BOOK_ROOT, SYMBOLS
from .timeframes import aggregate, TIMEFRAMES

ROOT = Path('data/study010')
SHORT = ['15m', '30m', '1h']


def alpha_inputs(bars, timeframe):
    f = alpha_features(aggregate(bars, timeframe)).reindex(bars.index, method='ffill')
    if list(f.columns) != list(ALPHA_FEATURES):
        raise ValueError('Unexpected AlphaTrend columns')
    return f


def book_window(times, values, decisions, lookback_seconds, latency=60):
    if lookback_seconds not in [900, 1800, 3600, 14400] or latency < 0:
        raise ValueError('Unsupported book horizon or latency')
    times, values = np.asarray(times, np.int64), np.asarray(values, float)
    decisions = pd.DatetimeIndex(decisions)
    if len(times) != len(values) or not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError('Expected aligned ordered snapshots')
    if decisions.tz is None or str(decisions.tz) != 'UTC':
        raise ValueError('UTC decisions required')
    valid = np.isfinite(values)
    if np.any(np.abs(values[valid]) > 1):
        raise ValueError('Unbounded imbalance')
    starts = times+latency
    ends = np.minimum(np.r_[starts[1:], starts[-1]+120], starts+120)
    duration = (ends-starts)*valid
    weighted_values = np.nan_to_num(values)
    cumulative_time = np.r_[0., np.cumsum(duration)]
    cumulative_value = np.r_[0., np.cumsum(duration*weighted_values)]

    def integral(points):
        ix = np.searchsorted(starts, points, side='right')-1
        safe = np.maximum(ix, 0)
        extra = np.maximum(0, np.minimum(points, ends[safe])-starts[safe])*valid[safe]*(ix >= 0)
        return cumulative_time[safe]+extra, cumulative_value[safe]+extra*weighted_values[safe]

    end = decisions.as_unit('ns').asi8//1_000_000_000
    hi_t, hi_v = integral(end)
    lo_t, lo_v = integral(end-lookback_seconds)
    covered, weighted = hi_t-lo_t, hi_v-lo_v
    latest = np.searchsorted(starts, end, side='left')-1
    safe = np.maximum(latest, 0)
    fresh = (latest >= 0) & valid[safe] & (end-starts[safe] <= 120)
    coverage = covered/lookback_seconds
    usable = (coverage >= .95-1e-12) & fresh
    scalar = np.divide(weighted, covered, out=np.zeros_like(weighted), where=covered > 0)
    return pd.DataFrame({'book_imbalance': np.where(usable, np.clip(scalar, -1, 1), 0.),
        'book_valid': usable, 'book_coverage': coverage,
        'last_snapshot_age_seconds': np.where(latest >= 0, end-starts[safe], np.nan)}, index=decisions)


def build():
    source = BOOK_ROOT/'feature_manifest.json'
    meta = json.loads(source.read_text())
    frozen = json.loads(Path('artifacts/experiment_009/protocol.json').read_text())
    assert sha256(source.read_bytes()).hexdigest() == frozen['feature_manifest_sha256']
    ROOT.mkdir(parents=True, exist_ok=True)
    index = pd.date_range('2023-01-01', '2026-09-01', freq='15min', inclusive='left', tz='UTC')
    output = {'source_feature_manifest_sha256': sha256(source.read_bytes()).hexdigest(),
              'module_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
              'timeframes': SHORT, 'latency_seconds': 60, 'max_carry_seconds': 120,
              'minimum_coverage': .95, 'files': {}, 'quality': []}
    for s in SYMBOLS:
        path = BOOK_ROOT/f'{s}_snapshots.npz'
        assert sha256(path.read_bytes()).hexdigest() == meta['symbols'][s]['snapshot_sha256']
        with np.load(path) as d:
            t, v = d['times'], d['values']
        for tf in SHORT:
            f = book_window(t, v, index, TIMEFRAMES[tf]*60)
            path = ROOT/f'{s}_{tf}.csv'
            f.to_csv(path)
            output['files'][path.as_posix()] = sha256(path.read_bytes()).hexdigest()
            eligible = (index.hour*60+index.minute) % TIMEFRAMES[tf] == 0
            use = f.loc[eligible]
            for year in range(2023, 2027):
                for half in [1, 2]:
                    part = use.loc[(use.index.year == year) & ((use.index.month <= 6) if half == 1 else (use.index.month >= 7))]
                    vals = part.loc[part.book_valid, 'book_imbalance']
                    output['quality'].append({'symbol': s, 'timeframe': tf, 'year': year, 'half': half,
                        'decisions': len(part), 'usable_fraction': float(part.book_valid.mean()),
                        'p05': float(vals.quantile(.05)), 'median': float(vals.median()), 'p95': float(vals.quantile(.95))})
            print(f'{s} {tf}: features saved, valid={use.book_valid.mean():.2%}', flush=True)
    (ROOT/'feature_manifest.json').write_text(json.dumps(output, indent=2), encoding='utf-8')


if __name__ == '__main__':
    build()
