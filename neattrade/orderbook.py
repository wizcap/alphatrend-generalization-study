"""Verified Binance aggregated depth and one causal, bounded book feature."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import time
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests

ROOT = Path('data/study009')
SYMBOLS = ['BTCUSDT', 'ETHUSDT']
START, END = '2022-12-31', '2026-08-31'
LOOKBACK_SECONDS = 4*3600
LATENCY_SECONDS = 60
MAX_HOLD_SECONDS = 120
MIN_COVERAGE = .95


def download_one(job):
    symbol, day, root = job
    name = f'{symbol}-bookDepth-{day}.zip'
    url = f'https://data.binance.vision/data/futures/um/daily/bookDepth/{symbol}/{name}'
    path = Path(root, symbol, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    checksum = path.with_suffix('.zip.CHECKSUM')
    for attempt in range(4):
        try:
            if checksum.exists():
                expected = checksum.read_text().split()[0]
            else:
                response = requests.get(url+'.CHECKSUM', timeout=40)
                if response.status_code == 404:
                    archive = requests.get(url, timeout=40)
                    if archive.status_code == 404:
                        return {'symbol': symbol, 'day': day, 'url': url, 'status': 'missing_404'}
                    archive.raise_for_status()
                    raise ValueError('Archive exists without its checksum')
                response.raise_for_status()
                expected = response.text.split()[0]
                if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
                    raise ValueError('Invalid published SHA256')
                checksum.write_text(response.text, encoding='utf-8')
            payload = path.read_bytes() if path.exists() else b''
            if sha256(payload).hexdigest() != expected:
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                payload = response.content
                if sha256(payload).hexdigest() != expected:
                    raise ValueError('Archive checksum mismatch')
                temporary = path.with_suffix('.tmp')
                temporary.write_bytes(payload)
                temporary.replace(path)
            return {'symbol': symbol, 'day': day, 'url': url, 'path': path.as_posix(),
                    'status': 'verified', 'sha256': expected, 'bytes': len(payload)}
        except (requests.RequestException, ValueError):
            if attempt == 3:
                raise
            time.sleep(1+attempt)


def download(root=ROOT, start=START, end=END):
    jobs = [(s, str(d.date()), str(root)) for d in pd.date_range(start, end) for s in SYMBOLS]
    manifest = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for entry in pool.map(download_one, jobs):
            manifest.append(entry)
            if len(manifest) % 100 == 0:
                print(f'Book archives: {len(manifest)}/{len(jobs)}', flush=True)
    Path(root, 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Book archives finished: {len(manifest)}, missing={sum(x["status"] != "verified" for x in manifest)}', flush=True)


def snapshots(frame, day, with_sides=False):
    """Pair -1% bid and +1% ask at the same exchange timestamp.

    Invalid/ambiguous snapshots explicitly break the carried state. Extra
    percentage bands are ignored, never added to these cumulative quantities.
    """
    required = ['timestamp', 'percentage', 'depth', 'notional']
    if list(frame.columns) != required:
        raise ValueError('Unexpected bookDepth schema')
    frame = frame.copy()
    frame['timestamp'] = pd.to_datetime(frame.timestamp, utc=True, errors='raise')
    start = pd.Timestamp(day, tz='UTC')
    if not ((frame.timestamp >= start) & (frame.timestamp < start+pd.Timedelta(days=1))).all():
        raise ValueError('Archive timestamp outside its stated day')
    times = pd.DatetimeIndex(frame.timestamp.unique()).sort_values()
    frame = frame.loc[frame.percentage.isin([-1, 1])]
    if not len(times):
        raise ValueError('No paired1percent bands in archive')
    exact_duplicates = int(frame.duplicated(required).sum())
    frame = frame.drop_duplicates(required)
    ambiguous = frame.duplicated(['timestamp', 'percentage'], keep=False)
    bad_times = frame.loc[ambiguous, 'timestamp'].unique()
    clean = frame.loc[~ambiguous]
    notionals = clean.pivot(index='timestamp', columns='percentage', values='notional').reindex(index=times, columns=[-1, 1])
    depths = clean.pivot(index='timestamp', columns='percentage', values='depth').reindex(index=times, columns=[-1, 1])
    valid = (np.isfinite(notionals).all(axis=1) & np.isfinite(depths).all(axis=1)
             & (notionals > 0).all(axis=1) & (depths > 0).all(axis=1) & ~times.isin(bad_times))
    # Both cumulative1% bands refer to the same mid. Their VWAPs must be
    # ordered and no farther apart than the two outer band boundaries.
    vwap=notionals/depths
    ratio=vwap[-1]/vwap[1]
    inconsistent=valid & ((ratio>1+1e-6)|(ratio<.99/1.01-1e-6))
    valid &= ~inconsistent
    values = ((notionals[-1]-notionals[1])/(notionals[-1]+notionals[1])).where(valid).to_numpy(float)
    seconds = times.as_unit('ns').asi8//1_000_000_000
    audit={'snapshots': len(times), 'invalid_snapshots': int((~valid).sum()),
        'exact_duplicate_rows_removed': exact_duplicates, 'ambiguous_snapshots': len(bad_times),
        'inconsistent_band_vwap':int(inconsistent.sum())}
    result=(seconds,values,audit)
    if with_sides:
        return (*result,np.column_stack([depths[-1],notionals[-1],depths[1],notionals[1]]))
    return result


def stale_sides(times, sides):
    """Causal quality heuristic: either side's quantity AND notional unchanged>120s."""
    invalid=np.zeros(len(times),bool)
    for a in [0,2]:
        changed=np.r_[True,np.any(sides[1:,a:a+2]!=sides[:-1,a:a+2],axis=1)]
        previous=np.maximum.accumulate(np.where(changed,np.arange(len(times)),0))
        invalid |= times-times[previous]>MAX_HOLD_SECONDS
    return invalid


def load_snapshots(symbol, root=ROOT):
    manifest = json.loads(Path(root, 'manifest.json').read_text())
    parts, audit = [], []
    for entry in manifest:
        if entry['symbol'] != symbol or entry['status'] != 'verified':
            continue
        payload = Path(entry['path']).read_bytes()
        if sha256(payload).hexdigest() != entry['sha256']:
            raise ValueError('Changed raw book archive')
        with ZipFile(BytesIO(payload)) as z:
            if len(z.namelist()) != 1:
                raise ValueError('Expected one CSV per archive')
            with z.open(z.namelist()[0]) as f:
                frame = pd.read_csv(f)
        t, value, row, sides = snapshots(frame, entry['day'],with_sides=True)
        parts.append((t, value, sides))
        audit.append({'day': entry['day'], **row})
    times = np.concatenate([p[0] for p in parts])
    values = np.concatenate([p[1] for p in parts])
    if np.any(np.diff(times) <= 0):
        raise ValueError('Book timestamps overlap or are unordered')
    sides=np.concatenate([p[2] for p in parts])
    stale=stale_sides(times,sides)
    values[stale]=np.nan
    day_labels=pd.to_datetime(times,unit='s',utc=True).strftime('%Y-%m-%d')
    counts=pd.Series(stale.astype(int),index=day_labels).groupby(level=0).sum()
    for row in audit:
        row['unchanged_side_over120s']=int(counts.get(row['day'],0))
    return times, values, audit


def book_feature(times, values, decisions, latency=LATENCY_SECONDS):
    """Time-weighted mean on [T-4h,T), using snapshots after assumed latency.

    A snapshot expires after120s; invalid updates terminate the earlier state.
    No interpolation, global scaling, or carrying liquidity across long gaps.
    """
    times, values = np.asarray(times, np.int64), np.asarray(values, float)
    decisions = pd.DatetimeIndex(decisions)
    if len(times) != len(values) or not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError('Expected aligned strictly increasing snapshots')
    if decisions.tz is None or str(decisions.tz) != 'UTC' or latency < 0:
        raise ValueError('UTC decisions and nonnegative latency required')
    valid = np.isfinite(values)
    if np.any(np.abs(values[valid]) > 1):
        raise ValueError('Imbalance outside[-1,1]')
    starts = times+latency
    ends = np.minimum(np.r_[starts[1:], starts[-1]+MAX_HOLD_SECONDS], starts+MAX_HOLD_SECONDS)
    duration = (ends-starts)*valid
    cumulative_time = np.r_[0., np.cumsum(duration)]
    cumulative_value = np.r_[0., np.cumsum(duration*np.nan_to_num(values))]

    def integral(points):
        ix = np.searchsorted(starts, points, side='right')-1
        safe = np.maximum(ix, 0)
        extra = np.maximum(0, np.minimum(points, ends[safe])-starts[safe])*valid[safe]*(ix >= 0)
        return (cumulative_time[safe]+extra,
                cumulative_value[safe]+extra*np.nan_to_num(values[safe]))

    end = decisions.as_unit('ns').asi8/1_000_000_000
    hi_time, hi_value = integral(end)
    lo_time, lo_value = integral(end-LOOKBACK_SECONDS)
    covered = hi_time-lo_time
    weighted = hi_value-lo_value
    latest = np.searchsorted(starts, end, side='left')-1
    safe = np.maximum(latest, 0)
    fresh = (latest >= 0) & valid[safe] & (end-starts[safe] <= MAX_HOLD_SECONDS)
    coverage = covered/LOOKBACK_SECONDS
    usable = (coverage >= MIN_COVERAGE-1e-12) & fresh
    scalar = np.divide(weighted, covered, out=np.zeros_like(weighted), where=covered > 0)
    scalar = np.clip(scalar, -1, 1)
    return pd.DataFrame({'book_imbalance': np.where(usable, scalar, 0.), 'book_valid': usable,
        'book_coverage': coverage, 'last_snapshot_age_seconds': np.where(latest >= 0, end-starts[safe], np.nan)}, index=decisions)


def build_features(root=ROOT):
    root = Path(root)
    index = pd.date_range('2023-01-01', '2026-09-01', freq='15min', inclusive='left', tz='UTC')
    metadata = {'definition': 'time weighted1percent notional imbalance over past4h',
        'latency_seconds': LATENCY_SECONDS, 'max_carry_seconds': MAX_HOLD_SECONDS,
        'minimum_coverage': MIN_COVERAGE, 'manifest_sha256': sha256((root/'manifest.json').read_bytes()).hexdigest(),
        'module_sha256': sha256(Path(__file__).read_bytes()).hexdigest(), 'symbols': {}}
    for symbol in SYMBOLS:
        t, value, audit = load_snapshots(symbol, root)
        feature = book_feature(t, value, index)
        path = root/f'{symbol}_features.csv'
        feature.to_csv(path)
        np.savez_compressed(root/f'{symbol}_snapshots.npz', times=t, values=value)
        metadata['symbols'][symbol] = {'snapshots': len(t), 'invalid': int((~np.isfinite(value)).sum()),
            'feature_sha256': sha256(path.read_bytes()).hexdigest(), 'usable_fraction': float(feature.book_valid.mean()),
            'days': audit, 'snapshot_sha256': sha256((root/f'{symbol}_snapshots.npz').read_bytes()).hexdigest()}
        print(f'{symbol}: {len(t):,} snapshots, usable={feature.book_valid.mean():.2%}', flush=True)
    (root/'feature_manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-only', action='store_true')
    args = parser.parse_args()
    if not args.features_only:
        download()
    build_features()
