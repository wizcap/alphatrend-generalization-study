"""Additional official archives for study005; separate cache and manifest."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

from .data import read_archive

ROOT = 'https://data.binance.vision/data'
SOURCES = {'spot': 'spot', 'premium': 'futures/um'}


def fetch(job):
    symbol, source, period, root = job
    kind = 'klines' if source == 'spot' else 'premiumIndexKlines'
    cadence = 'daily' if len(period) == 10 else 'monthly'
    filename = f'{symbol}-4h-{period}.zip'
    url = f'{ROOT}/{SOURCES[source]}/{cadence}/{kind}/{symbol}/4h/{filename}'
    path = Path(root, source, cadence, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    check = path.with_suffix('.zip.CHECKSUM')
    for attempt in range(4):
        try:
            if not check.exists():
                r = requests.get(url + '.CHECKSUM', timeout=30)
                r.raise_for_status()
                check.write_text(r.text)
            expected = check.read_text().split()[0]
            payload = path.read_bytes() if path.exists() else b''
            if sha256(payload).hexdigest() != expected:
                r = requests.get(url, timeout=45)
                r.raise_for_status()
                payload = r.content
                if sha256(payload).hexdigest() != expected:
                    raise ValueError(f'Checksum mismatch: {url}')
                path.write_bytes(payload)
            return {'symbol': symbol, 'source': source, 'url': url, 'path': str(path),
                    'sha256': expected, 'bytes': len(payload)}
        except (requests.RequestException, ValueError):
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def normalize(frame, positive=True):
    """Spot switches from milliseconds to microseconds in January 2025."""
    frame = frame.copy()
    for col in ['time', 'close_time']:
        micro = frame[col] >= 10**14
        frame[col] = pd.to_datetime(frame[col].where(~micro, frame[col] // 1000), unit='ms', utc=True)
    if frame.time.duplicated().any():
        raise ValueError('Duplicate source timestamp')
    frame = frame.set_index('time').sort_index()
    if not np.all(frame.index.as_unit('ns').asi8 % pd.Timedelta(hours=4).value == 0):
        raise ValueError('Source off 4h grid')
    if not np.all(frame.close_time == frame.index + pd.Timedelta(hours=4) - pd.Timedelta(milliseconds=1)):
        raise ValueError('Unexpected candle close timestamp')
    prices = frame[['open', 'high', 'low', 'close']]
    if not np.isfinite(prices).all().all() or (positive and (prices <= 0).any().any()):
        raise ValueError('Invalid source price')
    if ((frame.high < frame[['open', 'close', 'low']].max(axis=1)) |
            (frame.low > frame[['open', 'close', 'high']].min(axis=1))).any():
        raise ValueError('Invalid source OHLC ordering')
    if positive:
        if not np.isfinite(frame[['volume', 'buy_volume', 'quote_volume']]).all().all():
            raise ValueError('Non-finite source volume')
        if ((frame.volume < 0) | (frame.buy_volume < 0) | (frame.buy_volume > frame.volume + 1e-8)).any():
            raise ValueError('Invalid taker volumes')
    return frame


def load_extra(symbol, source, index, root='data/study005'):
    paths = sorted(Path(root, source, 'monthly').glob(f'{symbol}-*.zip'))
    if not paths:
        raise ValueError(f'Missing {symbol}/{source}')
    frame = normalize(pd.concat([read_archive(p) for p in paths]), positive=source == 'spot')
    missing = index.difference(frame.index)
    manifest_path = Path(root, 'manifest.json')
    manifest = json.loads(manifest_path.read_text())
    repaired = []
    for day in missing.strftime('%Y-%m-%d').unique():
        item = fetch((symbol, source, day, root))
        daily = normalize(read_archive(item['path']), positive=source == 'spot')
        frame = pd.concat([frame, daily.loc[daily.index.intersection(missing)]])
        repaired.append(day)
        if item['url'] not in {v['url'] for v in manifest}:
            manifest.append(item)
    if repaired:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    if frame.index.duplicated().any() or len(index.difference(frame.index)):
        raise ValueError(f'Unresolved source coverage: {symbol}/{source}')
    frame = frame.reindex(index)
    if not frame.index.equals(index) or frame[['open', 'close']].isna().any().any():
        raise ValueError('Source join failed')
    return frame, repaired


def download(root='data/study005', start='2022-01', end='2026-08'):
    jobs = [(s, k, m, root) for s in ['BTCUSDT', 'ETHUSDT'] for k in SOURCES
            for m in pd.period_range(start, end, freq='M').astype(str)]
    manifest = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for item in pool.map(fetch, jobs):
            manifest.append(item)
            if len(manifest) % 28 == 0:
                print(f'Supplementary archives verified {len(manifest)}/{len(jobs)}', flush=True)
    Path(root, 'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    download()
