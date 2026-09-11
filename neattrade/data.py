"""Verified Binance public archives. No private exchange credentials needed."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json
import time
import zipfile

import numpy as np
import pandas as pd
import requests

BASE = 'https://data.binance.vision/data/futures/um/monthly'
COLS = ['time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_volume', 'trades', 'buy_volume', 'buy_quote_volume', 'ignore']


def fetch_one(job):
    symbol, kind, month, root = job
    suffix = f'{symbol}-fundingRate-{month}.zip' if kind == 'fundingRate' else f'{symbol}-4h-{month}.zip'
    base = BASE.replace('/monthly', '/daily') if len(month) == 10 else BASE
    url = f'{base}/{kind}/{symbol}/' + ('' if kind == 'fundingRate' else '4h/') + suffix
    path = Path(root) / kind / suffix
    path.parent.mkdir(parents=True, exist_ok=True)
    checksum_path = path.with_suffix('.zip.CHECKSUM')
    for attempt in range(4):
        try:
            if not checksum_path.exists():
                r = requests.get(url + '.CHECKSUM', timeout=45)
                r.raise_for_status()
                checksum_path.write_text(r.text)
            expected = checksum_path.read_text().split()[0]
            payload = path.read_bytes() if path.exists() else b''
            if sha256(payload).hexdigest() != expected:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                payload = r.content
                if sha256(payload).hexdigest() != expected:
                    raise ValueError(f'Checksum failed: {url}')
                path.write_bytes(payload)
            return {'url': url, 'sha256': expected, 'bytes': len(payload), 'path': str(path)}
        except (requests.RequestException, ValueError):
            if attempt == 3:
                raise
            time.sleep(1 + attempt)


def download(root='data', start='2022-01', end='2026-01', symbols=('BTCUSDT', 'ETHUSDT')):
    months = pd.period_range(start, end, freq='M').astype(str)
    jobs = [(s, k, m, root) for s in symbols for k in ('klines', 'markPriceKlines', 'fundingRate') for m in months]
    with ThreadPoolExecutor(max_workers=8) as pool:
        manifest = []
        for item in pool.map(fetch_one, jobs):
            manifest.append(item)
            if len(manifest) % 30 == 0:
                print(f'Downloaded/verified {len(manifest)}/{len(jobs)}', flush=True)
    Path(root, 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def read_archive(path, funding=False):
    path = Path(path)
    checksum_path = path.with_suffix('.zip.CHECKSUM')
    if not checksum_path.exists() or sha256(path.read_bytes()).hexdigest() != checksum_path.read_text().split()[0]:
        raise ValueError(f'Missing or mismatched archive checksum: {path}')
    with zipfile.ZipFile(path) as z:
        frame = pd.read_csv(BytesIO(z.read(z.namelist()[0])), header=None)
    frame = frame[pd.to_numeric(frame.iloc[:, 0], errors='coerce').notna()].copy()
    frame.columns = ['time', 'interval', 'rate'] if funding else COLS
    for col in frame:
        frame[col] = pd.to_numeric(frame[col], errors='raise')
    return frame


def load(symbol, root='data'):
    def collect(kind):
        paths = sorted(Path(root, kind).glob(f'{symbol}-*.zip'))
        if not paths:
            raise ValueError(f'Missing {symbol} {kind}')
        return pd.concat([read_archive(p, kind == 'fundingRate') for p in paths], ignore_index=True).sort_values('time')
    bars, marks, funding = collect('klines'), collect('markPriceKlines'), collect('fundingRate')
    missing = bars.loc[~bars.time.isin(marks.time), 'time']
    if len(missing):
        # Known omissions in monthly mark archives: recover actual daily records, never forward-fill.
        days = pd.to_datetime(missing, unit='ms', utc=True).dt.strftime('%Y-%m-%d').unique()
        manifest_path = Path(root, 'manifest.json')
        manifest = json.loads(manifest_path.read_text())
        for day in days:
            item = fetch_one((symbol, 'markPriceKlines', day, str(Path(root, 'repairs'))))
            daily = read_archive(item['path'])
            marks = pd.concat([marks, daily[daily.time.isin(missing)]], ignore_index=True)
            if item['url'] not in {x['url'] for x in manifest}:
                manifest.append(item)
            print(f'Recovered {symbol} mark prices from daily archive {day}', flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        marks = marks.sort_values('time')
    for name, frame in [('klines', bars), ('marks', marks)]:
        if frame.time.duplicated().any() or not np.all(np.diff(frame.time) == 14_400_000):
            raise ValueError(f'{symbol} {name}: duplicate or missing 4h candles')
        if not np.isfinite(frame[['open', 'high', 'low', 'close']]).all().all():
            raise ValueError('Non-finite price')
        if (frame[['open', 'high', 'low', 'close']] <= 0).any().any():
            raise ValueError('Non-positive price')
        if ((frame.high < frame[['open', 'close', 'low']].max(axis=1)) |
                (frame.low > frame[['open', 'close', 'high']].min(axis=1))).any():
            raise ValueError('Invalid OHLC ordering')
    bars = bars.set_index('time')
    marks = marks.set_index('time')
    if not bars.index.equals(marks.index):
        raise ValueError('Mark and contract timestamps differ')
    # Snap millisecond settlement jitter; charge incumbent position before execution.
    rounded = (funding.time / 14_400_000).round().astype('int64') * 14_400_000
    if ((funding.time - rounded).abs() > 1000).any():
        raise ValueError('Funding settlement is not on a supported 4h boundary')
    funding['time'] = rounded
    if funding.time.duplicated().any() or not np.isfinite(funding.rate).all():
        raise ValueError('Invalid funding records')
    intervals = np.diff(funding.time) / 3_600_000
    if not np.allclose(intervals, funding.interval.to_numpy()[1:]):
        raise ValueError('Funding coverage gap or unsupported interval change')
    if funding.time.min() > bars.index.min() or funding.time.max() < bars.index.max() - 8 * 3_600_000:
        raise ValueError('Incomplete funding coverage')
    bars['mark'] = marks.open
    bars['funding'] = funding.set_index('time').rate.reindex(bars.index, fill_value=0)
    bars.index = pd.to_datetime(bars.index, unit='ms', utc=True)
    bars.index.name = 'time'
    return bars
