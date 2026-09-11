"""Verified official15m futures candles for a common multi-timeframe clock."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

from .data import read_archive

BASE = 'https://data.binance.vision/data/futures/um'
SYMBOLS = ['BTCUSDT', 'ETHUSDT']
START, LAST = '2022-01', '2026-08'


def fetch(job):
    symbol, kind, date, root = job[:4]
    interval = job[4] if len(job) > 4 else '15m'
    frequency = 'daily' if len(date) == 10 else 'monthly'
    name = f'{symbol}-{interval}-{date}.zip'
    url = f'{BASE}/{frequency}/{kind}/{symbol}/{interval}/{name}'
    path = Path(root, kind, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    checksum = path.with_suffix('.zip.CHECKSUM')
    for attempt in range(4):
        try:
            if not checksum.exists():
                response = requests.get(url+'.CHECKSUM', timeout=45)
                response.raise_for_status()
                checksum.write_text(response.text, encoding='utf-8')
            expected = checksum.read_text().split()[0]
            if len(expected) != 64:
                raise ValueError('Invalid published checksum')
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
            return {'url': url, 'path': str(path), 'sha256': expected, 'bytes': len(payload)}
        except (requests.RequestException, ValueError):
            if attempt == 3:
                raise
            time.sleep(1+attempt)


def download(root='data/study008'):
    months = pd.period_range(START, LAST, freq='M').astype(str)
    jobs = [(s, k, d, root) for s in SYMBOLS for k in ['klines', 'markPriceKlines'] for d in months]
    manifest = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for item in pool.map(fetch, jobs):
            manifest.append(item)
            if len(manifest) % 28 == 0:
                print(f'Verified15m archives: {len(manifest)}/{len(jobs)}', flush=True)
    Path(root, 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def load(symbol, root='data/study008', funding_root='data'):
    expected = pd.date_range('2022-01-01', '2026-09-01', freq='15min', inclusive='left', tz='UTC')
    frames, repair_records = {}, []
    for kind in ['klines', 'markPriceKlines']:
        paths = sorted(Path(root, kind).glob(f'{symbol}-15m-*.zip'))
        if not paths:
            raise ValueError('Run sources_v8 download first')
        frame = pd.concat([read_archive(p) for p in paths], ignore_index=True).sort_values('time')
        if frame.time.duplicated().any():
            raise ValueError('Duplicate monthly source timestamps')
        frame.index = pd.to_datetime(frame.time, unit='ms', utc=True)
        missing = expected.difference(frame.index)
        for day in sorted(set(missing.strftime('%Y-%m-%d'))):
            item = fetch((symbol, kind, day, str(Path(root, 'repairs'))))
            daily = read_archive(item['path'])
            daily.index = pd.to_datetime(daily.time, unit='ms', utc=True)
            frame = pd.concat([frame, daily.loc[daily.index.isin(missing)]])
            repair_records.append(item)
        missing = expected.difference(frame.index)
        for day in sorted(set(missing.strftime('%Y-%m-%d'))):
            # Both monthly and daily15m mark files omit2023-11-10 03:45.
            # Reconstruct only missing candles from complete verified1m bins.
            item = fetch((symbol, kind, day, str(Path(root, 'repairs1m')), '1m'))
            fine = read_archive(item['path'])
            fine.index = pd.to_datetime(fine.time, unit='ms', utc=True)
            if fine.index.has_duplicates:
                raise ValueError('Duplicate1m repair source')
            aggregation = {name: 'sum' for name in fine.columns}
            aggregation.update(time='first', open='first', high='max', low='min', close='last', close_time='last', ignore='last')
            grouped = fine.resample('15min', origin='epoch', label='left', closed='left')
            recovered = grouped.agg(aggregation)
            recovered = recovered.loc[(grouped.size() == 15) & recovered.index.isin(missing)]
            for t in recovered.index:
                actual = fine.loc[t:t+pd.Timedelta(minutes=14)].index
                required = pd.date_range(t, periods=15, freq='min')
                if not actual.equals(required):
                    raise ValueError('Incomplete1m repair bin')
            frame = pd.concat([frame, recovered])
            repair_records.append(item)
        frame = frame.sort_index()
        if frame.index.duplicated().any() or not frame.index.equals(expected):
            raise ValueError(f'{symbol}/{kind}: incomplete or unexpected15m grid')
        price = frame[['open', 'high', 'low', 'close']]
        if not np.isfinite(price).all().all() or (price <= 0).any().any():
            raise ValueError('Invalid price')
        if ((frame.high < frame[['open', 'close', 'low']].max(axis=1)) |
                (frame.low > frame[['open', 'close', 'high']].min(axis=1))).any():
            raise ValueError('Invalid OHLC ordering')
        if not np.isfinite(frame.volume).all() or (frame.volume < 0).any():
            raise ValueError('Invalid volume')
        frames[kind] = frame
    paths = [Path(funding_root, 'fundingRate', f'{symbol}-fundingRate-{month}.zip')
             for month in pd.period_range(START, LAST, freq='M').astype(str)]
    funding = pd.concat([read_archive(p, funding=True) for p in paths], ignore_index=True).sort_values('time')
    # Match the established <=1second settlement timestamp normalization.
    rounded = (funding.time/900_000).round().astype('int64')*900_000
    if (funding.time-rounded).abs().max() > 1000 or rounded.duplicated().any():
        raise ValueError('Invalid funding timestamps')
    if not np.isfinite(funding.rate).all() or not np.allclose(np.diff(rounded)/3_600_000, funding.interval.to_numpy()[1:]):
        raise ValueError('Missing funding settlements')
    funding.index = pd.to_datetime(rounded, unit='ms', utc=True)
    schedule = pd.date_range('2022-01-01', '2026-09-01', freq='8h', inclusive='left', tz='UTC')
    if not funding.index.equals(schedule):
        raise ValueError('BTC/ETH funding schedule differs from the audited8h calendar')
    if not funding.index.isin(expected).all() or funding.index[0] > expected[0] or funding.index[-1] < expected[-1]-pd.Timedelta(hours=8):
        raise ValueError('Incomplete funding coverage')
    result = frames['klines'].copy()
    result['mark'] = frames['markPriceKlines'].open
    result['funding'] = funding.rate.reindex(expected, fill_value=0.)
    result.index.name = 'time'
    manifest_path = Path(root, 'manifest.json')
    manifest = json.loads(manifest_path.read_text())
    known = {r['url'] for r in manifest}
    for item in repair_records:
        if item['url'] not in known:
            manifest.append(item)
            known.add(item['url'])
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    audit = {'symbol': symbol, 'bars': len(result), 'first': str(result.index[0]), 'last': str(result.index[-1]),
             'funding_events': len(funding), 'zero_rate_events': int((funding.rate == 0).sum()),
             'funding_calendar_including_zero_rates_verified': True,
             'repairs': repair_records, 'funding_files': [str(p) for p in paths]}
    return result, audit


if __name__ == '__main__':
    download()
    for symbol in SYMBOLS:
        _, audit = load(symbol)
        print(json.dumps(audit, indent=2), flush=True)
