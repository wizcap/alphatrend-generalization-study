"""Public, closed-candle acquisition with explicit immutable receipt evidence."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

from .data import COLS
from .adaptive_v13 import utc

SYMBOLS = ['BTCUSDT', 'ETHUSDT']
BASE = 'https://fapi.binance.com'
STEP = pd.Timedelta(minutes=15)


def now():
    return pd.Timestamp(datetime.now(timezone.utc))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def millis(value):
    return utc(value).value//1_000_000


def request(endpoint, params, root, evidence):
    if endpoint not in ['/fapi/v1/time', '/fapi/v1/klines', '/fapi/v1/markPriceKlines', '/fapi/v1/fundingRate']:
        raise ValueError('Only registered public market GET endpoints are permitted')
    for attempt in range(3):
        try:
            response = requests.get(BASE+endpoint, params=params, timeout=30)
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(attempt+1)
    received = now()
    name = f'{len(evidence):03d}.json'
    (root/name).write_bytes(response.content)
    evidence.append({'endpoint': endpoint, 'params': params, 'received_at': str(received),
                     'file': name, 'sha256': sha256(response.content).hexdigest()})
    return response.json(), received


def candles(rows, start, end, received):
    if not rows or any(not isinstance(r, list) or len(r) != len(COLS) for r in rows):
        raise ValueError('Invalid candle response schema')
    b = pd.DataFrame(rows, columns=COLS).apply(pd.to_numeric, errors='raise')
    if not np.isfinite(b.to_numpy()).all():
        raise ValueError('Nonfinite candle fields')
    b.index = pd.to_datetime(b.time, unit='ms', utc=True)
    expected = pd.date_range(utc(start), utc(end), freq='15min', inclusive='left')
    if not b.index.equals(expected) or b.index.has_duplicates:
        raise ValueError('Missing, duplicate, unordered or unexpected candles')
    if (b.close_time != b.time+899_999).any() or (b.close_time >= millis(received)).any():
        raise ValueError('Invalid or still-open candle close timestamp')
    prices = b[['open', 'high', 'low', 'close']]
    if (prices <= 0).any().any() or ((b.high < prices.max(axis=1)) | (b.low > prices.min(axis=1))).any():
        raise ValueError('Invalid positive OHLC ordering')
    if (b[['volume', 'quote_volume', 'trades', 'buy_volume', 'buy_quote_volume']] < 0).any().any():
        raise ValueError('Invalid candle volume')
    if (b.buy_volume > b.volume+1e-8).any():
        raise ValueError('Taker volume exceeds total volume')
    b.index.name = 'time'
    return b


def funding(rows, symbol, start, end):
    if not rows or any(r.get('symbol') != symbol for r in rows):
        raise ValueError('Invalid funding symbol or missing settlements')
    times = np.asarray([r['fundingTime'] for r in rows], dtype=np.int64)
    normalized = np.rint(times/900_000).astype(np.int64)*900_000
    if np.any(np.abs(times-normalized) > 1000):
        raise ValueError('Funding settlement not on supported grid')
    rates = np.asarray([r['fundingRate'] for r in rows], float)
    index = pd.to_datetime(normalized, unit='ms', utc=True)
    expected = pd.date_range(utc(start).ceil('8h'), utc(end), freq='8h', inclusive='left')
    if not index.equals(expected) or index.has_duplicates or not np.isfinite(rates).all():
        raise ValueError('Missing/duplicate funding or unsupported settlement schedule')
    return pd.Series(rates, index=index)


def merge_history(previous, incoming):
    if list(previous.columns) != list(incoming.columns) or incoming.index.has_duplicates:
        raise ValueError('Canonical history schema changed')
    overlap = previous.index.intersection(incoming.index)
    if len(overlap) < 96 or overlap[-1] != previous.index[-1]:
        raise ValueError('Require at least a full day of trailing history overlap')
    # Decimal CSV/JSON parsing can differ by one binary ULP. This tolerance is
    # below exchange price precision and is fixed before the first acquisition.
    if not np.allclose(previous.loc[overlap], incoming.loc[overlap], rtol=2e-14, atol=1e-12):
        raise ValueError('Already committed market observation was revised')
    extra = incoming.loc[incoming.index > previous.index[-1]]
    if len(extra) and extra.index[0] != previous.index[-1]+STEP:
        raise ValueError('Gap when appending market history')
    result = pd.concat([previous, extra])
    if result.index.has_duplicates or np.any(np.diff(result.index.as_unit('ns').asi8) != STEP.value):
        raise ValueError('Canonical history is not contiguous')
    return result


def acquire(previous, root):
    """Write a new raw receipt directory; return appended data without committing."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    evidence = []
    clock, received = request('/fapi/v1/time', {}, root, evidence)
    server = pd.Timestamp(clock['serverTime'], unit='ms', tz='UTC')
    if abs((received-server).total_seconds()) > 300:
        raise ValueError('Local and exchange clocks differ by more than five minutes')
    end = (server-pd.Timedelta(minutes=1)).floor('15min')
    result, counts = {}, {}
    if len({b.index[-1] for b in previous.values()}) != 1:
        raise ValueError('Prior symbols have different data endpoints')
    for symbol in SYMBOLS:
        start = previous[symbol].index[-1].normalize()-pd.Timedelta(days=1)
        if end <= previous[symbol].index[-1]:
            raise ValueError('Exchange data has regressed behind committed history')
        kinds = {}
        for kind in ['klines', 'markPriceKlines']:
            rows, cursor = [], millis(start)
            while cursor < millis(end):
                page, _ = request('/fapi/v1/'+kind, {'symbol': symbol, 'interval': '15m',
                    'startTime': cursor, 'endTime': millis(end)-1, 'limit': 1000}, root, evidence)
                if not isinstance(page, list) or not page or int(page[-1][0]) < cursor:
                    raise ValueError('Nonadvancing/incomplete candle page')
                rows.extend(page)
                cursor = int(page[-1][0])+900_000
            kinds[kind] = candles(rows, start, end, server-pd.Timedelta(minutes=1))
        rows, cursor = [], millis(start)
        while cursor < millis(end):
            page, _ = request('/fapi/v1/fundingRate', {'symbol': symbol, 'startTime': cursor,
                'endTime': millis(end)-1, 'limit': 1000}, root, evidence)
            if not isinstance(page, list):
                raise ValueError('Invalid funding page')
            if not page:
                break
            if int(page[-1]['fundingTime']) < cursor:
                raise ValueError('Nonadvancing funding page')
            rows.extend(page)
            cursor = int(page[-1]['fundingTime'])+1
            if len(page) < 1000:
                break
        rates = funding(rows, symbol, start, end)
        b = kinds['klines'].copy()
        b['mark'] = kinds['markPriceKlines'].open
        b['funding'] = rates.reindex(b.index, fill_value=0.)
        result[symbol] = merge_history(previous[symbol], b)
        counts[symbol] = {'downloaded_candles_per_kind': len(b), 'funding_events': len(rates),
            'appended_candles': len(result[symbol])-len(previous[symbol])}
    manifest = {'server_time': str(server), 'closed_before': str(end), 'received_at': str(now()),
                'usable_endpoint': str((end-STEP).floor('4h')), 'counts': counts, 'responses': evidence}
    write_json(root/'manifest.json', manifest)
    return result, manifest


def verify_receipts(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    for r in manifest['responses']:
        if sha256((root/r['file']).read_bytes()).hexdigest() != r['sha256']:
            raise ValueError('Corrupt market response evidence')
    return manifest
