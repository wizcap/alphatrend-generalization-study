# This file is subject to the Mozilla Public License 2.0:
# https://mozilla.org/MPL/2.0/
# AlphaTrend algorithm: copyright KivancOzbilgic. Python adaptation: study007.
"""Confirmed AlphaTrend signals and causal research inputs.

Original open strategy v2.0 is saved under references/alphatrend. Its ATR is
SMA(TR), NOT Wilder ATR. Its orders use raw crosses; the barssince filters in
that source only affect plot labels. This module does not use those labels.
"""
import numpy as np
import pandas as pd

from .features_v5 import ALL_FEATURES
from .indicator_types import TYPES

ALPHA_FEATURES = ['alpha_state', 'alpha_distance', 'alpha_slope', 'alpha_mfi']
FEATURE_NAMES = ALL_FEATURES + ALPHA_FEATURES
STATE = FEATURE_NAMES.index('alpha_state')


def cross_state(line):
    """Raw confirmed crosses, holding the last direction through equal plateaus."""
    line = pd.Series(line, copy=False)
    spread = line-line.shift(2)
    buy = (spread > 0) & (spread.shift() <= 0)
    sell = (spread < 0) & (spread.shift() >= 0)
    state = pd.Series(np.nan, index=line.index)
    state.loc[buy], state.loc[sell] = 1., -1.
    return buy, sell, state.ffill().fillna(0.)


def alpha_indicator(bars, period=14, multiplier=1.):
    """Values known at the close of each indexed bar; NOT executable inputs yet."""
    if not isinstance(period, int) or period < 1 or not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('Invalid AlphaTrend parameters')
    values = bars[['high', 'low', 'close', 'volume']].to_numpy(float)
    if not np.isfinite(values).all() or np.any(values[:, :3] <= 0) or np.any(values[:, 3] < 0):
        raise ValueError('Expected finite positive OHLC and nonnegative volume')
    if np.any(bars.high < bars.low):
        raise ValueError('High below low')
    typical = (bars.high+bars.low+bars.close)/3
    change = typical.diff()
    # Pine v5 ternary treats an na condition as false. The first money flow
    # consequently enters both sums; it drops out before ATR becomes valid.
    pos = (typical*bars.volume).where(~(change <= 0), 0.).rolling(period).sum()
    neg = (typical*bars.volume).where(~(change >= 0), 0.).rolling(period).sum()
    total = pos+neg
    mfi = 100*pos/total.replace(0, np.nan)
    tr = pd.concat([bars.high-bars.low, (bars.high-bars.close.shift()).abs(),
                    (bars.low-bars.close.shift()).abs()], axis=1).max(axis=1, skipna=False)
    atr = tr.rolling(period).mean()
    upper, lower = bars.low-multiplier*atr, bars.high+multiplier*atr
    line = np.full(len(bars), np.nan)
    previous = 0.
    for i, (up, down, momentum) in enumerate(zip(upper, lower, mfi)):
        # Preserve Pine's nz(previous) bootstrap and na comparison semantics.
        prior = previous if np.isfinite(previous) else 0.
        previous = (prior if up < prior else up) if momentum >= 50 else (prior if down > prior else down)
        line[i] = previous
    line = pd.Series(line, index=bars.index)
    buy, sell, state = cross_state(line)
    return pd.DataFrame({'line': line, 'atr_sma': atr, 'mfi': mfi,
                         'buy': buy, 'sell': sell, 'state': state}, index=bars.index)


def alpha_features(bars):
    indicator = alpha_indicator(bars)
    atr = indicator.atr_sma.clip(lower=1e-12)
    frame = pd.DataFrame({'alpha_state': indicator.state,
                          'alpha_distance': (bars.close-indicator.line)/atr,
                          'alpha_slope': (indicator.line-indicator.line.shift(2))/atr,
                          'alpha_mfi': indicator.mfi.fillna(50)/50-1}, index=bars.index)
    # Bar t close is available only at t+1 open. No current high/low/volume.
    return frame.clip(-5, 5).shift(1)


def specifications():
    return ([{'name': 'at_constant', 'features': []},
             {'name': 'at_neat', 'features': list(ALPHA_FEATURES)}]
            + [{'name': 'at_'+group, 'features': ALPHA_FEATURES+names} for group, names in TYPES.items()]
            + [{'name': 'at_all', 'features': list(FEATURE_NAMES)}])


def select_inputs(x, spec):
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
        raise ValueError('Expected full ordered AlphaTrend study features')
    names = spec['features']
    if len(names) != len(set(names)):
        raise ValueError('Duplicate inputs')
    return x[:, [FEATURE_NAMES.index(n) for n in names]] if names else np.ones((len(x), 1))


def alpha_targets(x, long_short=False):
    direction = x[:, STATE] if long_short else np.maximum(x[:, STATE], 0.)
    return direction*np.minimum(.5, .20/np.maximum(x[:, 6], .10))


def gated_targets(raw, x):
    scales = np.where(raw < -.35, 0., np.where(raw > .35, 1., .5))
    return scales*alpha_targets(x)[None, :]
