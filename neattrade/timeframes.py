"""Causal aggregation, common risk inputs and a15m accounting clock."""
import numpy as np
import pandas as pd

from .alphatrend import alpha_indicator
from .market import Market, simulate

TIMEFRAMES = {'15m': 15, '30m': 30, '1h': 60, '2h': 120, '4h': 240,
              '6h': 360, '8h': 480, '12h': 720, '1d': 1440, '1w': 10080}
PER_YEAR = 365.25*96


def aggregate(bars, timeframe):
    """Only complete OHLCV buckets; weekly bins start Monday00:00UTC."""
    if timeframe not in TIMEFRAMES:
        raise ValueError('Unsupported timeframe')
    index = bars.index
    if index.tz is None or str(index.tz) != 'UTC' or index.has_duplicates or len(index) < 2:
        raise ValueError('Expected unique UTC15m history')
    if not np.all(np.diff(index.as_unit('ns').asi8) == pd.Timedelta(minutes=15).value):
        raise ValueError('Incomplete source15m grid')
    if not ((index.minute % 15 == 0) & (index.second == 0) & (index.microsecond == 0)).all():
        raise ValueError('Source timestamps not aligned to15m boundaries')
    rule = 'W-MON' if timeframe == '1w' else f'{TIMEFRAMES[timeframe]}min'
    options = {} if timeframe == '1w' else {'origin': 'epoch'}
    resampler = bars.resample(rule, label='left', closed='left', **options)
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    for name in ['quote_volume', 'trades', 'buy_volume', 'buy_quote_volume']:
        if name in bars:
            agg[name] = 'sum'
    result = resampler.agg(agg)
    expected = TIMEFRAMES[timeframe]//15
    counts = resampler.size()
    if len(counts) > 2 and not (counts.iloc[1:-1] == expected).all():
        raise ValueError('Incomplete internal aggregate bucket')
    return result.loc[counts == expected]


def signals(bars, timeframe):
    agg = aggregate(bars, timeframe)
    indicator = alpha_indicator(agg)
    delta = pd.Timedelta(minutes=TIMEFRAMES[timeframe])
    state = indicator.state.copy()
    state.index = state.index+delta
    # Index labels are information-availability times, never candle starts.
    return state.reindex(bars.index, method='ffill'), state.index, indicator


def common_volatility(bars):
    four = aggregate(bars, '4h')
    vol = (np.log(four.close).diff().rolling(42).std().clip(lower=.001)*np.sqrt(365.25*6)).clip(-5, 5)
    vol.index = vol.index+pd.Timedelta(hours=4)
    return vol.reindex(bars.index, method='ffill')


def inputs(bars):
    vol = common_volatility(bars)
    states, closes = {}, {}
    for tf in TIMEFRAMES:
        states[tf], closes[tf], _ = signals(bars, tf)
    return {'vol': vol, 'states': states, 'closes': closes}


def make_market(symbol, bars, features, start, end):
    start, end = pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC')
    index = bars.index[(bars.index >= start) & (bars.index <= end)]
    if len(index) < 3 or index[0] != start or index[-1] != end:
        raise ValueError('Missing exact window endpoints')
    b = bars.reindex(index)
    x = np.zeros((len(index)-1, 7))
    x[:, 6] = features['vol'].reindex(index[:-1]).to_numpy()
    if not np.isfinite(x).all():
        raise ValueError('Insufficient common risk warmup')
    return Market(symbol, index, x, b.open.to_numpy(), b.mark.to_numpy(), b.funding.to_numpy(),
                  b.low.to_numpy()[:-1], b.high.to_numpy()[:-1])


def orders(m, features, timeframe, long_short, delay=0, passive=None):
    index = m.times[:-1]
    state = features['states'][timeframe].reindex(index).to_numpy(float)
    if not np.isfinite(state).all() or not np.isin(state, [-1, 0, 1]).all():
        raise ValueError('Unavailable AlphaTrend direction')
    if not long_short:
        state = np.maximum(state, 0.)
    if passive:
        if passive not in ['vol', 'half']:
            raise ValueError('Unknown passive control')
        state = np.ones(len(index))
    target = state*np.minimum(.5, .20/np.maximum(m.x[:, 6], .10))
    if passive == 'half':
        target = np.full(len(index), .5)
    allowed = np.asarray(index.isin(features['closes'][timeframe]))
    allowed[0] = True
    # Hold the known target between own-timeframe decisions; quantities remain
    # fixed in the simulator. Delay moves BOTH signal and order eligibility.
    anchors = np.maximum.accumulate(np.where(allowed, np.arange(len(index)), 0))
    target = target[anchors]
    if not isinstance(delay, int) or delay < 0:
        raise ValueError('Delay must be a nonnegative15m bar count')
    if delay:
        if delay >= len(target):
            return np.zeros(len(target)), np.zeros(len(target), bool)
        target = np.r_[np.zeros(delay), target[:-delay]]
        allowed = np.r_[np.zeros(delay, bool), allowed[:-delay]]
    return target, allowed


def metrics(returns):
    r = np.asarray(returns, float)
    if not len(r) or not np.isfinite(r).all() or np.any(r <= -1):
        raise ValueError('Invalid equity returns')
    eq = np.r_[1., np.cumprod(1+r)]
    return {'return': float(eq[-1]-1), 'max_drawdown': float(np.max(1-eq/np.maximum.accumulate(eq))),
            'sharpe': float(r.mean()/r.std()*np.sqrt(PER_YEAR)) if r.std() > 1e-12 else 0.,
            'equity': eq, 'returns': r}


def portfolio(results):
    curves = [r['equity'] for r in results.values()]
    if len({len(eq) for eq in curves}) != 1:
        raise ValueError('Unaligned portfolio')
    eq = np.mean(curves, axis=0)
    return metrics(eq[1:]/eq[:-1]-1)


def evaluate(m, features, timeframe, long_short, multiplier=1., delay=0, passive=None):
    targets, allowed = orders(m, features, timeframe, long_short, delay, passive)
    result = simulate(m, targets, 5*multiplier, 2*multiplier, trade_allowed=allowed, rebalance_band=.05)
    # Existing simulator accounting is interval-agnostic; its built-in Sharpe
    # annualization assumes4h. Replace that metric with the common15m clock.
    result['sharpe'] = metrics(result['returns'])['sharpe']
    return result


def daily_returns(returns, endpoints):
    index = (pd.DatetimeIndex(endpoints).tz_localize(None)-pd.Timedelta(nanoseconds=1)).to_period('D')
    return pd.Series(1+np.asarray(returns), index=index).groupby(level=0).prod()-1


def scenarios(timeframe):
    return [('base', {}), ('double_cost', {'multiplier': 2.}),
            ('delay15m', {'delay': 1}), ('delay_native', {'delay': TIMEFRAMES[timeframe]//15})]


def specifications():
    return ([{'name': f'{tf}_{mode}', 'timeframe': tf, 'long_short': mode == 'ls'}
             for tf in TIMEFRAMES for mode in ['long', 'ls']]
            + [{'name': 'passive_'+s, 'timeframe': '4h', 'long_short': False, 'passive': s} for s in ['vol', 'half']])
