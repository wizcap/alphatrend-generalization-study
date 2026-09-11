from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURES = ['ret1', 'ret6', 'ret42', 'ret180', 'trend42', 'trend180', 'vol', 'range', 'volume_z', 'buy_fraction']
BARS_YEAR = 365.25 * 6


class InsolventSimulation(ValueError):
    """A policy breached the simulator's equity safety floor."""


def features(bars):
    """Inputs available at the execution open. Every feature is lagged one bar."""
    c = bars.close
    r = np.log(c).diff()
    vol = r.rolling(42).std().clip(lower=0.001)
    f = pd.DataFrame(index=bars.index)
    for n in [1, 6, 42, 180]:
        f[f'ret{n}'] = np.log(c / c.shift(n)) / (vol * np.sqrt(n))
    for n in [42, 180]:
        f[f'trend{n}'] = np.log(c / c.ewm(span=n, adjust=False).mean()) / (vol * np.sqrt(n))
    f['vol'] = vol * np.sqrt(BARS_YEAR)
    f['range'] = (bars.high - bars.low) / c / vol
    v = np.log1p(bars.volume)
    f['volume_z'] = (v - v.rolling(42).mean()) / v.rolling(42).std().clip(lower=0.01)
    f['buy_fraction'] = (bars.buy_volume / bars.volume.replace(0, np.nan) - 0.5) * 4
    return f[FEATURES].clip(-5, 5).shift(1)


@dataclass
class Market:
    symbol: str
    times: pd.DatetimeIndex
    x: np.ndarray
    prices: np.ndarray
    marks: np.ndarray
    funding: np.ndarray
    lows: np.ndarray
    highs: np.ndarray


def make_market(symbol, bars, start, end):
    f = features(bars)
    mask = (bars.index >= pd.Timestamp(start, tz='UTC')) & (bars.index < pd.Timestamp(end, tz='UTC')) & f.notna().all(axis=1)
    ix = np.flatnonzero(mask)
    if len(ix) < 2 or ix[-1] + 1 >= len(bars):
        raise ValueError('Insufficient candles including terminal execution price')
    if not np.all(np.diff(ix) == 1):
        raise ValueError('Noncontiguous feature history')
    sl = slice(ix[0], ix[-1] + 2)
    if not np.all(np.diff(bars.index[sl].as_unit('ns').asi8) == pd.Timedelta(hours=4).value):
        raise ValueError('Market timestamps must form an uninterrupted 4h grid')
    return Market(symbol, bars.index[sl], f.iloc[ix].to_numpy(float), bars.open.iloc[sl].to_numpy(float),
                  bars.mark.iloc[sl].to_numpy(float), bars.funding.iloc[sl].to_numpy(float),
                  bars.low.iloc[ix].to_numpy(float), bars.high.iloc[ix].to_numpy(float))


def simulate(m, positions, fee_bps=5.0, slippage_bps=2.0, trade_allowed=None, rebalance_band=0., trace=False):
    """Fixed contract units between 4h rebalances; actual mark-price funding.

    Desired notional is fraction of pre-trade equity (maximum 0.5). At each next
    open, charge the position held over settlement, then rebalance. Liquidate at
    the terminal open including costs. A conservative intrabar equity floor
    invalidates a run; this is not an exchange liquidation engine.
    """
    p = np.asarray(positions, dtype=float)
    if len(p) != len(m.x) or not np.isfinite(p).all() or np.max(np.abs(p)) > 0.5000001:
        raise ValueError('Positions must be finite, <= 0.5 notional/equity, one per candle')
    cost = (fee_bps + slippage_bps) / 10000
    if fee_bps < 0 or slippage_bps < 0:
        raise ValueError('Costs cannot be negative')
    if not np.isfinite(rebalance_band) or rebalance_band < 0:
        raise ValueError('Rebalance band must be finite and nonnegative')
    allowed = np.ones(len(p), dtype=bool) if trade_allowed is None else np.asarray(trade_allowed, dtype=bool)
    if len(allowed) != len(p):
        raise ValueError('trade_allowed must match positions')
    e, q = 1.0, 0.0
    equity, returns, exposures = [e], [], []
    turnover = fees = funding_paid = 0.0
    changes = 0
    ledger = []
    for t in range(len(p)):
        before = e
        target = p[t] * e / m.prices[t] if allowed[t] else q
        # Entry, full exit and direction reversal bypass the no-trade band.
        if q * target > 0 and abs(target-q)*m.prices[t]/e < rebalance_band:
            target = q
        traded = abs(target - q) * m.prices[t]
        if traded <= 1e-10:
            category = 'none'
        elif q == 0:
            category = 'entry'
        elif target == 0:
            category = 'exit'
        elif target*q < 0:
            category = 'flip'
        else:
            category = 'small_adjustment' if traded/e < .05 else 'large_adjustment'
        turnover += traded / e
        fees += traded * cost
        changes += int(traded > 1e-10)
        e -= traded * cost
        q = target
        exposures.append(abs(q) * m.prices[t] / before)
        adverse = m.lows[t] if q >= 0 else m.highs[t]
        if e + q * (adverse - m.prices[t]) <= 0.05 * before:
            raise InsolventSimulation('Intrabar equity floor breached; requires liquidation model')
        price_pnl = q * (m.prices[t + 1] - m.prices[t])
        e += price_pnl
        payment = q * m.marks[t + 1] * m.funding[t + 1]
        e -= payment
        funding_paid += payment
        exit_fee = 0.
        if t == len(p) - 1:
            closing = abs(q) * m.prices[t + 1]
            exit_fee = closing * cost
            fees += closing * cost
            turnover += closing / max(e, 1e-12)
            changes += int(closing > 1e-10)
            e -= closing * cost
        if e <= 0 or not np.isfinite(e):
            raise InsolventSimulation('Insolvent simulation')
        returns.append(e / before - 1)
        equity.append(e)
        if trace:
            ledger.append({'time': str(m.times[t]), 'equity_before': before, 'equity_after': e,
                           'units': q, 'target_exposure': p[t], 'slow_trend': m.x[t, 5],
                           'price_pnl': price_pnl, 'funding_paid': payment, 'trade_cost': traded*cost,
                           'terminal_exit_cost': exit_fee, 'traded_equity': traded/before,
                           'order_category': category})
    eq = np.asarray(equity)
    r = np.asarray(returns)
    dd = 1 - eq / np.maximum.accumulate(eq)
    result = {'return': float(e - 1), 'max_drawdown': float(dd.max()),
            'sharpe': float(r.mean() / r.std() * np.sqrt(BARS_YEAR)) if r.std() > 1e-12 else 0.,
            'turnover': float(turnover), 'order_count': changes, 'cost_paid_initial_equity': float(fees),
            'funding_paid_initial_equity': float(funding_paid), 'mean_abs_exposure': float(np.mean(exposures)),
            'equity': eq, 'returns': r}
    if trace:
        result['ledger'] = ledger
    return result


def score(result):
    return float(np.log1p(result['return']) - 1.5 * result['max_drawdown'])


def summary(result):
    return {k: v for k, v in result.items() if k not in ('equity', 'returns', 'ledger')}
