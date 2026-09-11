"""Compiled population accounting on the unchanged 15m economic clock.

No fastmath or parallel reduction. The original scalar simulator is the audit
reference; this training adapter never supplies published evaluation curves.
"""
import numpy as np
from numba import njit

from .timeframes import PER_YEAR


@njit(cache=True)
def _population(p, anchors, prices, marks, funding, lows, highs, cost, band, save_curves=False):
    result = np.zeros((len(p), 8))
    curves = np.ones((len(p), len(lows)+1)) if save_curves else np.empty((0, 0))
    for g in range(len(p)):
        equity, units, peak = 1., 0., 1.
        drawdown, fees, paid, ruined = 0., 0., 0., False
        orders, turnover, exposures = 0, 0., 0.
        j = 0
        for t in range(len(lows)):
            before = equity
            if j < len(anchors) and t == anchors[j]:
                target = p[g, j] * equity / prices[t]
                if target*units > 0 and abs(target-units)*prices[t]/equity < band:
                    target = units
                fee = abs(target-units)*prices[t]*cost
                traded = abs(target-units)*prices[t]
                orders += int(traded > 1e-10)
                turnover += traded/equity
                equity -= fee
                fees += fee
                units = target
                j += 1
            exposures += abs(units)*prices[t]/before
            adverse = lows[t] if units >= 0 else highs[t]
            if equity + units*(adverse-prices[t]) <= .05*before:
                ruined = True
                break
            equity += units*(prices[t+1]-prices[t])
            payment = units*marks[t+1]*funding[t+1]
            equity -= payment
            paid += payment
            if t == len(lows)-1:
                fee = abs(units)*prices[t+1]*cost
                equity -= fee
                fees += fee
                closing = abs(units)*prices[t+1]
                orders += int(closing > 1e-10)
                turnover += closing/max(equity+fee, 1e-12)
            if not np.isfinite(equity) or equity <= 0:
                ruined = True
                break
            peak = max(peak, equity)
            drawdown = max(drawdown, 1-equity/peak)
            if save_curves:
                curves[g, t+1] = equity
        result[g, 0] = 0. if ruined else equity-1.
        result[g, 1] = drawdown
        result[g, 2] = fees
        result[g, 3] = paid
        result[g, 4] = 1. if ruined else 0.
        result[g, 5] = orders
        result[g, 6] = turnover
        result[g, 7] = exposures/len(lows)
    return result, curves


def population_accounting(prepared, targets, cost_bps=7., rebalance_band=.05):
    p = np.ascontiguousarray(targets, dtype=float)
    if p.ndim != 2 or p.shape[1] != len(prepared.anchors) or not p.size or not np.isfinite(p).all() or np.max(np.abs(p)) > .5000001:
        raise ValueError('Expected finite population by decision targets capped at 0.5')
    if not np.isfinite(cost_bps) or cost_bps < 0 or not np.isfinite(rebalance_band) or rebalance_band < 0:
        raise ValueError('Invalid cost or rebalance band')
    m = prepared.market
    r, _ = _population(p, prepared.anchors, m.prices, m.marks, m.funding, m.lows, m.highs,
                      cost_bps/10000., rebalance_band)
    return {'return': r[:, 0], 'max_drawdown': r[:, 1], 'cost_paid_initial_equity': r[:, 2],
            'funding_paid_initial_equity': r[:, 3], 'ruined': r[:, 4].astype(bool),
            'years': len(m.x)/PER_YEAR}


def audit_simulate(m, positions, allowed, cost_bps=7., rebalance_band=.05):
    """Independent compiled replay of the published scalar evaluation curves."""
    from .market import InsolventSimulation
    from .timeframes import metrics
    anchors = np.flatnonzero(allowed)
    p = np.ascontiguousarray(np.asarray(positions)[anchors][None, :], dtype=float)
    if not np.isfinite(p).all() or (p.size and np.max(np.abs(p)) > .5000001):
        raise ValueError('Invalid audit targets')
    r, curves = _population(p, anchors, m.prices, m.marks, m.funding, m.lows, m.highs,
                            cost_bps/10000., rebalance_band, True)
    if r[0, 4]:
        raise InsolventSimulation('Independent audit equity floor')
    eq = curves[0]
    returns = eq[1:]/eq[:-1]-1
    return {'return': r[0, 0], 'max_drawdown': r[0, 1], 'cost_paid_initial_equity': r[0, 2],
            'funding_paid_initial_equity': r[0, 3], 'order_count': int(r[0, 5]), 'turnover': r[0, 6],
            'mean_abs_exposure': r[0, 7], 'equity': eq, 'returns': returns,
            'sharpe': metrics(returns)['sharpe']}
