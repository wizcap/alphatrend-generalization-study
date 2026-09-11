"""Population fitness accounting, vectorized across genomes and daily holds.

Independent from the reference simulator; parity tests cover changing quantities,
mark-price funding, execution costs, terminal exit and four-hour drawdowns.
"""
from dataclasses import dataclass

import numpy as np

from .market import BARS_YEAR


@dataclass
class DailyMarket:
    market: object
    anchors: np.ndarray
    x: np.ndarray
    segments: list


def prepare(m):
    anchors = np.flatnonzero(np.r_[True, m.times[1:-1].hour == 0])
    segments = []
    for a, z in zip(anchors, np.r_[anchors[1:], len(m.x)]):
        cumulative_pnl_per_unit = m.prices[a+1:z+1] - m.prices[a] - np.cumsum(m.marks[a+1:z+1] * m.funding[a+1:z+1])
        segments.append((int(a), int(z), cumulative_pnl_per_unit))
    return DailyMarket(m, anchors, m.x[anchors], segments)


def fitness_accounting(prepared, targets, cost_bps=7, rebalance_band=0.):
    """Return one result per genome; target columns correspond to daily anchors."""
    p = np.asarray(targets, dtype=float)
    if p.ndim != 2 or p.shape[1] != len(prepared.anchors) or not np.isfinite(p).all() or np.max(np.abs(p)) > .5000001:
        raise ValueError('Expected finite population x daily targets capped at 0.5')
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError('Invalid execution cost')
    if not np.isfinite(rebalance_band) or rebalance_band < 0:
        raise ValueError('Invalid rebalance band')
    m = prepared.market
    size = len(p)
    equity = np.ones(size)
    units = np.zeros(size)
    peak = np.ones(size)
    max_dd = np.zeros(size)
    costs = np.zeros(size)
    funding = np.zeros(size)
    ruined = np.zeros(size, dtype=bool)
    for j, (a, z, cumulative) in enumerate(prepared.segments):
        target = p[:, j] * equity / m.prices[a]
        hold = (target*units > 0) & (np.abs(target-units)*m.prices[a]/equity < rebalance_band)
        target = np.where(hold, units, target)
        fee = np.abs(target - units) * m.prices[a] * cost_bps / 10000
        post_trade = equity - fee
        units = target
        curve = post_trade[:, None] + units[:, None] * cumulative
        previous = np.column_stack([post_trade, curve[:, :-1]])
        adverse = np.where(units[:, None] >= 0, m.lows[a:z], m.highs[a:z])
        intrabar = previous + units[:, None] * (adverse - m.prices[a:z])
        reference_before = np.column_stack([equity, curve[:, :-1]])
        ruined |= np.any(intrabar <= .05 * reference_before, axis=1)
        costs += fee
        funding += units * np.sum(m.marks[a+1:z+1] * m.funding[a+1:z+1])
        if z == len(m.x):
            exit_fee = np.abs(units) * m.prices[z] * cost_bps / 10000
            curve[:, -1] -= exit_fee
            costs += exit_fee
        running_peak = np.maximum.accumulate(np.column_stack([peak, curve]), axis=1)[:, 1:]
        max_dd = np.maximum(max_dd, np.max(1 - curve / running_peak, axis=1))
        peak = running_peak[:, -1]
        ruined |= np.any((curve <= 0) | ~np.isfinite(curve), axis=1)
        equity = np.where(ruined, 1., curve[:, -1])
        units = np.where(ruined, 0., units)
    return {'return': equity - 1, 'max_drawdown': max_dd, 'ruined': ruined,
            'cost_paid_initial_equity': costs, 'funding_paid_initial_equity': funding,
            'years': len(m.x) / BARS_YEAR}


def growth_objective(results):
    """Annualize growth, penalize only drawdown beyond the declared 20% budget.

    No exposure/trade-count reward; a cash policy scores exactly zero. Annualized
    growth fixes the former mismatch between year and half-year evaluation spans.
    """
    values = []
    weights = []
    for r in results:
        annual_growth = np.log1p(r['return']) / r['years']
        value = annual_growth - 2 * np.maximum(r['max_drawdown'] - .20, 0)
        values.append(np.where(r['ruined'], -100., value))
        weights.append(r['years'])
    return np.average(values, axis=0, weights=weights)


def legacy_objective(results):
    values = [np.where(r['ruined'], -100., np.log1p(r['return']) - 1.5*r['max_drawdown']) for r in results]
    return np.mean(values, axis=0) - .5*np.std(values, axis=0)
