"""Causal long-only NEAT confidence overlay on a fixed slow-trend baseline."""
import numpy as np


def baseline_targets(x):
    return (x[:,5]>0)*np.minimum(.5,.20/np.maximum(x[:,6],.10))


def confidence(raw, trend_positive, mode):
    """Rows are independent policies, columns are chronological daily decisions.

    Negative slow trend always forces exit and resets state. Hysteresis avoids
    repeated changes near a single threshold; it never delays a strong exit.
    """
    r = np.asarray(raw,dtype=float)
    trend = np.asarray(trend_positive,dtype=bool)
    if r.ndim!=2 or r.shape[1]!=len(trend) or not np.isfinite(r).all():
        raise ValueError('Expected finite population x chronological decisions')
    if mode=='plain':
        return np.where(r<-.35,0.,np.where(r>.35,1.,.5))*trend
    if mode!='hysteretic':
        raise ValueError('Unknown overlay mode')
    state = np.zeros(len(r))
    result = np.zeros_like(r)
    for j in range(r.shape[1]):
        value=r[:,j]
        old=state.copy()
        state=np.where(value>=.5,1.,np.where(value<=-.5,0.,state))
        state=np.where((old==0)&(value>=.15)&(value<.5),.5,state)
        state=np.where((old==1)&(value<=-.15)&(value>-.5),.5,state)
        if not trend[j]:
            state[:]=0
        result[:,j]=state
    return result


def overlay_targets(raw,x,mode):
    return confidence(raw,x[:,5]>0,mode)*baseline_targets(x)


def incremental_objective(results,benchmarks):
    """Reward consistent annualized excess net growth over a matched baseline.

    The baseline shares sizing, execution band and costs. No activity reward.
    Cash can outperform a losing baseline but receives no special treatment.
    """
    values,weights=[],[]
    for r,b in zip(results,benchmarks):
        excess=(np.log1p(r['return'])-np.log1p(b['return'][0]))/r['years']
        excess-=2*np.maximum(r['max_drawdown']-.20,0)
        values.append(np.where(r['ruined'],-100.,excess))
        weights.append(r['years'])
    mean=np.average(values,axis=0,weights=weights)
    deviation=np.sqrt(np.average((np.asarray(values)-mean)**2,axis=0,weights=weights))
    return mean-.5*deviation
