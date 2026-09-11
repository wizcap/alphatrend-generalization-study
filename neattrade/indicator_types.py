"""Study006's fixed, exhaustive indicator taxonomy and compact input selection."""
import numpy as np

from .features_v5 import ALL_FEATURES

TYPES = {
    'momentum': ['ret1', 'ret6', 'ret42', 'ret180', 'rsi14', 'macd_hist'],
    'trend': ['trend42', 'trend180', 'adx14'],
    'volatility': ['vol', 'range', 'bollinger42', 'atr14_relative'],
    'flow': ['volume_z', 'buy_fraction', 'spot_buy1d', 'spot_relative_volume'],
    'funding': ['funding_last_bps', 'funding7d'],
    'basis': ['premium1d_bps', 'premium_z', 'mark_basis_bps', 'spot_basis1d_bps', 'spot_basis_z'],
    'interasset': ['relative_momentum7d', 'peer_trend180'],
}
LABELS = {'momentum': '動能', 'trend': '趨勢與強度', 'volatility': '波動與區間位置',
          'flow': '成交量與買賣流', 'funding': '資金費率', 'basis': '溢價與價差', 'interasset': '跨幣種'}


def specifications():
    flat = [n for names in TYPES.values() for n in names]
    if len(flat) != len(set(flat)) or set(flat) != set(ALL_FEATURES):
        raise ValueError('Indicator types must partition every feature exactly once')
    result = [{'name': 'all', 'mode': 'all', 'type': None, 'features': list(ALL_FEATURES)},
              {'name': 'intercept', 'mode': 'intercept', 'type': None, 'features': []}]
    for mode in ['only', 'without']:
        for group, names in TYPES.items():
            wanted = set(names) if mode == 'only' else set(ALL_FEATURES)-set(names)
            result.append({'name': f'{mode}_{group}', 'mode': mode, 'type': group,
                           'features': [n for n in ALL_FEATURES if n in wanted]})
    return result


def select_inputs(x, spec):
    if x.ndim != 2 or x.shape[1] != len(ALL_FEATURES):
        raise ValueError('Expected the complete ordered feature matrix')
    if spec['mode'] == 'intercept':
        return np.ones((len(x), 1))
    if not spec['features'] or len(spec['features']) != len(set(spec['features'])):
        raise ValueError('Empty or repeated input features')
    return x[:, [ALL_FEATURES.index(n) for n in spec['features']]]


def fixed_risk_scale(x, rule):
    """Separately test both sides of the already declared funding/premium rule."""
    funding = x[:, ALL_FEATURES.index('funding_last_bps')] > 3.
    premium = x[:, ALL_FEATURES.index('premium1d_bps')] >= 5.
    if rule not in ['funding', 'premium', 'both']:
        raise ValueError('Unknown fixed rule')
    condition = {'funding': funding, 'premium': premium, 'both': funding | premium}[rule]
    return np.where(condition, .5, 1.)
