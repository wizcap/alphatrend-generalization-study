"""Independent AlphaTrend formula loop, accounting, network and result replay."""
from hashlib import sha256
import json
from pathlib import Path
import pickle
import re
import subprocess
import sys

import neat
import numpy as np
import pandas as pd

from .alphatrend import alpha_indicator, select_inputs, specifications, gated_targets
from .batch import fitness_accounting, prepare
from .features_v5 import with_features
from .research import batch_activate, portfolio
from .research_v3 import FOLDS, SEEDS, stitched_metrics
from .research_v5 import model_config
from .research_v6 import SCENARIOS, write_json
from .research_v7 import controls, dataset, evaluate


def independent_alpha(b, period=14):
    """Scalar reference from author recurrence, without pandas rolling/shift."""
    h, l, c, v = (b[name].to_numpy() for name in ['high', 'low', 'close', 'volume'])
    price = (h+l+c)/3
    positive, negative, tr, lines, states = [], [], [], [], []
    state = 0.
    for t in range(len(b)):
        positive.append(price[t]*v[t] if t == 0 or price[t] > price[t-1] else 0.)
        negative.append(price[t]*v[t] if t == 0 or price[t] < price[t-1] else 0.)
        tr.append(max(h[t]-l[t], abs(h[t]-c[t-1]), abs(l[t]-c[t-1])) if t else np.nan)
        if t < period:
            line = np.nan
        else:
            up, down = sum(positive[t-period+1:t+1]), sum(negative[t-period+1:t+1])
            momentum = 100*up/(up+down) if up+down else np.nan
            atr = sum(tr[t-period+1:t+1])/period
            previous = lines[-1] if np.isfinite(lines[-1]) else 0.
            line = max(previous, l[t]-atr) if momentum >= 50 else min(previous, h[t]+atr)
        lines.append(line)
        if t >= 3:
            if line > lines[t-2] and lines[t-1] <= lines[t-3]:
                state = 1.
            elif line < lines[t-2] and lines[t-1] >= lines[t-3]:
                state = -1.
        states.append(state)
    return np.asarray(lines), np.asarray(states)


def main():
    out = Path('artifacts/experiment_007')
    protocol = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    selection = json.loads((out/'selection.json').read_text(encoding='utf-8'))
    digest = sha256((out/'protocol.json').read_bytes()).hexdigest()
    assert protocol['specifications'] == specifications()
    for name, expected in protocol['source_hashes'].items():
        assert sha256((out/'source'/name).read_bytes()).hexdigest() == expected
        assert sha256(Path(name).read_bytes()).hexdigest() == expected
    for label, expected in protocol['manifest_hashes'].items():
        path = out/f'{label}-manifest.json'
        assert sha256(path.read_bytes()).hexdigest() == expected
        for entry in json.loads(path.read_text()):
            assert sha256(Path(entry['path']).read_bytes()).hexdigest() == entry['sha256']
    bars, frames, late, _ = dataset()
    formula_bars = 0
    for symbol, b in bars.items():
        # Only score the frozen development range; no new2026 result inspection.
        b = b.loc[:'2025-12-31 20:00']
        line, state = independent_alpha(b)
        reference = alpha_indicator(b)
        np.testing.assert_allclose(reference.line, line, atol=1e-9, rtol=0, equal_nan=True)
        np.testing.assert_array_equal(reference.state, state)
        formula_bars += len(b)
    sequences, checked, accounting = {}, 0, 0
    seed_records = {(r['fold'], r['spec'], r['seed']): r for r in selection['per_seed']}
    for fold, (a, z) in enumerate(FOLDS):
        markets = [with_features(s, b, frames[s], a, z) for s, b in bars.items()]
        delayed = [with_features(s, b, late[s], a, z) for s, b in bars.items()]
        candidates = controls()
        for spec in specifications():
            members = []
            for seed in SEEDS:
                path = out/'jobs'/f'fold{fold}_{spec["name"]}_{seed}.pkl'
                meta = json.loads(path.with_suffix('.json').read_text())
                assert meta['model_sha256'] == sha256(path.read_bytes()).hexdigest()
                assert meta['protocol_sha256'] == digest
                with path.open('rb') as f:
                    member = pickle.load(f)
                assert member['spec'] == spec and member['seed'] == seed
                assert pd.Timestamp(member['train_end']) <= pd.Timestamp(a, tz='UTC')-pd.Timedelta(days=7)
                members.append(member)
                results = evaluate(member, markets)
                record = seed_records[fold, spec['name'], seed]
                for market in markets:
                    d = prepare(market)
                    x = select_inputs(d.x, spec)
                    c = model_config(inputs=x.shape[1])
                    net = neat.nn.FeedForwardNetwork.create(member['genome'], c)
                    reference = np.array([net.activate(v)[0] for v in x])
                    np.testing.assert_allclose(batch_activate(net, x), reference, atol=1e-12, rtol=0)
                    fast = fitness_accounting(d, gated_targets(reference[None, :], d.x), rebalance_band=.05)
                    for field in ['return', 'max_drawdown', 'funding_paid_initial_equity', 'cost_paid_initial_equity']:
                        np.testing.assert_allclose(fast[field][0], results[market.symbol][field], atol=2e-12, rtol=0)
                        np.testing.assert_allclose(results[market.symbol][field], record['by_symbol'][market.symbol][field], atol=2e-12, rtol=0)
                    accounting += 1
                checked += 1
            candidates.append({'name': spec['name'], 'kind': 'ensemble', 'members': members})
        for candidate in candidates:
            for scenario, options in SCENARIOS:
                opts = dict(options)
                use = delayed if opts.pop('lagged', False) else markets
                r = evaluate(candidate, use, **opts)
                sequences.setdefault((candidate['name'], scenario), []).append(portfolio(r)['returns'])
    rows = {r['name']: r for r in selection['development']}
    for (name, scenario), parts in sequences.items():
        replay = stitched_metrics(parts)
        for field in ['return', 'max_drawdown', 'sharpe']:
            np.testing.assert_allclose(replay[field], rows[name]['scenarios'][scenario][field], atol=2e-12, rtol=0)
    old = {r['name']: r for r in json.loads(Path('artifacts/experiment_006/selection.json').read_text())['development']}
    for new, prior in [('baseline', 'baseline'), ('ema_premium', 'rule_premium')]:
        for scenario, _ in SCENARIOS:
            for field in ['return', 'max_drawdown', 'sharpe']:
                np.testing.assert_allclose(rows[new]['scenarios'][scenario][field], old[prior]['scenarios'][scenario][field], atol=2e-12, rtol=0)
    months = pd.read_csv(out/'monthly_returns.csv', index_col=0)
    curves = pd.read_csv(out/'development_equity.csv', index_col=0)
    for name in rows:
        np.testing.assert_allclose(np.prod(1+months[name])-1, rows[name]['scenarios']['base']['return'], atol=2e-12, rtol=0)
        np.testing.assert_allclose(curves[name], stitched_metrics(sequences[name, 'base'])['equity'], atol=2e-12, rtol=0)
    test = subprocess.run([sys.executable, '-m', 'pytest', '-q'], capture_output=True, text=True, check=True)
    (out/'pytest-output.txt').write_text(test.stdout)
    passed = re.search(r'(\d+) passed', test.stdout)
    assert passed, test.stdout
    result = {'tests_passed': int(passed.group(1)), 'frozen_sources_and_archives_verified': True,
              'independent_formula_bars': formula_bars, 'formula_states_match': True,
              'pine_runtime_parity_tested': False,
              'trained_model_hashes_and_reference_networks': checked, 'reference_accounting_checks': accounting,
              'replayed_candidate_scenarios': len(sequences), 'prior_study006_controls_match_all_scenarios': True,
              'monthly_compounding_and_full_curves_verified': True, 'transfer_accessed': False}
    write_json(out/'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
