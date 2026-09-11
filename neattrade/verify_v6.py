"""Verify frozen taxonomy, model hashes, actual genomes and saved study006 results."""
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

from .batch import fitness_accounting, prepare
from .features_v5 import load_dataset, with_features
from .indicator_types import select_inputs, specifications
from .overlay import overlay_targets
from .research import batch_activate, portfolio
from .research_v3 import FOLDS, SEEDS, stitched_metrics
from .research_v5 import model_config
from .research_v6 import controls, evaluate, write_json


def main():
    out = Path('artifacts/experiment_006')
    protocol = json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    selection = json.loads((out/'selection.json').read_text(encoding='utf-8'))
    digest = sha256((out/'protocol.json').read_bytes()).hexdigest()
    assert protocol['specifications'] == specifications()
    for name, expected in protocol['source_hashes'].items():
        assert sha256((out/'source'/name).read_bytes()).hexdigest() == expected
    for label, expected in protocol['manifest_hashes'].items():
        path = out/f'{label}-manifest.json'
        assert sha256(path.read_bytes()).hexdigest() == expected
        for entry in json.loads(path.read_text()):
            assert sha256(Path(entry['path']).read_bytes()).hexdigest() == entry['sha256']
    bars, frames, _ = load_dataset()
    seq, checked, accounting = {}, 0, 0
    seed_records = {(r['fold'], r['spec'], r['seed']): r for r in selection['per_seed']}
    for fold, (a, z) in enumerate(FOLDS):
        markets = [with_features(s, b, frames[s], a, z) for s, b in bars.items()]
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
                    fast = fitness_accounting(d, overlay_targets(reference[None, :], d.x, 'plain'), rebalance_band=.05)
                    for field in ['return', 'max_drawdown', 'funding_paid_initial_equity', 'cost_paid_initial_equity']:
                        np.testing.assert_allclose(fast[field][0], results[market.symbol][field], atol=2e-12, rtol=0)
                        np.testing.assert_allclose(results[market.symbol][field], record['by_symbol'][market.symbol][field], atol=2e-12, rtol=0)
                    accounting += 1
                checked += 1
            candidates.append({'name': spec['name'], 'kind': 'ensemble', 'members': members})
        for candidate in candidates:
            r = evaluate(candidate, markets)
            seq.setdefault(candidate['name'], []).append(portfolio(r)['returns'])
    rows = {r['name']: r for r in selection['development']}
    for name, parts in seq.items():
        replay = stitched_metrics(parts)
        for field in ['return', 'max_drawdown', 'sharpe']:
            np.testing.assert_allclose(replay[field], rows[name]['scenarios']['base'][field], atol=2e-12, rtol=0)
    previous = json.loads(Path('artifacts/experiment_005_completed/selection.json').read_text())
    old = {r['name']: r for r in previous['development']}
    for new, prior in [('all', 'neat_combined'), ('baseline', 'baseline'), ('rule_both', 'rule_derivatives')]:
        for scenario in ['base', 'double_cost', 'delay_4h', 'source_lag_4h']:
            for field in ['return', 'max_drawdown', 'sharpe']:
                np.testing.assert_allclose(rows[new]['scenarios'][scenario][field], old[prior]['scenarios'][scenario][field], atol=2e-12, rtol=0)
    months = pd.read_csv(out/'monthly_returns.csv', index_col=0)
    for name in rows:
        np.testing.assert_allclose(np.prod(1+months[name])-1, rows[name]['scenarios']['base']['return'], atol=2e-12, rtol=0)
    test = subprocess.run([sys.executable, '-m', 'pytest', '-q'], capture_output=True, text=True, check=True)
    (out/'pytest-output.txt').write_text(test.stdout)
    passed = re.search(r'(\d+) passed', test.stdout)
    assert passed, test.stdout
    result = {'tests_passed': int(passed.group(1)), 'frozen_sources_and_archives_verified': True,
              'trained_model_hashes_and_reference_networks': checked, 'reference_accounting_checks': accounting,
              'replayed_candidates': len(rows), 'prior_study005_controls_match_all_scenarios': True,
              'monthly_compounding_verified': True, 'transfer_accessed': False}
    write_json(out/'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
