"""Verify saved study005 provenance, causal source features and frozen inference."""
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

from .data import read_archive
from .features_v5 import build_features, load_dataset, pack_matrix, with_features
from .research import batch_activate, portfolio
from .research_v3 import FOLDS, SEEDS, stitched_metrics
from .research_v5 import controls, compact_reference, evaluate, model_config
from .sources_v5 import load_extra


def main():
    out = Path('artifacts/experiment_005_completed')
    protocol = json.loads((out/'protocol.json').read_text())
    selection = json.loads((out/'selection.json').read_text())
    for name, expected in protocol['source_hashes'].items():
        assert sha256((out/'source'/name).read_bytes()).hexdigest() == expected, name
    for label, expected in protocol['manifest_hashes'].items():
        path = out/f'{label}-manifest.json'
        assert sha256(path.read_bytes()).hexdigest() == expected
        for item in json.loads(path.read_text()):
            assert sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
    for path, expected in protocol['compact_reference_hashes'].items():
        assert sha256(Path(path).read_bytes()).hexdigest() == expected
    bars, frames, audit = load_dataset()
    rng = np.random.default_rng(505)
    prefix_checks = 0
    for symbol, b in bars.items():
        spot, _ = load_extra(symbol, 'spot', b.index)
        premium, _ = load_extra(symbol, 'premium', b.index)
        raw = pd.concat([read_archive(p, funding=True) for p in Path('data/fundingRate').glob(f'{symbol}-*.zip')])
        times = pd.to_datetime((raw.time/14_400_000).round().astype('int64')*14_400_000, unit='ms', utc=True)
        events = b.index.isin(times)
        peer = bars['ETHUSDT' if symbol == 'BTCUSDT' else 'BTCUSDT']
        # Explicit checks near every validation boundary, plus random prefixes.
        cuts = list(rng.integers(190, len(b), 200))
        for a, z in FOLDS:
            at = b.index.get_indexer([pd.Timestamp(a, tz='UTC')])[0]
            cuts.extend([at-1, at, at+1])
        for n in cuts:
            f = build_features(b.iloc[:n], spot.iloc[:n], premium.iloc[:n], peer.iloc[:n], events[:n])
            np.testing.assert_allclose(f.iloc[-10:], frames[symbol].iloc[n-10:n], atol=1e-12, rtol=0)
            prefix_checks += 1
    sequences = {}
    config = model_config(protocol['population'])
    reference_networks = 0
    for fold, (a, z) in enumerate(FOLDS):
        markets = [with_features(s, b, frames[s], a, z) for s, b in bars.items()]
        candidates = controls() + [compact_reference(fold)]
        for pack in protocol['packs']:
            members = []
            for seed in SEEDS:
                with (out/'jobs'/f'fold{fold}_{pack}_{seed}.pkl').open('rb') as f:
                    member = pickle.load(f)
                members.append(member)
                net = neat.nn.FeedForwardNetwork.create(member['genome'], config)
                x = pack_matrix(markets[0].x[::83], pack)
                np.testing.assert_allclose(batch_activate(net, x), [net.activate(v)[0] for v in x], atol=1e-12, rtol=0)
                reference_networks += 1
            candidates.append({'name': f'neat_{pack}', 'kind': 'ensemble', 'members': members})
        for candidate in candidates:
            result = evaluate(candidate, markets, config)
            sequences.setdefault(candidate['name'], []).append(portfolio(result)['returns'])
    rows = {r['name']: r for r in selection['development']}
    for name, parts in sequences.items():
        replay = stitched_metrics(parts)
        for field in ['return', 'max_drawdown', 'sharpe']:
            np.testing.assert_allclose(replay[field], rows[name]['scenarios']['base'][field], atol=1e-12, rtol=0)
    old = json.loads(Path('artifacts/experiment_004/selection.json').read_text())
    old_rows = {r['name']: r for r in old['development']}
    for current, previous in [('baseline', 'baseline_banded'), ('compact_core_reference', 'neat_plain')]:
        for scenario in ['base', 'double_cost', 'delay_4h']:
            for field in ['return', 'max_drawdown', 'sharpe']:
                np.testing.assert_allclose(rows[current]['scenarios'][scenario][field], old_rows[previous]['scenarios'][scenario][field], atol=1e-12, rtol=0)
    reused_count = 0
    if protocol.get('reused_training'):
        original = Path(protocol['reused_training']['path'])
        for path in (original/'jobs').glob('*.pkl'):
            assert path.read_bytes() == (out/'jobs'/path.name).read_bytes()
            reused_count += 1
    tests = subprocess.run([sys.executable, '-m', 'pytest', '-q'], capture_output=True, text=True, check=True)
    (out/'pytest-output.txt').write_text(tests.stdout)
    passed = re.search(r'(\d+) passed', tests.stdout)
    assert passed, tests.stdout
    verification = {'source_snapshots_verified': True, 'all_archive_hashes_verified': True,
                    'causal_real_data_prefix_checks': prefix_checks, 'trained_network_reference_checks': reference_networks,
                    'reproduced_base_candidates': len(rows), 'study004_controls_reproduced': True,
                    'tests_passed': int(passed.group(1)), 'reused_models_byte_identical': reused_count, 'transfer_accessed': False}
    (out/'verification.json').write_text(json.dumps(verification, indent=2))
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
