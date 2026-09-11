"""Verify recorded model inference/accounting after defensive code changes.

This replays previously observed periods without selecting or training anything.
Only use with trusted, locally generated model pickle files.
"""
import json
from hashlib import sha256
from pathlib import Path
import pickle

import numpy as np

from .data import load
from .market import make_market
from . import research, research_v2


def main():
    bars = {s: load(s) for s in research.SYMBOLS}
    findings = []
    for name, module in [('experiment_001', research), ('experiment_002', research_v2)]:
        path = Path('artifacts', name)
        protocol = json.loads((path/'protocol.json').read_text())
        expected = json.loads((path/'report.json').read_text())
        with (path/'models.pkl').open('rb') as f:
            saved = pickle.load(f)
        markets = [make_market(s, b, *protocol['test']) for s, b in bars.items()]
        scenarios = [('base', {}), ('delay_4h', {'delay': 1}), ('no_funding', {'no_funding': True})]
        scenarios.append(('double_cost', {'fee_bps': 10, 'slippage_bps': 4} if module is research else {'multiplier': 2}))
        for label, options in scenarios:
            result = module.evaluate(saved['selected'], markets, saved['config'], **options)
            for symbol, actual in result.items():
                for key, value in expected['test'][label]['by_symbol'][symbol].items():
                    np.testing.assert_allclose(actual[key], value, rtol=1e-12, atol=1e-12)
        for candidate in saved['candidates']:
            if candidate['name'] not in expected['comparators']:
                continue
            result = module.evaluate(candidate, markets, saved['config'])
            for symbol, actual in result.items():
                for key, value in expected['comparators'][candidate['name']]['by_symbol'][symbol].items():
                    np.testing.assert_allclose(actual[key], value, rtol=1e-12, atol=1e-12)
        for file, expected_hash in protocol['source_hashes'].items():
            assert sha256((path/'source'/file).read_bytes()).hexdigest() == expected_hash, file
        assert sha256((path/'data-manifest.json').read_bytes()).hexdigest() == protocol['manifest_hash']
        findings.append({'study': name, 'recorded_metrics_reproduced': True,
                         'frozen_source_hashes_verified': True, 'frozen_manifest_hash_verified': True})
    Path('artifacts/verification.json').write_text(json.dumps(findings, indent=2))
    print(json.dumps(findings, indent=2))


if __name__ == '__main__':
    main()
