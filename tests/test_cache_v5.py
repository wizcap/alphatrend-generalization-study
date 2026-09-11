"""Interrupted expensive training cannot be reused across changed inputs/code."""
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from neattrade.features_v5 import PACKS
from neattrade.research_v3 import FOLDS, SEEDS
from neattrade.research_v5 import validate_model_cache


def fixture(tmp_path):
    cache = tmp_path/'cache'
    frozen = cache/'source/neattrade/research_v5.py'
    frozen.parent.mkdir(parents=True)
    frozen.write_bytes(Path('neattrade/research_v5.py').read_bytes())
    for folder in ['data', 'extra']:
        root = tmp_path/folder
        root.mkdir()
        (root/'manifest.json').write_text('[]')
    protocol = {'population': 64, 'generations': 30, 'packs': PACKS, 'seeds': SEEDS,
                'folds': FOLDS, 'band': .05,
                'source_hashes': {'neattrade/research_v5.py': sha256(frozen.read_bytes()).hexdigest()},
                'manifest_hashes': {'base': sha256(b'[]').hexdigest(), 'extra': sha256(b'[]').hexdigest()}}
    (cache/'protocol.json').write_text(json.dumps(protocol))
    args = SimpleNamespace(population=64, generations=30, data=str(tmp_path/'data'), extra=str(tmp_path/'extra'))
    return cache, frozen, args, protocol


def test_cached_training_rejects_different_budget_or_data(tmp_path):
    cache, frozen, args, protocol = fixture(tmp_path)
    assert validate_model_cache(cache, args)['path'] == str(cache)
    args.population = 65
    with pytest.raises(ValueError, match='protocol differs'):
        validate_model_cache(cache, args)
    args.population = 64
    Path(args.extra, 'manifest.json').write_text('[1]')
    with pytest.raises(ValueError, match='manifest differs'):
        validate_model_cache(cache, args)


def test_cached_training_rejects_changed_training_function(tmp_path):
    cache, frozen, args, protocol = fixture(tmp_path)
    frozen.write_text(frozen.read_text().replace('values[i] - .0001*len', 'values[i] - .001*len'))
    protocol['source_hashes']['neattrade/research_v5.py'] = sha256(frozen.read_bytes()).hexdigest()
    (cache/'protocol.json').write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match='function differs: train_one'):
        validate_model_cache(cache, args)
