"""Post-hoc cost diagnostics of frozen study 003 models; no selection."""
import json
from pathlib import Path
import pickle

from .data import load
from .market import make_market, summary
from .research import config, portfolio
from .research_v3 import FOLDS, SEEDS, evaluate, stitched_metrics


def main():
    out = Path('artifacts/experiment_003')
    bars = {s: load(s) for s in ('BTCUSDT', 'ETHUSDT')}
    c = config(64)
    sequences, fold_rows = {}, []
    for fold, (a, z) in enumerate(FOLDS):
        members = []
        for seed in SEEDS:
            with (out/'jobs'/f'fold{fold}_growth_{seed}.pkl').open('rb') as f:
                members.append(pickle.load(f))
        candidate = {'kind': 'ensemble', 'members': members}
        markets = [make_market(s, b, a, z) for s, b in bars.items()]
        for multiplier in [0., .5, 1., 1.5, 2.]:
            result = portfolio(evaluate(candidate, markets, c, multiplier=multiplier))
            sequences.setdefault(multiplier, []).append(result['returns'])
            fold_rows.append({'fold': fold, 'cost_multiplier': multiplier, **summary(result)})
    ladder = [{'cost_multiplier': k, **summary(stitched_metrics(v))} for k, v in sequences.items()]
    report = {'label': 'Post-hoc diagnostic of frozen models; not an additional selection trial',
              'actual_funding_in_all_cases': True, 'cost_ladder': ladder, 'folds': fold_rows}
    (out/'diagnostics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'cost_ladder': ladder, 'final_fold_zero_cost': next(r for r in fold_rows if r['fold']==3 and r['cost_multiplier']==0)}, indent=2))


if __name__ == '__main__':
    main()
