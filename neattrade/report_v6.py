"""Tables and type-comparison figures for completed study006."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from .indicator_types import TYPES, LABELS
from .research_v6 import seed_compound

SHORT = ['Momentum', 'Trend', 'Vol. / range', 'Volume / flow', 'Funding', 'Basis', 'Interasset']


def main():
    run, out = Path('artifacts/experiment_006'), Path('reports')
    selection = json.loads((run/'selection.json').read_text(encoding='utf-8'))
    report = json.loads((run/'report.json').read_text(encoding='utf-8'))
    rows = {r['name']: r for r in selection['development']}
    comparison = {r['type']: r for r in report['contrasts']}
    records, type_rows, seed_rows, seed_folds = [], [], [], []
    for name, row in rows.items():
        records.append({'candidate': name, **row['scenarios']['base'],
                        **{s+'_return': row['scenarios'][s]['return'] for s in ['double_cost', 'delay_4h', 'source_lag_4h']},
                        'btc_return': row['asset_compounded_returns']['BTCUSDT'],
                        'eth_return': row['asset_compounded_returns']['ETHUSDT'],
                        'orders': sum(v['order_count'] for f in row['folds'] for v in f['by_symbol'].values()),
                        'mean_sleeve_turnover_per_fold': float(np.mean([v['turnover'] for f in row['folds'] for v in f['by_symbol'].values()])),
                        'mean_sleeve_abs_exposure': float(np.mean([v['mean_abs_exposure'] for f in row['folds'] for v in f['by_symbol'].values()])),
                        'absolute_pass': row['absolute_pass'], 'incremental_pass': row['incremental_neat_pass']})
    for group, features in TYPES.items():
        a, b = rows['only_'+group], rows['without_'+group]
        c = comparison[group]
        type_rows.append({'type': group, 'label': LABELS[group], 'input_count': len(features),
                          'only_return': a['scenarios']['base']['return'], 'only_drawdown': a['scenarios']['base']['max_drawdown'],
                          'only_double_cost_return': a['scenarios']['double_cost']['return'],
                          'only_source_lag_return': a['scenarios']['source_lag_4h']['return'],
                          'without_return': b['scenarios']['base']['return'], 'without_drawdown': b['scenarios']['base']['max_drawdown'],
                          'all_minus_without_pp': c['all_minus_without_pp'],
                          'only_minus_intercept_pp': c['only_minus_intercept_pp'],
                          'only_positive_seed_excess_count': sum(v>0 for v in c['only_paired_seed_excess']),
                          'all_positive_seed_minus_without_count': sum(v>0 for v in c['all_paired_seed_minus_without'])})
    for row in selection['per_seed']:
        seed_folds.append({k: v for k, v in row.items() if k not in ['portfolio', 'by_symbol']} | row['portfolio'])
    for spec in sorted({r['spec'] for r in selection['per_seed']}):
        for seed in [11, 29, 47]:
            parts = [r for r in seed_folds if r['spec']==spec and r['seed']==seed]
            seed_rows.append({'spec': spec, 'seed': seed, 'return': seed_compound(selection['per_seed'], spec, seed),
                              'positive_folds': sum(r['return']>0 for r in parts),
                              'mean_connections': float(np.mean([r['connections'] for r in parts]))})
    pd.DataFrame(records).to_csv(out/'study006_metrics.csv', index=False)
    pd.DataFrame(type_rows).to_csv(out/'study006_types.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(seed_rows).to_csv(out/'study006_seeds.csv', index=False)
    pd.DataFrame(seed_folds).to_csv(out/'study006_seed_folds.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), gridspec_kw={'height_ratios': [1, 1.1]})
    pos = np.arange(7)
    ax = axes[0, 0]
    ax.bar(pos-.17, [r['only_return'] for r in type_rows], .33, color='#167e88', label='Base costs')
    ax.bar(pos+.17, [r['only_double_cost_return'] for r in type_rows], .33, color='#86bcc2', label='Double costs')
    ax.axhline(rows['baseline']['scenarios']['base']['return'], color='#a57e23', ls='--', label='Full baseline')
    ax.axhline(rows['intercept']['scenarios']['base']['return'], color='#536478', ls=':', label='Constant-only gate')
    ax.set_xticks(pos, SHORT, rotation=20, ha='right')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title('NEAT sees only one indicator type', loc='left', fontweight='bold')
    ax.legend(fontsize=8, ncols=2, frameon=False)
    ax = axes[0, 1]
    effect = [r['all_minus_without_pp'] for r in type_rows]
    ax.barh(pos, effect, color=['#167e88' if e>0 else '#c77560' for e in effect], alpha=.8)
    for i, group in enumerate(TYPES):
        ax.scatter(np.array(comparison[group]['all_paired_seed_minus_without'])*100, [i-.15, i, i+.15], color='#3e4248', s=15, zorder=3)
    ax.axvline(0, color='#888', linewidth=.8)
    ax.set_yticks(pos, SHORT)
    ax.invert_yaxis()
    ax.set_xlabel('All minus retrained-without (percentage points)')
    ax.set_title('Does retaining this type help the full model?', loc='left', fontweight='bold')
    ax.text(.01, -.22, 'Positive: retaining helps. Dots: three seed differences, not a confidence interval.', transform=ax.transAxes, fontsize=8)
    ax = axes[1, 0]
    names = ['baseline', 'intercept']+['only_'+g for g in TYPES]
    values = np.array([[f['portfolio']['return'] for f in rows[n]['folds']] for n in names])
    limit = max(.001, np.max(np.abs(values)))
    ax.imshow(values, cmap='RdYlGn', vmin=-limit, vmax=limit, aspect='auto')
    ax.set_yticks(range(len(names)), ['Baseline', 'Constant']+SHORT)
    ax.set_xticks(range(4), ['2024 H1', '2024 H2', '2025 H1', '2025 H2'])
    for i in range(len(names)):
        for j in range(4):
            ax.text(j, i, f'{values[i,j]:+.1%}', ha='center', va='center', fontsize=9)
    ax.set_title('Only-type returns by half-year', loc='left', fontsize=12)
    ax = axes[1, 1]
    names = ['baseline', 'rule_funding', 'rule_premium', 'rule_both']
    pp = np.arange(4)
    ax.bar(pp-.17, [rows[n]['scenarios']['base']['return'] for n in names], .33, color='#167e88', label='Net return')
    ax.bar(pp+.17, [rows[n]['scenarios']['base']['max_drawdown'] for n in names], .33, color='#cda466', label='Max. drawdown')
    ax.set_xticks(pp, ['Baseline', 'Funding rule', 'Premium rule', 'Either rule'])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False, fontsize=9)
    ax.set_title('Separating the fixed funding / premium rule', loc='left', fontsize=12)
    for ax in [axes[0,0], axes[1,1]]:
        ax.axhline(0, color='#888', linewidth=.7)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    for ax in [*axes[0], axes[1,1]]:
        ax.spines[['right', 'top']].set_visible(False)
    fig.suptitle('Indicator types: standalone gates and retrained removals', fontsize=17, fontweight='bold', x=.055, ha='left')
    fig.text(.055, .018, '2024-2025 observed development | Common trend/risk layer retained | Costs + funding included | Different input counts change search dynamics', fontsize=9)
    fig.tight_layout(rect=[0, .045, 1, .945], h_pad=3)
    fig.savefig(out/'study006_type_comparison.png', dpi=160)
    plt.close(fig)
    correlations = pd.read_csv(run/'training_feature_correlations.csv', index_col=0)
    block = np.zeros((7, 7))
    for i, first in enumerate(TYPES.values()):
        for j, second in enumerate(TYPES.values()):
            block[i, j] = correlations.loc[first, second].abs().to_numpy().max() if i != j else 1.
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(block, cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(pos, SHORT, rotation=30, ha='right')
    ax.set_yticks(pos, SHORT)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f'{block[i,j]:.2f}', ha='center', va='center', color='white' if block[i,j]>.75 else '#333')
    fig.colorbar(im, ax=ax, shrink=.8)
    ax.set_title('Type overlap: largest absolute feature-pair correlation\nTraining data only: Jan 2022 through Dec 24, 2023', loc='left')
    fig.tight_layout()
    fig.savefig(out/'study006_training_overlap.png', dpi=160)
    plt.close(fig)
    print(pd.DataFrame(type_rows).to_string(index=False))
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == '__main__':
    main()
