"""Render study005 results without further fitting or selection."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


def main():
    out = Path('reports')
    run = Path('artifacts/experiment_005_completed')
    selection = json.loads((run/'selection.json').read_text())
    report = json.loads((run/'report.json').read_text())
    records = selection['development']
    rows = {r['name']: r for r in records}
    table = []
    for r in records:
        name = r['name']
        base = r['scenarios']['base']
        table.append({'candidate': name, **base, 'btc_return': r['asset_compounded_returns']['BTCUSDT'],
                      'eth_return': r['asset_compounded_returns']['ETHUSDT'],
                      **{s+'_return': r['scenarios'][s]['return'] for s in ['double_cost', 'delay_4h', 'source_lag_4h']},
                      'known2026_return': report['known2026_context'].get(name, {}).get('portfolio', {}).get('return'),
                      'absolute_pass': r['absolute_pass'], 'incremental_neat_pass': r['incremental_neat_pass'],
                      'added_source_screen': r.get('added_source_screen', False)})
    pd.DataFrame(table).to_csv(out/'study005_metrics.csv', index=False)
    seed_folds, seeds = [], []
    for r in selection['per_seed']:
        seed_folds.append({k: v for k, v in r.items() if k not in ['portfolio', 'by_symbol']} | r['portfolio'])
    pd.DataFrame(seed_folds).to_csv(out/'study005_seed_folds.csv', index=False)
    for pack in ['core', 'technical', 'derivatives', 'cross', 'combined']:
        for seed in [11, 29, 47]:
            parts = [r for r in seed_folds if r['pack'] == pack and r['seed'] == seed]
            seeds.append({'pack': pack, 'seed': seed, 'return': float(np.prod([1+r['return'] for r in parts])-1),
                          'positive_folds': sum(r['return'] > 0 for r in parts),
                          'mean_enabled_from_masked_inputs': float(np.mean([r['enabled_from_masked_inputs'] for r in parts]))})
    pd.DataFrame(seeds).to_csv(out/'study005_seeds.csv', index=False)
    ci = [{'pack': r['name'], **r['paired_core_bootstrap']} for r in records if 'paired_core_bootstrap' in r]
    (out/'study005_uncertainty.json').write_text(json.dumps(ci, indent=2))
    series = [('baseline', 'Slow-trend baseline', '#b08b20'), ('rule_derivatives', 'Funding / premium rule', '#549747'), ('neat_core', 'NEAT core', '#708090'),
              ('neat_technical', '+ Technical', '#8874bc'), ('neat_derivatives', '+ Derivatives', '#087f8c'),
              ('neat_cross', '+ Spot / cross-market', '#d56655'), ('neat_combined', 'All groups', '#243f6b')]
    curves = pd.read_csv(run/'development_equity.csv', index_col=0, parse_dates=True)
    recent = pd.read_csv(run/'known2026_equity.csv', index_col=0, parse_dates=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), gridspec_kw={'height_ratios': [1.6, 1]})
    for key, label, color in series:
        axes[0, 0].plot(curves.index, curves[key]*100, label=label, color=color, linewidth=1.5)
        axes[0, 1].plot(recent.index, recent[key]*100, label=label, color=color, linewidth=1.5)
    for ax in axes[0]:
        ax.axhline(100, color='#888', linestyle='--', linewidth=.7)
        ax.legend(loc='upper left', frameon=False, fontsize=8, ncols=2)
        ax.tick_params(axis='x', rotation=20)
    axes[0, 0].set_title('2024-2025 rolling development', loc='left', fontweight='bold')
    axes[0, 1].set_title('Jan-Aug 2026: previously observed context', loc='left', fontweight='bold')
    axes[0, 0].set_ylabel('Equity (initial = 100)')
    keys = [v[0] for v in series]
    labels = ['Baseline', 'Carry rule', 'Core', '+Tech', '+Deriv.', '+Cross', 'All']
    values = np.array([[v['portfolio']['return'] for v in rows[k]['folds']] for k in keys])
    bound = max(.001, np.max(np.abs(values)))
    axes[1, 0].imshow(values, cmap='RdYlGn', vmin=-bound, vmax=bound, aspect='auto')
    axes[1, 0].set_yticks(range(len(keys)), labels)
    axes[1, 0].set_xticks(range(4), ['2024 H1', '2024 H2', '2025 H1', '2025 H2'])
    for i in range(len(keys)):
        for j in range(4):
            axes[1, 0].text(j, i, f'{values[i, j]:+.1%}', ha='center', va='center', fontsize=9)
    axes[1, 0].set_title('Paired half-year returns', loc='left', fontsize=11)
    xpos = np.arange(len(keys))
    for offset, scenario, label, color in [(-.25, 'base', 'Base', '#879db6'), (0, 'double_cost', '2x cost', '#ce9d6c'),
                                         (.25, 'source_lag_4h', 'External data +4h old', '#538b82')]:
        axes[1, 1].bar(xpos+offset, [rows[k]['scenarios'][scenario]['return'] for k in keys], width=.24, label=label, color=color)
    axes[1, 1].set_xticks(xpos, labels)
    axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1, 1].legend(fontsize=7, ncols=2, frameon=False)
    axes[1, 1].set_title('Development stress comparisons', loc='left', fontsize=11)
    axes[1, 1].axhline(0, color='#888', linewidth=.7)
    for ax in [*axes[0], axes[1, 1]]:
        ax.spines[['right', 'top']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('Which indicators and data sources add value?', fontsize=17, fontweight='bold', x=.055, ha='left')
    fig.text(.055, .021, 'Fixed seeds / budgets | Official verified Binance archives | Fees, slippage and funding included | All periods are known development/context', fontsize=8)
    fig.tight_layout(rect=[0, .045, 1, .95])
    fig.savefig(out/'study005_comparison.png', dpi=160)
    plt.close(fig)
    print(pd.DataFrame(table).to_string(index=False))
    print('Seed results:', json.dumps(seeds))
    print('Inference group-zero sensitivity:', json.dumps(report['inference_zero_group_sensitivity']))


if __name__ == '__main__':
    main()
