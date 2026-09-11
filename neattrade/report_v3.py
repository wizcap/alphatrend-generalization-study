"""Summarize frozen study 003; no model selection or training."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


def main():
    run = Path('artifacts/experiment_003')
    out = Path('reports')
    d = json.loads((run/'selection.json').read_text())
    rows = {r['name']: r for r in d['development']}
    curves = pd.read_csv(run/'development_equity.csv', index_col=0, parse_dates=True)
    series = [('neat_growth', 'NEAT: growth objective', '#087f8c'),
              ('neat_legacy', 'NEAT: legacy objective', '#d1495b'),
              ('daily_trend_180_long_only', 'Slow trend long-only baseline', '#be8a20')]
    fig = plt.figure(figsize=(13, 9))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.65, 1])
    top = fig.add_subplot(grid[0, :])
    for key, label, color in series:
        top.plot(curves.index, curves[key]*100, label=label, color=color, linewidth=1.8)
    top.axhline(100, color='#a1a1aa', linestyle='--', linewidth=1)
    top.set_ylabel('Equity (initial capital = 100)')
    top.set_title('Paired walk-forward development | 2024-2025', loc='left', fontweight='bold')
    top.legend(loc='lower left', frameon=False, fontsize=9)
    left, right = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    for i, (key, label, color) in enumerate(series):
        left.bar(np.arange(4)+(i-1)*.24, [f['portfolio']['return'] for f in rows[key]['folds']], width=.23, color=color)
        right.bar(np.arange(3)+(i-1)*.24, [rows[key]['scenarios'][s]['return'] for s in ['base', 'double_cost', 'delay_4h']], width=.23, color=color)
    left.set_xticks(range(4), ['2024 H1', '2024 H2', '2025 H1', '2025 H2'])
    left.set_title('Returns by validation fold', loc='left', fontsize=11)
    right.set_xticks(range(3), ['Base cost', 'Double cost', '4h delay'])
    right.set_title('Compounded development returns', loc='left', fontsize=11)
    for ax in [left, right]:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.axhline(0, color='#777', linewidth=.7)
    for ax in [top, left, right]:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle('Fitness improved; an investable NEAT edge remains unproven', fontsize=16, fontweight='bold', x=.075, ha='left')
    fig.text(.075, .023, 'Previously inspected BTC/ETH development data | Actual funding, 5 bps fee + 2 bps slippage | No SOL/BNB transfer data accessed', fontsize=9)
    fig.tight_layout(rect=[0, .055, 1, .96])
    fig.savefig(out/'study003_comparison.png', dpi=160)
    plt.close(fig)
    table = []
    for r in d['development']:
        table.append({'candidate': r['name'], **r['scenarios']['base'],
                      'double_cost_return': r['scenarios']['double_cost']['return'],
                      'delayed_return': r['scenarios']['delay_4h']['return'],
                      'positive_folds': sum(f['portfolio']['return'] > 0 for f in r['folds']),
                      'development_pass': r['development_pass']})
    pd.DataFrame(table).to_csv(out/'study003_metrics.csv', index=False)
    seed_summary = []
    for objective in ['legacy', 'growth']:
        for seed in [11, 29, 47]:
            data = [r for r in d['per_seed_diagnostics'] if r['objective'] == objective and r['seed'] == seed]
            seed_summary.append({'objective': objective, 'seed': seed,
                                 'compounded_fold_return': float(np.prod([1+r['portfolio']['return'] for r in data])-1),
                                 'train_cash_folds': sum(r['train_cash'] for r in data)})
    pd.DataFrame(seed_summary).to_csv(out/'study003_seeds.csv', index=False)
    print(json.dumps(seed_summary, indent=2))


if __name__ == '__main__':
    main()
