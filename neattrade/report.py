"""Render saved experiment metrics without training, inference, or reselection."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd


def main():
    out = Path('reports')
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), gridspec_kw={'height_ratios': [2, 1]}, sharex='col')
    colors = ['#087f8c', '#d1495b', '#6b7280']
    for col, (study, ensemble, baseline, title) in enumerate([
        ('experiment_001', 'neat_ensemble', 'long_50pct', 'Study 001 | 2025 holdout'),
        ('experiment_002', 'neat_filter_ensemble', 'weekly_long_50pct', 'Study 002 | Jan-Aug 2026 holdout'),
    ]):
        frame = pd.read_csv(Path('artifacts', study, 'equity.csv'), index_col=0, parse_dates=True)
        for key, label, color in zip(['base', ensemble, baseline], ['Selected policy: cash', 'NEAT ensemble', '50% long comparator'], colors):
            curve = frame[key]
            axes[0, col].plot(frame.index, curve * 100, color=color, label=label, linewidth=1.8,
                              linestyle='--' if key == ensemble and col == 1 else '-')
            axes[1, col].plot(frame.index, curve / curve.cummax() - 1, color=color, linewidth=1.4,
                              linestyle='--' if key == ensemble and col == 1 else '-')
        axes[0, col].set_title(title, loc='left', fontweight='bold')
        axes[0, col].legend(loc='lower left', frameon=False, fontsize=8)
        axes[1, col].yaxis.set_major_formatter(PercentFormatter(1))
        axes[1, col].tick_params(axis='x', rotation=25)
    axes[0, 0].set_ylabel('Equity (initial capital = 100)')
    axes[1, 0].set_ylabel('Drawdown at 4h boundaries')
    for ax in axes.flat:
        ax.grid(alpha=0.18)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('No demonstrated trading edge in either study', fontsize=16, fontweight='bold', x=0.065, ha='left')
    fig.text(0.065, 0.015, 'Official Binance USD-M archives | 5 bps fee + 2 bps slippage per side | actual funding | separate holdout periods; not a paired improvement estimate', fontsize=8)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out/'equity_comparison.png', dpi=160)
    plt.close(fig)
    rows = []
    for study in ['experiment_001', 'experiment_002']:
        r = json.loads(Path('artifacts', study, 'report.json').read_text())
        rows.append({'study': study, 'policy': 'selected_' + r['selected'], **r['test']['base']['portfolio']})
        rows.extend({'study': study, 'policy': name, **v['portfolio']} for name, v in r['comparators'].items())
    pd.DataFrame(rows).to_csv(out/'metrics.csv', index=False)
    print(str((out/'equity_comparison.png').resolve()))


if __name__ == '__main__':
    main()
