"""Render the completed four-step study004 batch without new selection."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


def main():
    run=Path('artifacts/experiment_004');out=Path('reports')
    selection=json.loads((run/'selection.json').read_text());report=json.loads((run/'report.json').read_text())
    rows={r['name']:r for r in selection['development']}
    curves=pd.read_csv(run/'development_equity.csv',index_col=0,parse_dates=True)
    recent=pd.read_csv(run/'known2026_equity.csv',index_col=0,parse_dates=True)
    series=[('baseline_banded','Matched slow-trend baseline','#be8a20'),('neat_plain','NEAT confidence gate','#087f8c'),
            ('neat_hysteretic','NEAT gate with hysteresis','#8874bc')]
    fig,axes=plt.subplots(2,2,figsize=(14,9),gridspec_kw={'height_ratios':[1.7,1]})
    for key,label,color in series:
        axes[0,0].plot(curves.index,curves[key]*100,color=color,label=label,linewidth=1.6)
        axes[0,1].plot(recent.index,recent[key]*100,color=color,label=label,linewidth=1.6)
    for ax in axes[0]:
        ax.axhline(100,color='#9ca3af',linestyle='--',linewidth=.8)
        ax.legend(loc='upper left',frameon=False,fontsize=8)
        ax.tick_params(axis='x',rotation=20)
    axes[0,0].set_title('2024-2025 rolling development',loc='left',fontweight='bold')
    axes[0,1].set_title('Jan-Aug 2026: previously observed context',loc='left',fontweight='bold')
    axes[0,0].set_ylabel('Equity (initial capital = 100)')
    for i,(key,label,color) in enumerate(series):
        axes[1,0].bar(np.arange(4)+(i-1)*.24,[f['portfolio']['return'] for f in rows[key]['folds']],width=.23,color=color)
        axes[1,1].bar(np.arange(4)+(i-1)*.24,[rows[key]['scenarios'][s]['return'] for s in ['base','cost_1_5','double_cost','delay_4h']],width=.23,color=color)
    axes[1,0].set_xticks(range(4),['2024 H1','2024 H2','2025 H1','2025 H2'])
    axes[1,0].set_title('Half-year net returns',loc='left',fontsize=11)
    axes[1,1].set_xticks(range(4),['Base','1.5x cost','2x cost','4h delay'])
    axes[1,1].set_title('Development cost and timing stress',loc='left',fontsize=11)
    for ax in axes[1]:
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.axhline(0,color='#888',linewidth=.7)
    for ax in axes.flat:
        ax.spines[['right','top']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('Positive backtests; NEAT has not consistently beaten the baseline',fontsize=16,fontweight='bold',x=.055,ha='left')
    fig.text(.055,.023,'Official Binance USD-M data | Actual funding + 5 bps fee + 2 bps slippage | All displayed periods are development/context; transfer data remains untouched',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,.95]);fig.savefig(out/'study004_comparison.png',dpi=160);plt.close(fig)
    table=[]
    for row in rows.values():
        name=row['name']
        table.append({'candidate':name,**row['scenarios']['base'],
                      'double_cost_return':row['scenarios']['double_cost']['return'],
                      'delayed_return':row['scenarios']['delay_4h']['return'],
                      'known2026_return':report['known2026_context'][name]['portfolio']['return'],
                      'known2026_drawdown':report['known2026_context'][name]['portfolio']['max_drawdown'],
                      'absolute_pass':row['absolute_pass'],'incremental_neat_pass':row['incremental_neat_pass']})
    pd.DataFrame(table).to_csv(out/'study004_metrics.csv',index=False)
    seeds=[]
    for mode in ['plain','hysteretic']:
        for seed in [11,29,47]:
            records=[r for r in selection['per_seed'] if r['mode']==mode and r['seed']==seed]
            seeds.append({'mode':mode,'seed':seed,'compounded_return':float(np.prod([1+r['portfolio']['return'] for r in records])-1),
                          'train_cash_folds':sum(r['train_cash'] for r in records)})
    pd.DataFrame(seeds).to_csv(out/'study004_seeds.csv',index=False)
    print(json.dumps(seeds,indent=2))


if __name__=='__main__':main()
