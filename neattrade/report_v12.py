"""All loss-repair results and relative equity, without post-result selection."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .repair_v12 import FRAMES, KINDS, MODES, CONTROLS
from .research_v12 import OUT, SCENARIOS


def run():
    report = json.loads((OUT/'report.json').read_text())
    rows = {v['name']: v for v in report['strategies']}
    metrics, assets, intervals = [], [], []
    for name, row in rows.items():
        for period in ['development', 'recent']:
            for sc, _ in SCENARIOS:
                v = row[period][sc]
                rep = row[period+'_repair'][sc]['all']
                metrics.append({'name': name, 'period': period, 'scenario': sc,
                    **{k: v[k] for k in ['return', 'max_drawdown', 'daily_sharpe', 'orders', 'mean_abs_exposure']},
                    **{'year_'+k: n for k, n in v['annual_returns'].items()},
                    **{s+'_return': n for s, n in v['by_symbol_return'].items()}, **rep})
        for sc, windows in row['folds'].items():
            for fold, period in enumerate(windows):
                assets.extend({'name': name, 'scenario': sc, 'fold': fold, 'symbol': s, **v} for s, v in period['by_symbol'].items())
    for c in report['comparisons']:
        intervals.extend({'candidate': c['candidate'], 'reference': c['reference'], 'block_days': int(block),
            'log_wealth_excess': c['log_wealth_excess'], 'lower': lo, 'upper': hi} for block, (lo, hi) in c['intervals'].items())
    pd.DataFrame(metrics).to_csv('reports/study012_metrics.csv', index=False)
    pd.DataFrame(assets).to_csv('reports/study012_assets.csv', index=False)
    pd.DataFrame(intervals).to_csv('reports/study012_intervals.csv', index=False)
    for filename in ['trades', 'actions', 'baseline_episodes']:
        pd.DataFrame(json.loads((OUT/(filename+'.json')).read_text())).to_csv(f'reports/study012_{filename}.csv', index=False)
    pd.DataFrame(report['models']).to_csv('reports/study012_training.csv', index=False)
    events = json.loads((OUT/'actions.json').read_text())
    diagnostic = []
    model_rows = {m['name']: m for m in report['models']}
    for name in rows:
        for fold in range(6):
            for sc, _ in SCENARIOS:
                observed = [e for e in events if e['name'] == name and e['fold'] == fold and e['scenario'] == sc and e['prediction'] is not None]
                for head in ['entry', 'exit']:
                    values = np.asarray([e['prediction'] for e in observed if e['head'] == head])
                    if not len(values):
                        continue
                    tf, kind, mode = name.split('_')
                    fitted = model_rows[f'{tf}_fold{min(fold, 4)}_{kind}_{head}']
                    diagnostic.append({'name': name, 'fold': fold, 'scenario': sc, 'head': head,
                        'scored_events': len(values), 'prediction_min': float(values.min()), 'prediction_median': float(np.median(values)),
                        'prediction_max': float(values.max()), 'prediction_std': float(values.std()),
                        'negative_prediction_fraction': float(np.mean(values < 0)),
                        'sign_disagreement_with_training_mean': float(np.mean((values < 0) != (fitted['mean'] < 0)))})
    pd.DataFrame(diagnostic).to_csv('reports/study012_prediction_diagnostics.csv', index=False)
    text = ['# Study012：學習修正 AlphaTrend 虧損的完整結果', '',
        '本輪統一在訊號後15分鐘執行，含每邊7bps成本與歷史資金費率；基準對應study011的延遲情境。雙倍成本為14bps，另測總延遲30分鐘。2024–25為兩年合計，2026至8月31日20:00UTC。全部為已知歷史探索。', '',
        'entry = 只過濾預測淨收益為負的進場；exit = 只在持倉訊號24小時後，預測繼續持有較差時提前退出；both = 合併。mean是歷史平均值控制，ridge與boost才是特徵模型。exit24h與loss1atr24h為固定退出規則。', '',
        '虧損修復率與贏家保留率使用BTC／ETH逐筆加總的資產對數財富變化，不能當成組合報酬百分點。完整表包含半年度邊界交易；自然結束交易的單獨統計也保留於report.json，沒有用它過濾組合績效。', '']
    def fmt(v):
        return '—' if v is None else f'{100*v:+.2f}%'
    for tf in FRAMES:
        text += [f'## {tf}', '', '| 策略 | 2024–25 | 最大回撤 | 2025 | 2026 | 雙倍成本 2024–25 | 虧損修復率 | 贏家保留率 |',
                 '|---|---:|---:|---:|---:|---:|---:|---:|']
        for arm in CONTROLS+[kind+'_'+mode for kind in KINDS for mode in MODES]:
            row = rows[tf+'_'+arm]
            a, b = row['development'], row['recent']
            rep = row['development_repair']['base']['all']
            v = [a['base']['return'], a['base']['max_drawdown'], a['base']['annual_returns']['2025'],
                b['base']['return'], a['double_cost']['return'], rep['loss_repair_fraction'], rep['winner_retention']]
            text.append('| '+arm+' | '+' | '.join(fmt(n) for n in v)+' |')
        text += ['']
    text += ['## 預先指定的研究門檻', '', '| 模型 | 未通過條件 |', '|---|---|']
    for gate in report['gates']:
        failed = ', '.join(k for k, value in gate['checks'].items() if not value)
        text.append('| '+gate['name']+' | '+(failed or '全部通過，仍需新資料驗證')+' |')
    text += ['', 'net_improvement：兩年淨改善；drawdown_no_worse：回撤不增加；double_cost_no_worse：雙倍成本相對基準不差；recent_no_worse：2026相對基準不差；repair_20pct：至少修復20%虧損；retain_90pct：保留90%贏家對數獲利；both_intervals_positive：兩個調整區間下界皆正。這是研究診斷門檻，並非實盤授權。', '',
        '32個預先指定配對比較，64個Bonferroni調整區間；無後見閾值、模型或時間框架搜尋。尚未自然結束的訓練交易沒有被製造成輸贏標籤。模型不增加部位、不在AlphaTrend空倉時另行進場、不在同一段訊號內重新進場。']
    Path('reports/study012_results_zh.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    colors = plt.cm.tab10(np.arange(6))
    arms = [kind+'_'+mode for kind in ['ridge', 'boost'] for mode in MODES]
    for i, period in enumerate(['development', 'recent']):
        for j, tf in enumerate(FRAMES):
            ax = axes[i, j]
            with np.load(OUT/f'{tf}_{period}_equity.npz') as curves:
                dates = pd.to_datetime(curves['timestamps'], utc=True)
                baseline = curves[tf+'_alpha__base']
                ax.axhline(1., color='black', linewidth=1.2, label='AlphaTrend')
                for color, arm in zip(colors, arms):
                    ax.plot(dates, curves[tf+'_'+arm+'__base']/baseline, color=color, label=arm, linewidth=1.25)
            ax.set_title(tf+' | '+('2024-2025' if i == 0 else '2026 Jan-Aug'))
            ax.set_ylabel('Equity / AlphaTrend equity')
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4)
    fig.suptitle('Study012: does loss repair preserve net wealth?', fontsize=16)
    fig.savefig('reports/study012_relative_equity.png', dpi=160)
    plt.close(fig)
    print('Study012 full tables and relative-equity figure written.')


if __name__ == '__main__':
    run()
