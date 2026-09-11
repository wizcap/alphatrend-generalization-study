"""Export every frozen study011 arm, without selecting or tuning a strategy."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .indicators_v11 import FRAMES, PACKS
from .research_v11 import OUT, KINDS, CONTROLS, SCENARIOS, SPECS


def run():
    result = json.loads((OUT/'report.json').read_text())
    rows = {v['name']: v for v in result['strategies']}
    flat, assets, intervals = [], [], []
    for name, row in rows.items():
        for period in ['development', 'recent']:
            for scenario, _ in SCENARIOS:
                v = row[period][scenario]
                flat.append({'name': name, 'timeframe': row['timeframe'], 'arm': row['arm'],
                    'period': period, 'scenario': scenario,
                    **{k: v[k] for k in ['return', 'max_drawdown', 'daily_sharpe', 'orders', 'mean_abs_exposure']},
                    **{'year_'+k: n for k, n in v['annual_returns'].items()},
                    **{s+'_return': n for s, n in v['by_symbol_return'].items()}})
        for scenario, periods in row['folds'].items():
            for fold, window in enumerate(periods):
                for symbol, v in window['by_symbol'].items():
                    assets.append({'name': name, 'scenario': scenario, 'fold': fold, 'symbol': symbol,
                        'start': window['window'][0], 'end': window['window'][1], **v})
    for c in result['comparisons']:
        for block, (lo, hi) in c['adjusted_intervals'].items():
            intervals.append({'candidate': c['candidate'], 'reference': c['reference'],
                'block_days': int(block), 'log_wealth_excess': c['log_wealth_excess'],
                'lower': lo, 'upper': hi, 'nonzero_days': c['nonzero_days']})
    pd.DataFrame(flat).to_csv('reports/study011_metrics.csv', index=False)
    pd.DataFrame(assets).to_csv('reports/study011_assets.csv', index=False)
    pd.DataFrame(intervals).to_csv('reports/study011_intervals.csv', index=False)
    pd.DataFrame(result['decision_diagnostics']).to_csv('reports/study011_decisions.csv', index=False)
    pd.DataFrame(result['training_times']).to_csv('reports/study011_training.csv', index=False)
    text = ['# Study011 完整結果：移除訂單簿後的指標、模型與時間框架', '',
        '全部為既有歷史資料上的探索結果。2024–2025 是四個半年視窗串接；2026 至 8 月 31 日 20:00 UTC，兩個視窗使用相同的 2025 年底模型。所有數字已扣每邊 5 bps 手續費、2 bps 滑價與實際資金費率；雙倍成本為每邊 14 bps。下列模型列均為三種子的目標部位平均。', '',
        'Alpha = 四個 AlphaTrend 特徵；Momentum = Alpha 加趨勢動能；Vol/flow = Alpha 加波動與成交量；All = 十二特徵。線性與 MLP 使用 CMA-ES，NEAT 同时演化拓撲與權重。模型可自行做多或空倉，固定規則作對照。', '',
        f"已執行 {result['training_jobs']} 個訓練、{result['actual_candidate_evaluations']:,} 次候選評估、{result['strategy_scenario_windows']:,} 個組合情境視窗。未選定實盤模型。", '']
    for tf in FRAMES:
        text += [f'## {tf}', '', '| 模型 / 特徵 | 2024–25 淨報酬 | 最大回撤 | 2024 | 2025 | 2026 | 雙倍成本 2024–25 | 雙倍成本 2026 | 延遲 2024–25 |',
                 '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for arm in CONTROLS+[k+'_'+p for k, p in SPECS]:
            row = rows[tf+'_'+arm]
            dev, recent = row['development'], row['recent']
            values = [dev['base']['return'], dev['base']['max_drawdown'], dev['base']['annual_returns']['2024'],
                dev['base']['annual_returns']['2025'], recent['base']['return'], dev['double_cost']['return'],
                recent['double_cost']['return'], dev['delay15m']['return']]
            text.append('| '+arm+' | '+' | '.join(f'{v*100:+.2f}%' for v in values)+' |')
        text += ['']
    text += ['## 統計與限制', '',
        '預先指定 63 個配對比較，每個使用 7 日、28 日區塊，合计 126 個 Bonferroni 調整區間。比較使用每日對數財富差，不能把端點當成報酬百分點。模型間與時間框架排名僅作描述。修正不涵蓋過往多輪研究；已知歷史不構成新驗證。', '',
        '每個種子的所有情境、各資產、年度、成本、資金費率、訂單數與持倉暴露均保留在 CSV；完整曲線及模型在 artifacts/experiment_011。最大回撤以 15 分鐘帳務曲線計算。沒有掛單簿欄位、缺資料掩碼或回退；成交量中的主動買入比例來自成交資料。', '',
        '時間框架同時改變指標的實際回看時間與交易頻率；特徵包增加輸入也增加模型容量。此輪無獨立動能或波動特徵組，衡量的是在 AlphaTrend 輸入上新增特徵的整體效果。模型動作空間也與 study010 不同，因此不能將兩輪差異單獨歸因於移除訂單簿。']
    Path('reports/study011_results_zh.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    matrices = [np.array([[rows[f'{tf}_{kind}_{pack}'][period]['base']['return']*100 for kind in KINDS] for pack in PACKS])
                for period in ['development', 'recent'] for tf in FRAMES]
    limit = max(5., max(np.abs(m).max() for m in matrices))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    for ax, values, (period, tf) in zip(axes.flat, matrices, [(p, t) for p in ['2024-2025', '2026 Jan-Aug'] for t in FRAMES]):
        im = ax.imshow(values, cmap='RdYlGn', vmin=-limit, vmax=limit, aspect='auto')
        ax.set_xticks(range(3), ['Linear CMA', 'MLP CMA', 'NEAT'])
        ax.set_yticks(range(4), ['Alpha', '+ Momentum', '+ Vol/flow', 'All'])
        ax.set_title(tf+' | '+period)
        for (i, j), value in np.ndenumerate(values):
            ax.text(j, i, f'{value:+.1f}%', ha='center', va='center', fontsize=11,
                    color='white' if abs(value) > limit*.8 else '#17202a')
    fig.colorbar(im, ax=axes, shrink=.8, label='Net return (%)')
    fig.suptitle('Study011: candle-only inputs | three-seed target ensembles', fontsize=16)
    fig.savefig('reports/study011_matrix.png', dpi=160)
    plt.close(fig)
    print('Study011 tables, 954 metric rows, 5724 asset rows and comparison figure written.')


if __name__ == '__main__':
    run()
