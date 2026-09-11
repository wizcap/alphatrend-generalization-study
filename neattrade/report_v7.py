"""Publish the frozen AlphaTrend comparison, with all controls and stresses."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from .indicator_types import LABELS, TYPES
from .research_v3 import SEEDS
from .research_v6 import seed_compound

NAMES = {'cash': '空手', 'baseline': '原 EMA180 基線', 'ema_premium': 'EMA180＋固定溢價減倉',
         'alpha_only': 'AlphaTrend 單獨', 'alpha_half': 'AlphaTrend 半倉',
         'alpha_funding': 'AlphaTrend＋固定 funding 減倉', 'alpha_premium': 'AlphaTrend＋固定 premium 減倉',
         'alpha_both': 'AlphaTrend＋固定 funding/premium OR',
         'alpha_daily_ls': 'AlphaTrend 每日多空', 'alpha_4h_long': 'AlphaTrend 每4h多/空手',
         'alpha_4h_ls': 'AlphaTrend 每4h多空', 'at_constant': 'AlphaTrend＋常數 NEAT',
         'at_neat': 'AlphaTrend 特徵 NEAT', 'at_all': 'AlphaTrend＋全部26指標'}
NAMES.update({'at_'+g: 'AlphaTrend＋'+label for g, label in LABELS.items()})


def main():
    root, out = Path('artifacts/experiment_007'), Path('reports')
    selection = json.loads((root/'selection.json').read_text(encoding='utf-8'))
    report = json.loads((root/'report.json').read_text(encoding='utf-8'))
    verification = json.loads((root/'verification.json').read_text(encoding='utf-8'))
    rows = {r['name']: r for r in selection['development']}
    records = []
    for name, row in rows.items():
        records.append({'candidate': name, 'label': NAMES[name], **row['scenarios']['base'],
            **{s+'_return': row['scenarios'][s]['return'] for s in ['double_cost', 'delay_4h', 'source_lag_4h']},
            'btc_return': row['asset_compounded_returns']['BTCUSDT'], 'eth_return': row['asset_compounded_returns']['ETHUSDT'],
            'orders': sum(v['order_count'] for f in row['folds'] for v in f['by_symbol'].values()),
            'mean_sleeve_abs_exposure': np.mean([v['mean_abs_exposure'] for f in row['folds'] for v in f['by_symbol'].values()]),
            'mean_sleeve_turnover_per_fold': np.mean([v['turnover'] for f in row['folds'] for v in f['by_symbol'].values()]),
            'mean_cost_per_sleeve_fold': np.mean([v['cost_paid_initial_equity'] for f in row['folds'] for v in f['by_symbol'].values()]),
            'mean_funding_per_sleeve_fold': np.mean([v['funding_paid_initial_equity'] for f in row['folds'] for v in f['by_symbol'].values()]),
            'pass': row['incremental_neat_pass']})
    metrics = {r['candidate']: r for r in records}
    pd.DataFrame(records).to_csv(out/'study007_metrics.csv', index=False, encoding='utf-8-sig')
    seed_rows, seed_folds = [], []
    for row in selection['per_seed']:
        seed_folds.append({k: v for k, v in row.items() if k not in ['portfolio', 'by_symbol']} | row['portfolio'])
    for name in sorted({r['spec'] for r in seed_folds}):
        for seed in SEEDS:
            seed_rows.append({'candidate': name, 'seed': seed, 'return': seed_compound(selection['per_seed'], name, seed)})
    pd.DataFrame(seed_rows).to_csv(out/'study007_seeds.csv', index=False)
    pd.DataFrame(seed_folds).to_csv(out/'study007_seed_folds.csv', index=False)
    pd.DataFrame([{'candidate': c['candidate'], 'reference': c['reference'], 'difference_pp': c['return_difference_pp'],
                  'log_wealth_excess': c['interval']['log_wealth_excess'],
                  'interval_low': c['interval']['family_adjusted_interval'][0],
                  'interval_high': c['interval']['family_adjusted_interval'][1],
                  **{f'seed{seed}_excess': v for seed, v in zip(SEEDS, c['seed_compound_excess'])}}
                  for c in report['contrasts']]).to_csv(out/'study007_contrasts.csv', index=False)

    names = ['alpha_only', 'at_neat']+['at_'+g for g in TYPES]+['at_all']
    labels = ['AT only', 'AT NEAT', '+Momentum', '+Trend', '+Vol/range', '+Flow', '+Funding', '+Basis', '+Interasset', '+All26']
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    pos = np.arange(len(names))
    ax = axes[0, 0]
    ax.bar(pos-.17, [metrics[n]['return'] for n in names], .34, color='#127d8d', label='Base costs')
    ax.bar(pos+.17, [metrics[n]['double_cost_return'] for n in names], .34, color='#8cbfc8', label='Double costs')
    ax.axhline(metrics['baseline']['return'], color='#a57725', ls='--', label='EMA baseline')
    ax.axhline(0, color='#888', lw=.8)
    ax.set_xticks(pos, labels, rotation=35, ha='right')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title('Net return: daily long-only, 2024-2025', loc='left', fontweight='bold')
    ax.legend(fontsize=8, frameon=False)
    ax = axes[0, 1]
    ax.bar(pos, [metrics[n]['max_drawdown'] for n in names], color='#c68d64')
    ax.axhline(.20, color='#7f3636', ls='--', label='20% screening limit')
    ax.set_xticks(pos, labels, rotation=35, ha='right')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title('Maximum drawdown at 4h boundaries', loc='left', fontweight='bold')
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1, 0]
    folds = np.array([[f['portfolio']['return'] for f in rows[n]['folds']] for n in names])
    limit = max(.001, np.abs(folds).max())
    ax.imshow(folds, cmap='RdYlGn', vmin=-limit, vmax=limit, aspect='auto')
    ax.set_yticks(pos, labels)
    ax.set_xticks(range(4), ['2024H1', '2024H2', '2025H1', '2025H2'])
    for i in range(len(names)):
        for j in range(4):
            ax.text(j, i, f'{folds[i,j]:+.1%}', ha='center', va='center', fontsize=8)
    ax.set_title('Each half-year is disclosed', loc='left', fontweight='bold')
    ax = axes[1, 1]
    diagnostic = ['alpha_only', 'alpha_daily_ls', 'alpha_4h_long', 'alpha_4h_ls']
    p = np.arange(4)
    ax.bar(p-.17, [metrics[n]['return'] for n in diagnostic], .34, color='#127d8d', label='Net return')
    ax.bar(p+.17, [metrics[n]['max_drawdown'] for n in diagnostic], .34, color='#c68d64', label='Max drawdown')
    ax.axhline(0, color='#888', lw=.8)
    ax.set_xticks(p, ['Daily long', 'Daily L/S', '4h long', '4h L/S'])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title('Predeclared frequency / direction diagnostics', loc='left', fontweight='bold')
    ax.legend(fontsize=8, frameon=False)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('AlphaTrend + indicator families | Frozen development comparison', fontsize=16, fontweight='bold')
    fig.text(.5, .012, 'BTC/ETH equal initial sleeves. Fees, slippage and funding included. Known development data; no live-profitability claim.',
             ha='center', fontsize=9, color='#444')
    fig.tight_layout(rect=[0, .035, 1, .96], h_pad=2.5, w_pad=2.5)
    fig.savefig(out/'study007_comparison.png', dpi=160)
    plt.close(fig)

    def table(items):
        lines = ['| 策略 | 累計淨報酬 | 最大回撤 | 雙倍費滑 | 延遲4h | 外部來源延遲4h |',
                 '|---|---:|---:|---:|---:|---:|']
        for n in items:
            r = metrics[n]
            lines.append(f'| {NAMES[n]} | {r["return"]:+.2%} | {r["max_drawdown"]:.2%} | {r["double_cost_return"]:+.2%} | {r["delay_4h_return"]:+.2%} | {r["source_lag_4h_return"]:+.2%} |')
        return '\n'.join(lines)

    best = max(names[2:], key=lambda n: metrics[n]['return'])
    baseline, alone, strongest = metrics['baseline'], metrics['alpha_only'], metrics[best]
    selected = selection['selected']
    no_zero = sum(not (c['interval']['family_adjusted_interval'][0] <= 0 <= c['interval']['family_adjusted_interval'][1]) for c in report['contrasts'])
    text = ['# Study007：AlphaTrend 單獨與指標類別組合', '',
        f'已完成 120 次訓練、230,400 次個體評分，以及 21 個策略的四種成本／延遲情境。主要結論：AlphaTrend 單獨為 {alone["return"]:+.2%}，新增類別中最高報酬是 **{NAMES[best]} {strongest["return"]:+.2%}**，原 EMA180 基線為 {baseline["return"]:+.2%}。', '',
        f'事前完整門檻選拔結果：**{selected if selected else "沒有 NEAT 候選通過"}**。這是反覆看過的 BTC/ETH 2024–2025 開發資料，不是新的獨立驗證，更不表示實盤可以獲利。', '',
        '## 1. AlphaTrend 與七類組合', '',
        '以下全為 4h K 線計算、每日 UTC 00:00 調倉、只做多/空手。AlphaTrend 替代 EMA180 方向資格，保留相同波動倉位控制。四個半年重設帳戶後串接；報酬為兩年累計而非年化。單獨策略沒有 NEAT；組合的 NEAT 只能減少 AlphaTrend 原有多頭倉位。', '',
        table(['baseline', 'alpha_only', 'alpha_half', 'at_constant', 'at_neat']+names[2:]), '',
        '七類沿用 study006：動能6項、趨勢3項、波動4項、成交量與買賣流4項、資金費率2項、溢價價差5項、跨幣種2項。每類組合也含共同4個 AlphaTrend 特徵；全部組合為30輸入。類別並非互相獨立，AlphaTrend 本身已使用 MFI、ATR，不能把組合差額說成某指標的純粹因果貢獻。', '',
        '## 2. 每個半年與兩個幣種', '',
        '| 策略 | 2024H1 | 2024H2 | 2025H1 | 2025H2 | BTC累計 | ETH累計 | 總下單數 | 平均絕對曝險 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n in ['baseline']+names:
        r = metrics[n]
        returns = ' | '.join(f'{f["portfolio"]["return"]:+.2%}' for f in rows[n]['folds'])
        text.append(f'| {NAMES[n]} | {returns} | {r["btc_return"]:+.2%} | {r["eth_return"]:+.2%} | {r["orders"]} | {r["mean_sleeve_abs_exposure"]:.2%} |')
    text += ['', '## 3. 固定規則及原生4h訊號', '',
        table(['ema_premium', 'alpha_funding', 'alpha_premium', 'alpha_both', 'alpha_daily_ls', 'alpha_4h_long', 'alpha_4h_ls']), '',
        '固定 funding>3bps、premium>=5bps 時減半，沿用之前閾值。原生4h及多空只是事前診斷，沒有用這些结果重新調整每日 NEAT 訓練。多空代表賣訊轉負倉位；多/空手代表賣訊離場。這些策略的曝險與交易次數不同，不能當成單純新增指標的效果。', '',
        '## 4. 配對比較、隨機種子與未通過條件', '',
        f'17 項事前配對比較中，{17-no_zero} 項描述性區間包含零，{no_zero} 項不包含零。區間位於 log 終值財富超額空間，不能直接讀成報酬百分點；5,000 次 7 日區塊 bootstrap 加17比較校正，仍未校正前六輪研究、重訓及選擇偏差。', '',
        '| 組合 | 對單獨策略差額（百分點） | 對Alpha特徵NEAT差額（百分點） | 對Alpha特徵NEAT三種子超額 | 描述性log超額區間 |',
        '|---|---:|---:|---|---|']
    contrasts = {(c['candidate'], c['reference']): c for c in report['contrasts']}
    for n in names[2:]:
        a, b = contrasts[n, 'alpha_only'], contrasts[n, 'at_neat']
        low, high = b['interval']['family_adjusted_interval']
        seeds = ' / '.join(f'{v*100:+.2f}pp' for v in b['seed_compound_excess'])
        text.append(f'| {NAMES[n]} | {a["return_difference_pp"]:+.2f} | {b["return_difference_pp"]:+.2f} | {seeds} | [{low:+.4f}, {high:+.4f}] |')
    text += ['', '種子順序11/29/47，主結果為固定三種子目標倉位平均，沒有挑最好種子。全部120筆種子/窗口紀錄與每月報酬均保留。不同輸入維度造成搜尋容量與亂數消耗差異；相同種子標籤不等於同一隨機路徑。', '']
    for n in ['at_constant', 'at_neat']+names[2:]:
        row = rows[n]
        failed = [k for k, v in row['absolute_gates'].items() if not v]
        for ref, gates in row['relative_gates'].items():
            failed += [ref+':'+k for k, v in gates.items() if not v]
        if not row['source_lag_pass']:
            failed.append('source_lag_pass')
        text.append(f'- {NAMES[n]}：'+('全部通過' if not failed else '未通過 '+', '.join(failed)))
    text += ['', '## 5. 驗證與重現', '',
        f'- {verification["tests_passed"]} 項測試通過：包含因果截斷、未來資料突變、下一根開盤訊號、相等平台與原始交叉、零量/平價、EMA解耦、来源延遲範圍、調倉頻率及延遲。',
        f'- {verification["independent_formula_bars"]:,} 根歷史K線，以獨立逐根公式重算，AlphaTrend 線及方向完全對上；沒有聲稱已在 TradingView 執行環境逐根比對。',
        f'- {verification["trained_model_hashes_and_reference_networks"]} 個已訓練模型雜湊與標準 NEAT 推論核對；{verification["reference_accounting_checks"]} 次向量化訓練帳務與獨立模擬器對帳。',
        f'- {verification["replayed_candidate_scenarios"]} 個策略/情境重播，原基線及溢價規則四情境重現 study006；完整權益與每月複利對帳通過。',
        '- 原始 Binance 檔案校驗、凍結原碼、方案與模型關聯雜湊均核對。未新增外部算力、未下單、未使用新幣種測試。', '',
        '公式及完整事前設計：[study007_method.md](study007_method.md)。原作者來源：[AlphaTrend](https://www.tradingview.com/script/o50NYLAZ-AlphaTrend/)、[策略原碼頁](https://www.tradingview.com/script/3wdQu7P3-AlphaTrend-Strategy/)。', '',
        '```powershell', '.venv/Scripts/python.exe -m pytest -q',
        '# 另一次實驗必須用全新的輸出目錄；這不是要求再搜尋參數',
        '.venv/Scripts/python.exe -m neattrade.research_v7 --output artifacts/experiment_007_new',
        '# 核對與報告讀取本次保存的 experiment_007',
        '.venv/Scripts/python.exe -m neattrade.verify_v7',
        '.venv/Scripts/python.exe -m neattrade.report_v7', '```', '',
        '完整資料表：[全策略](study007_metrics.csv)、[配對比較](study007_contrasts.csv)、[種子](study007_seeds.csv)、[種子與窗口](study007_seed_folds.csv)。', '',
        '![AlphaTrend comparison](study007_comparison.png)', '']
    (out/'study007_results_zh.md').write_text('\n'.join(text), encoding='utf-8')
    print('Saved study007 report, tables and chart.')


if __name__ == '__main__':
    main()
