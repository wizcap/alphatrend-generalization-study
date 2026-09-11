"""Full timeframe results, annual regimes, controls and source sensitivity."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from .research_v8 import OUT
from .timeframes import TIMEFRAMES


def run():
    report = json.loads((OUT/'report.json').read_text())
    verification = json.loads((OUT/'verification.json').read_text())
    legacy = json.loads((OUT/'legacy_4h_reconciliation.json').read_text())
    rows = {r['name']: r for r in report['strategies']}
    out = Path('reports')
    records, folds = [], []
    for name, row in rows.items():
        base, recent = row['development']['base'], row['recent']['base']
        records.append({'name': name, 'timeframe': row['timeframe'], 'mode': 'passive' if row.get('passive') else ('long_short' if row['long_short'] else 'long_flat'),
            **{k: base[k] for k in ['return', 'max_drawdown', 'sharpe', 'daily_sharpe', 'order_count', 'mean_abs_exposure',
                'mean_turnover_per_sleeve_window', 'mean_cost_per_sleeve_window', 'mean_funding_per_sleeve_window']},
            '2024_return': base['annual_returns']['2024'], '2025_return': base['annual_returns']['2025'],
            '2026_known_return': recent['return'], '2026_known_drawdown': recent['max_drawdown'],
            'btc_return': base['by_symbol_return']['BTCUSDT'], 'eth_return': base['by_symbol_return']['ETHUSDT'],
            **{s+'_return': row['development'][s]['return'] for s in ['double_cost', 'delay15m', 'delay_native', 'source_sensitivity']},
            'source_difference_pp': 100*(row['development']['source_sensitivity']['return']-base['return']),
            'descriptive_checks_pass': row['all_descriptive_checks_pass']})
        for scenario, parts in row['folds'].items():
            for i, record in enumerate(parts):
                folds.append({'name': name, 'scenario': scenario, 'window': i, 'start': record['window'][0], 'end': record['window'][1],
                              **record['portfolio'], 'btc_return': record['by_symbol']['BTCUSDT']['return'],
                              'eth_return': record['by_symbol']['ETHUSDT']['return']})
    table = {r['name']: r for r in records}
    pd.DataFrame(records).to_csv(out/'study008_metrics.csv', index=False)
    pd.DataFrame(folds).to_csv(out/'study008_windows.csv', index=False)
    contrast_rows = []
    for c in report['contrasts']:
        for block, interval in c['adjusted_intervals'].items():
            contrast_rows.append({'candidate': c['candidate'], 'reference': c['reference'], 'block_days': block,
                'log_wealth_excess': c['log_wealth_excess'], 'ci_low': interval[0], 'ci_high': interval[1],
                'nonzero_difference_days': c['nonzero_difference_days'], 'nonzero_runs_within_windows': c['nonzero_runs_within_windows']})
    pd.DataFrame(contrast_rows).to_csv(out/'study008_contrasts.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    tfs, pos = list(TIMEFRAMES), np.arange(len(TIMEFRAMES))
    for col, mode in enumerate(['long', 'ls']):
        names = [tf+'_'+mode for tf in tfs]
        ax = axes[0, col]
        ax.bar(pos-.17, [table[n]['return'] for n in names], .34, color='#167f89', label='Base costs')
        ax.bar(pos+.17, [table[n]['double_cost_return'] for n in names], .34, color='#83b8c0', label='Double costs')
        ax.axhline(table['passive_vol']['return'], color='#a67b35', ls='--', label='Always long, common vol sizing (4h)')
        ax.axhline(0, color='#555', lw=.8)
        ax.set_xticks(pos, tfs)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(('Long / flat' if mode == 'long' else 'Long / short')+' | net return 2024-2025', loc='left', fontweight='bold')
        ax.legend(fontsize=8, frameon=False)
        ax = axes[1, col]
        values = np.array([[table[n]['2024_return'], table[n]['2025_return'], table[n]['2026_known_return']] for n in names])
        limit = max(.001, np.abs(values).max())
        ax.imshow(values, cmap='RdYlGn', vmin=-limit, vmax=limit, aspect='auto')
        ax.set_yticks(pos, tfs)
        ax.set_xticks(range(3), ['2024', '2025', '2026 Jan-Aug\nknown context'])
        for i in range(len(names)):
            for j in range(3):
                ax.text(j, i, f'{values[i,j]:+.1%}', ha='center', va='center', fontsize=10)
        ax.set_title('Annual / partial-year outcomes', loc='left', fontweight='bold')
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('AlphaTrend14/1 across10timeframes | Same15m accounting grid', fontsize=17, fontweight='bold')
    fig.text(.5, .015, 'BTC/ETH equal initial sleeves. Fees, slippage and funding included. Known development data; no deployment selection.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, .96], h_pad=3)
    fig.savefig(out/'study008_comparison.png', dpi=160)
    plt.close(fig)

    alpha = [r for r in report['strategies'] if not r.get('passive')]
    passing = [r['name'] for r in alpha if r['all_descriptive_checks_pass']]
    text = ['# Study008：AlphaTrend 各時間框架成效', '',
        '已完成10個時間框架×多/空手、多空，共20個AlphaTrend策略，加2個被動多頭對照。固定原版14期、ATR倍數1，沒有訓練NEAT或按結果調參。', '',
        f'共有{len(passing)}/20個AlphaTrend策略通過描述性清單：{", ".join(passing) if passing else "無"}。這不是統計顯著性或部署選拔。2024–2025是已觀察開發資料，2026至08-31 20:00UTC是已觀察近期對照，不能稱新holdout。', '',
        '## 主比較', '',
        '每個時間框架真正使用該週期的完整OHLCV計算AlphaTrend，並在收盤後開盤執行；不是把相同4h訊號換調倉頻率。每個策略保留同一42根4h波動估計與最大50%目標名目曝險，按自身週期調倉。成本為每側5bps手續費＋2bps滑價，另計實際資金費。', '',
        '累計報酬涵蓋2024–2025兩年；回撤用共同15m權益格點。半年終點平倉、等資金分配兩幣後串接。', '']
    for mode, label in [('long', '只做多／空手'), ('ls', '多空')]:
        text += [f'### {label}', '', '| 週期 | 兩年淨報酬 | 最大回撤 | 2024 | 2025 | 2026已觀察對照 | 雙倍費滑 |',
                 '|---|---:|---:|---:|---:|---:|---:|']
        for tf in tfs:
            r = table[tf+'_'+mode]
            text.append(f'| {tf} | {r["return"]:+.2%} | {r["max_drawdown"]:.2%} | {r["2024_return"]:+.2%} | {r["2025_return"]:+.2%} | {r["2026_known_return"]:+.2%} | {r["double_cost_return"]:+.2%} |')
        text.append('')
    text += ['## 被動對照', '', '| 對照（每4h調倉） | 兩年淨報酬 | 回撤 | 2024 | 2025 | 2026對照 |', '|---|---:|---:|---:|---:|---:|']
    for n, label in [('passive_vol', '同波動倉位，永遠多頭'), ('passive_half', '固定目標50%，永遠多頭')]:
        r = table[n]
        text.append(f'| {label} | {r["return"]:+.2%} | {r["max_drawdown"]:.2%} | {r["2024_return"]:+.2%} | {r["2025_return"]:+.2%} | {r["2026_known_return"]:+.2%} |')
    text += ['', '對照用同樣費用、調倉帶與窗口重設。若AlphaTrend在多頭年份獲利但落後被動多頭，不能只據正報酬主張訊號有效。兩幣高度相關，不能當兩項獨立確認。', '',
        '## 交易頻率、延遲與來源敏感度', '',
        '| 策略 | 下單數 | 平均絕對曝險 | 延遲15m報酬 | 延遲自身一根報酬 | 替代來源報酬 | 來源差額pp |', '|---|---:|---:|---:|---:|---:|---:|']
    for row in alpha:
        r = table[row['name']]
        text.append(f'| {r["name"]} | {r["order_count"]} | {r["mean_abs_exposure"]:.2%} | {r["delay15m_return"]:+.2%} | {r["delay_native_return"]:+.2%} | {r["source_sensitivity_return"]:+.2%} | {r["source_difference_pp"]:+.3f} |')
    largest = max(records, key=lambda r: abs(r['source_difference_pp']))
    text += ['', '自身一根延遲從15分鐘到7天，包含每窗起點空手等待，不能當成相同網路延遲。15m策略兩種延遲本來就相同。慢週期較少更新實際風控倉位；期末強制平倉對少交易策略相對影響較大。目標50%上限不會阻止其後名目曝險隨行情漂移。', '',
        f'官方不同解析度檔案存在少量價量不一致。主資料固定用15m，替代來源只在評分前找出的8個幣種/日期換用完整1m聚合。最大累計報酬差額為{largest["name"]} {largest["source_difference_pp"]:+.3f}百分點；沒有根據較佳績效選擇來源。', '',
        '## 舊4h結果與同來源帳務核對', '',
        '| 方向 | 情境 | 本次15m來源報酬 | study007舊4h來源 | 差額pp | 本次15m回撤 | 舊4h回撤 |', '|---|---|---:|---:|---:|---:|---:|']
    for r in legacy['old_source_differences']:
        text.append(f'| {r["name"]} | {r["scenario"]} | {r["new15m_source_return"]:+.3%} | {r["study007_return"]:+.3%} | {r["return_difference_pp"]:+.4f} | {r["new15m_drawdown"]:.3%} | {r["study007_4h_drawdown"]:.3%} |')
    text += ['', '相同15m來源上，4h策略以粗4h帳務及細15m帳務計算，其4h邊界完整權益、費用與funding在2e−10絕對容許差內一致。與舊study007來源差額另列，沒有宣稱兩個不同資料版本逐筆相同。', '',
        '## 描述性配對比較與清單', '',
        '18項非4h框架對同方向4h比較，以各半年內每日log財富差作7日及28日循環區塊，各100,000次重抽樣，36個區間一併校正。低頻策略的交易與分歧事件較少，區間仍可能不穩；這不是預測未來獲利的檢定。', '',
        '| 策略 | log財富超額 | 7日區塊區間 | 28日區塊區間 | 非零差異日數 |', '|---|---:|---|---|---:|']
    for c in report['contrasts']:
        intervals = [f'[{c["adjusted_intervals"][b][0]:+.3f}, {c["adjusted_intervals"][b][1]:+.3f}]' for b in ['7', '28']]
        text.append(f'| {c["candidate"]} | {c["log_wealth_excess"]:+.3f} | {intervals[0]} | {intervals[1]} | {c["nonzero_difference_days"]} |')
    text += ['', '清單為兩開發年度正、兩幣正、15m回撤<=20%、15m Sharpe>=0.5、至少3/4半年正、双倍費滑與兩種延遲正、2026已觀察對照正。清單沒有估計誤判率，不等於新獨立證據。各策略未通過項如下：', '']
    for row in alpha:
        failed = [k for k, v in row['descriptive_checks'].items() if not v]
        text.append('- '+row['name']+'：'+(', '.join(failed) if failed else '清單全部通過'))
    text += ['', '## 檔案與驗證', '',
        f'- {verification["tests_passed"]}項測試通過；{verification["verified_source_archives"]}份原始檔雜湊核對。',
        f'- {verification["formula_bars"]:,}根各框架K線獨立公式重算一致；沒有宣稱Pine執行環境逐根比對。',
        f'- {verification["independent_asset_accounting_checks"]:,}次單幣獨立分段帳務與逐筆模擬核對、{verification["replayed_strategy_scenario_windows"]}個策略/情境/窗口重播。',
        '- 年度、每月複利、完整權益曲線及凍結程式通過核對。沒有新幣種、實盤單或外部租用算力。', '',
        '方法與來源：[study008_method.md](study008_method.md)；[Claude審閱處理](study008_review.md)。原作者：[AlphaTrend策略](https://www.tradingview.com/script/3wdQu7P3-AlphaTrend-Strategy/)。官方來源：[Binance公開資料](https://github.com/binance/binance-public-data)。', '',
        '資料表：[全策略](study008_metrics.csv)、[每窗口及每情境](study008_windows.csv)、[配對區間](study008_contrasts.csv)。原始紀錄與曲線在`artifacts/experiment_008/`，包含`aggregation_4h_audit.json`、`legacy_4h_reconciliation.json`、`verification.json`。', '',
        '```powershell', '.venv/Scripts/python.exe -m neattrade.verify_v8', '.venv/Scripts/python.exe -m neattrade.report_v8', '```', '',
        '原實驗由`python -m neattrade.research_v8`建立；完成目錄保持不可覆寫。', '',
        '![AlphaTrend timeframes](study008_comparison.png)', '']
    (out/'study008_results_zh.md').write_text('\n'.join(text), encoding='utf-8')
    print('Saved study008 tables, chart and report.')


if __name__ == '__main__':
    run()
