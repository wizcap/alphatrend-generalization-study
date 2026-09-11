"""Full short-timeframe study tables, coverage, decisions and readable chart."""
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from .research_v10 import OUT, SHORT, SPECS


def run():
    report = json.loads((OUT/'report.json').read_text())
    verification = json.loads((OUT/'verification.json').read_text())
    rows = {r['name']: r for r in report['strategies']}
    quality = json.loads((OUT/'feature_manifest.json').read_text())['quality']
    out = Path('reports')
    metrics_rows = []
    for name, row in rows.items():
        for group in ['development', 'recent']:
            for scenario, value in row[group].items():
                metrics_rows.append({'name': name, 'timeframe': row['timeframe'], 'arm': row['arm'],
                    'period': group, 'scenario': scenario,
                    **{k: v for k, v in value.items() if not isinstance(v, dict)},
                    **{year+'_return': v for year, v in value['annual_returns'].items()},
                    **value['by_symbol_return']})
    pd.DataFrame(metrics_rows).to_csv(out/'study010_metrics.csv', index=False)
    records = []
    for row in rows.values():
        for scenario, folds in row['folds'].items():
            for fold, record in enumerate(folds):
                for symbol, value in record['by_symbol'].items():
                    records.append({'name': row['name'], 'timeframe': row['timeframe'], 'scenario': scenario,
                        'fold': fold, 'symbol': symbol, 'start': record['window'][0], 'end': record['window'][1],
                        'portfolio_return': record['portfolio']['return'], **value})
    pd.DataFrame(records).to_csv(out/'study010_assets.csv', index=False)
    pd.DataFrame(report['decision_diagnostics']).to_csv(out/'study010_decisions.csv', index=False)
    pd.DataFrame(quality).to_csv(out/'study010_quality.csv', index=False)
    crows = [{'candidate': c['candidate'], 'reference': c['reference'], 'block_days': b,
              'log_wealth_excess': c['log_wealth_excess'], 'lower': ci[0], 'upper': ci[1],
              'nonzero_days': c['nonzero_days'], 'adjusted_intervals': c['intervals_corrected']}
             for c in report['comparisons'] for b, ci in c['adjusted_intervals'].items()]
    pd.DataFrame(crows).to_csv(out/'study010_intervals.csv', index=False)
    pd.DataFrame(report['training_times']).to_csv(out/'study010_training_times.csv', index=False)
    text = ['# Study010：15分、30分、1小時的模型與訂單簿比較', '',
        'BTC／ETH，AlphaTrend做多／空手方向與共用風控；兩年累積報酬含費用、滑價與資金費。'
        '15m／30m／1h同時調整指標框架、決策頻率及訂單簿平均窗口。2024–2025與2026皆為已觀察資料。'
        '四個學習組的主結果為三種子目標倉位平均，完整單種子另列，沒有選取上線模型。', '',
        '| 框架／策略 | 2024–25淨報酬 | 15m最大回撤 | 2024 | 2025 | 2026已知對照 | 雙倍費滑兩年 | 訂單數 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    main = ['alpha', 'alpha_half', 'passive_vol', 'cash', *SPECS]
    for tf in SHORT:
        for arm in main:
            row = rows[tf+'_'+arm]
            m = row['development']['base']
            text.append(f'| {tf}／{arm} | {m["return"]:+.2%} | {m["max_drawdown"]:.2%} | '
                f'{m["annual_returns"]["2024"]:+.2%} | {m["annual_returns"]["2025"]:+.2%} | '
                f'{row["recent"]["base"]["return"]:+.2%} | {row["development"]["double_cost"]["return"]:+.2%} | {m["orders"]} |')
    text += ['', '## 相同框架內的訂單簿增量', '',
        '差額以百分點呈現。模型架構、資料可用旗標、成本及訓練預算在各配對內相同；'
        '零欄與變動欄仍會改變搜尋地形及後續隨機演化路徑，不能視為完全隔離的因果特徵重要度。', '',
        '| 框架／模型 | 書本減零欄 | 歸零改變目標／總決策 | 回退比例 | 2026歸零改變目標 |',
        '|---|---:|---:|---:|---:|']
    for tf in SHORT:
        for kind in ['at', 'cma']:
            name = tf+'_'+kind+'_book'
            delta = rows[name]['development']['base']['return']-rows[tf+'_'+kind+'_zero']['development']['base']['return']
            d = [x for x in report['decision_diagnostics'] if x['name'] == name and x['fold'] < 4]
            recent = [x for x in report['decision_diagnostics'] if x['name'] == name and x['fold'] >= 4]
            total = sum(x['total'] for x in d)
            text.append(f'| {tf}／{kind} | {delta*100:+.2f}百分點 | '
                f'{sum(x["book_zeroing_changed_targets"] for x in d):,}／{total:,} | '
                f'{sum(x["missing_fallbacks"] for x in d)/total:.2%} | '
                f'{sum(x["book_zeroing_changed_targets"] for x in recent):,} |')
    text += ['', '## 全部單種子', '', '| 策略 | 兩年淨報酬 | 2026對照 |', '|---|---:|---:|']
    for name, row in rows.items():
        if '_seed' in name:
            text.append(f'| {name} | {row["development"]["base"]["return"]:+.2%} | {row["recent"]["base"]["return"]:+.2%} |')
    text += ['', '## 壓力與推論消融', '',
        '| 策略 | 成交延後15m | 書本延後一框架 | 書本歸零 | 平均絕對曝險 |', '|---|---:|---:|---:|---:|']
    for tf in SHORT:
        for arm in SPECS:
            name = tf+'_'+arm
            r = rows[name]['development']
            text.append(f'| {name} | {r["delay15m"]["return"]:+.2%} | {r["book_lag1bar"]["return"]:+.2%} | '
                f'{r["book_zeroed"]["return"]:+.2%} | {r["base"]["mean_abs_exposure"]:.2%} |')
    text += ['', '書本延後同步移動有效旗標，零欄也可能因共同回退而變動。歸零為固定模型的推論診斷，'
        '不等於重新訓練。15m延遲在15m策略是整整一根，在1h策略是四分之一根，並不是相等的相對延遲。', '',
        '## 配對差異區間', '',
        '每半年內分別抽樣，然後合成兩年對數財富差；21項比較、每項7／28日各100,000次，'
        '共42區間尾機率校正。表中是對數財富差而非報酬百分點。區間條件於既定模型，'
        '不處理完整重訓變異、先前框架搜尋或市場存活偏差。', '',
        '| 候選／對照 | 差異 | 7日區塊 | 28日區塊 |', '|---|---:|---:|---:|']
    for c in report['comparisons']:
        a, b = c['adjusted_intervals']['7'], c['adjusted_intervals']['28']
        text.append(f'| {c["candidate"]}／{c["reference"]} | {c["log_wealth_excess"]:+.4f} | '
                    f'[{a[0]:+.4f}, {a[1]:+.4f}] | [{b[0]:+.4f}, {b[1]:+.4f}] |')
    text += ['', '## 訂單簿品質', '',
        '每個窗口至少95%有效秒數，加最新狀態不過期檢查；下表為通過此規則的決策比例，兩者分母不同。'
        '60秒可用延遲與120秒持續上限是假設。底層是±1%累積深度快照，不含排隊、逐筆撤單或可成交衝擊。', '',
        '| 框架 | 幣種 | 半年 | 可用率 |', '|---|---|---|---:|']
    for q in quality:
        text.append(f'| {q["timeframe"]} | {q["symbol"]} | {q["year"]}H{q["half"]} | {q["usable_fraction"]:.2%} |')
    text += ['', '2026H2僅至8月底，品質表來源端點與模型末端精確切分不同。'
        '缺資料時共用回退，市場損益仍保留；不同框架回退遮罩亦不同，不能將跨框架差異完全歸因訊號頻率。', '',
        '## 驗證與耗時', '']
    for tf in SHORT:
        for kind in ['neat', 'cma']:
            seconds = [x['seconds'] for x in report['training_times'] if x['timeframe'] == tf and x['kind'] == kind]
            text.append(f'- {tf}／{kind}：{len(seconds)}工作，中位耗時{statistics.median(seconds):.2f}秒。')
    text += ['', f'主研究耗時{report["elapsed_seconds"]:.1f}秒，含訓練、情境及區間，不含資料特徵建置與獨立驗證；'
        '本機3工作程序，沒有專用硬體速度排名。180工作、207,360候選評估預算。', '',
        f'{verification["tests_passed"]}項測試；{verification["independent_book_window_checks"]:,}個特徵窗口獨立積分；'
        f'{verification["selected_models_replayed"]}個模型前向與訓練適應度；'
        f'{verification["independent_asset_accounting_checks"]:,}次單幣帳務；全曲線與年度、study008同框架固定控制一致。', '',
        f'實際候選評估{report["actual_candidate_evaluations"]:,}次；42個配對區間以不同的區塊前綴和實作重算一致。'
        '缺資料回退次數按原決策時點統計；成交延後不另製造一筆訊號。', '',
        '[事前方法](study010_method.md) · [Claude審閱處理](study010_review.md) · [全部指標](study010_metrics.csv) · '
        '[單幣成本與窗口](study010_assets.csv) · [決策](study010_decisions.csv) · [品質](study010_quality.csv) · '
        '[42個區間](study010_intervals.csv)', '', '![短框架模型比較](study010_comparison.png)', '']
    (out/'study010_results_zh.md').write_text('\n'.join(text), encoding='utf-8')
    arms = ['alpha', 'at_zero', 'at_book', 'cma_zero', 'cma_book']
    labels = ['AlphaTrend', 'NEAT zero', 'NEAT book', 'CMA zero', 'CMA book']
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey='row')
    x = np.arange(len(arms))
    for col, tf in enumerate(SHORT):
        for row, group in enumerate(['development', 'recent']):
            ax = axes[row, col]
            ax.bar(x-.18, [rows[tf+'_'+arm][group]['base']['return'] for arm in arms], .36, label='Base costs', color='#167f89')
            ax.bar(x+.18, [rows[tf+'_'+arm][group]['double_cost']['return'] for arm in arms], .36, label='Double costs', color='#84bbc3')
            ax.axhline(0, color='#555', linewidth=.7)
            ax.set_title(tf+' | '+('2024–2025 development' if row == 0 else '2026 Jan–Aug known context'), loc='left', fontsize=11)
            ax.set_xticks(x, labels, rotation=25, ha='right')
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_axisbelow(True)
            ax.grid(axis='y', alpha=.15)
    fig.suptitle('Short timeframes: same risk rules, shorter AlphaTrend and book windows', fontsize=15, y=.98)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .94), ncol=2, frameon=False)
    fig.text(.5, .012, 'BTC/ETH, costs and funding included. Three-seed target ensembles. Previously observed history; no deployment selection.', ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .04, 1, .88])
    fig.savefig(out/'study010_comparison.png', dpi=170)
    plt.close(fig)
    print('Saved study010 complete report, six tables and chart.')


if __name__ == '__main__':
    run()
