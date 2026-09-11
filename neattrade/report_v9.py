"""Deliver all book-feature learner, seed, window and stress results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from .research_v9 import OUT, SCENARIOS, SPECS


def run():
    report=json.loads((OUT/'report.json').read_text())
    verification=json.loads((OUT/'verification.json').read_text())
    rows={r['name']:r for r in report['strategies']}
    out=Path('reports')
    records=[]
    for name,r in rows.items():
        for group in ['development','recent']:
            for scenario,m in r[group].items():
                records.append({'name':name,'period':group,'scenario':scenario,
                    **{k:v for k,v in m.items() if not isinstance(v,dict)},
                    **{y+'_return':v for y,v in m['annual_returns'].items()},
                    **m['by_symbol_return']})
    pd.DataFrame(records).to_csv(out/'study009_metrics.csv',index=False)
    pd.DataFrame([{'name':r['name'],'scenario':s,'window':i,**record['portfolio'],
        **{symbol+'_return':m['return'] for symbol,m in record['by_symbol'].items()}}
        for r in rows.values() for s,parts in r['folds'].items() for i,record in enumerate(parts)]).to_csv(out/'study009_windows.csv',index=False)
    pd.DataFrame(report['decision_diagnostics']).to_csv(out/'study009_decisions.csv',index=False)
    names=['alpha','alpha_half','passive_vol','at_zero','at_book','cma_zero','cma_book']
    labels=['AlphaTrend','Half AlphaTrend','Always long','NEAT + zero','NEAT + book','CMA-ES + zero','CMA-ES + book']
    fig,axes=plt.subplots(1,2,figsize=(16,6))
    x=np.arange(len(names))
    for ax,group,title in zip(axes,['development','recent'],['2024–2025 development','2026 Jan–Aug known context']):
        ax.bar(x-.17,[rows[n][group]['base']['return'] for n in names],.34,color='#167f89',label='Base costs')
        ax.bar(x+.17,[rows[n][group]['double_cost']['return'] for n in names],.34,color='#84bbc3',label='Double costs')
        ax.axhline(0,color='#555',linewidth=.8)
        ax.set_xticks(x,labels,rotation=22,ha='right')
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(title,loc='left',fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
        ax.legend(frameon=False)
    fig.suptitle('One book feature: NEAT and fixed MLP + CMA-ES',fontsize=15,fontweight='bold')
    fig.text(.5,.015,'BTC/ETH equal initial sleeves. Costs and funding included. Known-data research; no deployment selection.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,.92])
    fig.savefig(out/'study009_comparison.png',dpi=170)
    plt.close(fig)
    a,z=rows['at_book']['development']['base'],rows['at_zero']['development']['base']
    recent_a,recent_z=rows['at_book']['recent']['base'],rows['at_zero']['recent']['base']
    text=['# Study009：訂單簿特徵、NEAT與CMA-ES比較','',
        '固定4h AlphaTrend方向，模型學習0/0.5/1倍風險。四組皆有5個輸入：原4項AlphaTrend，加固定0或真實訂單簿不平衡。NEAT演化拓撲，CMA-ES優化固定5→4→1的29參數小網路。主結果為各自3種子目標倉位平均的集成策略；所有單種子一併列出。','',
        f'NEAT在2024–2025：加入訂單簿{a["return"]:+.2%}，零欄對照{z["return"]:+.2%}，差額{100*(a["return"]-z["return"]):+.2f}百分點。',
        f'NEAT的2026已觀察對照：加入訂單簿{recent_a["return"]:+.2%}，零欄對照{recent_z["return"]:+.2%}。',
        f'CMA-ES在2024–2025：加入訂單簿{rows["cma_book"]["development"]["base"]["return"]:+.2%}，零欄對照{rows["cma_zero"]["development"]["base"]["return"]:+.2%}。','',
        '## 特徵與因果時點','',
        '使用官方bookDepth的−1%買方及+1%賣方累積掛單USDT金額，先算(買−賣)/(買＋賣)，再按可用時間對過去4h加權平均，範圍[-1,1]。假定快照延後60秒可用，最多持續120秒，涵蓋率至少95%；資料不足時各組共同回退固定AlphaTrend。先排除區間平均價格不一致與單側金額/數量停滯的快照。它不含逐筆撤單、排隊、價差或可成交價格衝擊。','',
        '2024–2025是已觀察開發歷史；2026至08-31 20:00UTC亦為已觀察對照，不是新holdout。訓練從2023-01-02開始，每窗保留7天間隔，60個工作×48族群×24代=69,120個候選評分預算。CMA-ES在使用者詢問架構後、訓練與評分前加入；沒有依結果擴展搜尋。','',
        '## 全策略比較','',
        '| 策略 | 2024–25淨報酬 | 15m最大回撤 | 2024 | 2025 | 2026對照 | 雙倍費滑（兩年） |','|---|---:|---:|---:|---:|---:|---:|']
    for name,r in rows.items():
        m=r['development']['base']
        text.append(f'| {name} | {m["return"]:+.2%} | {m["max_drawdown"]:.2%} | {m["annual_returns"]["2024"]:+.2%} | {m["annual_returns"]["2025"]:+.2%} | {r["recent"]["base"]["return"]:+.2%} | {r["development"]["double_cost"]["return"]:+.2%} |')
    text += ['', '## 特徵依賴與時效','',
        '| 策略 | 延遲成交15m | 書本延遲4h | 推論時書本歸零 | 原始下單數 | 平均絕對曝險 |', '|---|---:|---:|---:|---:|---:|']
    for name in SPECS:
        r=rows[name]['development']
        text.append(f'| {name} | {r["delay15m"]["return"]:+.2%} | {r["book_lag4h"]["return"]:+.2%} | {r["book_zeroed"]["return"]:+.2%} | {r["base"]["orders"]} | {r["base"]["mean_abs_exposure"]:.2%} |')
    for name in SPECS:
        d=[r for r in report['decision_diagnostics'] if r['name']==name and r['fold']<4]
        text.append(f'- {name}：開發期{sum(r["total"] for r in d):,}個幣種/決策，缺資料回退{sum(r["missing_fallbacks"] for r in d):,}次；固定模型書本歸零後改變{sum(r["book_zeroing_changed_targets"] for r in d):,}個目標。')
    text += ['', '## 配對差異的不確定性','',
        '四半年內分別重抽樣，不跨窗口拼接區塊；7日與28日各100,000次，對三項比較的六個區間校正。兩項比較是各架構加入書本的差異，一項為兩個書本模型的整體架構差異。','']
    for c in report['comparisons']:
        text.append(f'- {c["candidate"]}對{c["reference"]}：log終值財富差{c["log_wealth_excess"]:+.4f}；非零差異日{c["nonzero_days"]}。')
        for block,ci in c['adjusted_intervals'].items():
            text.append(f'  - {block}日區塊：[{ci[0]:+.4f}, {ci[1]:+.4f}]。')
    text += ['', '這些區間以已訓練策略為條件，不涵蓋完整重訓變異或前幾輪研究偏差。三種子高度共享資料，BTC/ETH也高度相關。加入一個變動欄與固定0欄的比較，不能完全排除搜尋地形或正則化效果。跨NEAT/CMA-ES同時改變模型架構與搜尋方法，不能單獨歸因於優化器；相同評估次數也不等於相同耗時。','',
        '缺資料回退不等於模型自行判斷空手；平均曝險已列出，沒有按結果重新調整基準倉位。未重建客戶端接收時點、歷史檔修訂或完整逐價深度，故不能將60秒假設當成已驗證的實盤延遲。','',
        '## 驗證','',
        f'- {verification["tests_passed"]}項測試；{verification["raw_archives_verified"]}份來源檔雜湊。',
        f'- {verification["independent_book_window_checks"]:,}個4h決策點的書本窗口獨立積分一致。',
        f'- {verification["selected_models_replayed"]}個模型以獨立前向運算（NEAT參考實作／逐節點MLP）及訓練適應度重播。',
        f'- {verification["independent_asset_accounting_checks"]}次單幣分段帳務核對、完整曲線/年度及study008固定AlphaTrend對照一致。','',
        '資料：[全情境](study009_metrics.csv)、[各窗口](study009_windows.csv)、[決策依賴](study009_decisions.csv)。方法：[事前方案](study009_method.md)；[Claude審閱處理](study009_review.md)。原始記錄：`artifacts/experiment_009/`。','',
        '![Order-book feature comparison](study009_comparison.png)','']
    (out/'study009_results_zh.md').write_text('\n'.join(text),encoding='utf-8')
    print('Saved study009 report, tables and chart.')


if __name__=='__main__':
    run()
