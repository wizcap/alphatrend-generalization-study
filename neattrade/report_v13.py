"""Complete monthly-update comparison and controller decision artifacts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .adaptive_v13 import FRAMES, ARMS, utc
from .research_v13 import OUT, SCENARIOS


def run():
    report = json.loads((OUT/'report.json').read_text())
    rows = {r['name']: r for r in report['strategies']}
    metrics, assets, comparisons, updates, decisions = [], [], [], [], []
    for row in rows.values():
        for period in ['development', 'recent']:
            for sc, _ in SCENARIOS:
                v = row[period][sc]
                metrics.append({'name': row['name'], 'period': period, 'scenario': sc,
                    **{k: v[k] for k in ['return', 'max_drawdown', 'daily_sharpe', 'orders', 'mean_abs_exposure']},
                    **{'year_'+k: value for k, value in v['annual_returns'].items()},
                    **v['by_symbol_return'], **row[period+'_repair'][sc]['all']})
        for sc, windows in row['folds'].items():
            for fold, window in enumerate(windows):
                assets += [{'name': row['name'], 'scenario': sc, 'fold': fold, 'symbol': s, **v} for s, v in window['by_symbol'].items()]
    for c in report['comparisons']:
        comparisons += [{'candidate': c['candidate'], 'reference': c['reference'], 'block_days': block,
            'log_wealth_excess': c['log_wealth_excess'], 'lower': lo, 'upper': hi} for block, (lo, hi) in c['intervals'].items()]
    for tf, history in report['controller_history'].items():
        for r in history:
            value = {k: v for k, v in r.items() if k not in ['health', 'validation', 'actions']}
            value.update(timeframe=tf, actions=';'.join(r['actions']))
            for source in ['health', 'validation']:
                q = r[source]
                if q is not None:
                    value.update({source+'_'+k: q[k] for k in ['checkpoints', 'early_exits']})
                    value[source+'_episodes'] = len(q['ids'])
                    value.update({source+'_'+k: v for k, v in q['economics']['base'].items()})
                    value[source+'_double_excess'] = q['economics']['double_cost']['alpha_excess']
                    if source == 'validation':
                        value['failed_checks'] = ';'.join(k for k, v in q['checks'].items() if not v)
            updates.append(value)
    actions = json.loads((OUT/'actions.json').read_text())
    model_rows = {m['name']: m for m in report['models']}
    for name in rows:
        for period, folds in [('development', range(4)), ('recent', range(4, 6))]:
            use = [e for e in actions if e['name'] == name and e['scenario'] == 'base' and e['fold'] in folds]
            entries = [e for e in use if e['head'] == 'entry']
            assigned = [e for e in entries if e['version'] is not None]
            ages = [(utc(e['time'])-utc(model_rows[e['version']]['available'])).total_seconds()/86400 for e in assigned]
            decisions.append({'name': name, 'period': period, 'entries': len(entries),
                'entries_with_model': len(assigned), 'versions_used': len({e['version'] for e in assigned}),
                'early_exits': sum(e['reason'] == 'early_exit' for e in use),
                'mean_model_age_days_at_entry': float(np.mean(ages)) if ages else None,
                'max_model_age_days_at_entry': max(ages) if ages else None})
    for name, data in [('metrics', metrics), ('assets', assets), ('intervals', comparisons), ('updates', updates),
        ('decisions', decisions), ('models', report['models']), ('actions', actions),
        ('trades', json.loads((OUT/'trades.json').read_text()))]:
        pd.DataFrame(data).to_csv(f'reports/study013_{name}.csv', index=False)
    text = ['# Study013：模型定期學習與自動版本替換的完整結果', '',
        '全部使用已知歷史。每邊7bps成本及歷史資金費率，訊號後15分鐘執行；雙倍成本為14bps，另測總延遲30分鐘。2024–25為兩年合計，2026至8月31日20:00UTC。', '',
        'frozen：僅2024年初訓練；semiannual：上輪半年更新；monthly：每月累積重訓；weighted：每月重訓並讓資料權重每180日減半；guarded：候選版本先經後續交易檢驗，通過才替換，表現轉差可退回Alpha。所有版本皆僅決定24h檢查點的退出。', '']
    for tf in FRAMES:
        history = report['controller_history'][tf]
        counts = {k: sum(k in r['actions'] for r in history) for k in ['queue', 'promote', 'reject', 'expire', 'rollback']}
        text += [f'## {tf}', '', '控制器事件：'+', '.join(f'{k}={v}' for k, v in counts.items())+'。這些是模擬版本操作，不是部署批准。', '',
            '| 方法 | 2024–25 | 最大回撤 | 2025 | 2026 | 雙倍成本2024–25 | 虧損修復率 | 贏家保留率 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for arm in ARMS:
            row = rows[tf+'_'+arm]; a = row['development']; rep = row['development_repair']['base']['all']
            values = [a['base']['return'], a['base']['max_drawdown'], a['base']['annual_returns']['2025'],
                row['recent']['base']['return'], a['double_cost']['return'], rep['loss_repair_fraction'], rep['winner_retention']]
            text.append('| '+arm+' | '+' | '.join('—' if v is None else f'{v*100:+.2f}%' for v in values)+' |')
        text += ['']
    nonzero = sum(c['lower'] > 0 or c['upper'] < 0 for c in comparisons)
    text += [f'24個預先指定且經多重比較調整的區間中，{nonzero}個排除零。控制器篩選分數是逐筆獨立資產對數收益加總；上表淨報酬是等初始資金組合，兩者不能當成相同單位。虧損修復與贏家保留同樣以逐筆對數收益計算。', '',
        '月更不會更換已持倉交易的管理版本，也不會在月初平倉。模型更新、觀察期與回退紀錄完整保留；未完成的交易不供訓練或篩選使用，仍完整保留在策略績效中。壓力情境沿用基礎情境選出的更新時程，沒有在壓力結果上重新選模。', '',
        '本輪是可續跑的離線研究流程，沒有啟動長駐排程或實盤服務。不能將「可以自動更新」等同於「一定持續改善」。']
    Path('reports/study013_results_zh.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    palette = dict(zip(ARMS, ['black', 'gray', '#9467bd', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']))
    for i, group in enumerate(['development', 'recent']):
        for j, tf in enumerate(FRAMES):
            ax = axes[i, j]
            with np.load(OUT/f'{tf}_{group}_equity.npz') as curves:
                dates = pd.to_datetime(curves['timestamps'], utc=True)
                for arm in ARMS:
                    ax.plot(dates, curves[tf+'_'+arm+'__base'], label=arm, color=palette[arm],
                        linewidth=1.8 if arm == 'guarded' else 1.1, linestyle='--' if arm == 'guarded' else '-')
            ax.set_title(tf+' | '+('2024-2025' if i == 0 else '2026 Jan-Aug'))
            ax.set_ylabel('Net equity (initial = 1)'); ax.grid(alpha=.2)
    fig.suptitle('Study013: does continued learning improve trading?', fontsize=15)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside lower center', ncol=4)
    fig.savefig('reports/study013_equity.png', dpi=160); plt.close(fig)
    print('Study013 tables, update decisions and equity chart written.')


if __name__ == '__main__':
    run()
