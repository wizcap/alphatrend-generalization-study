"""Step 1: reconcile frozen study003 PnL and order-size attribution."""
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from .data import load
from .market import make_market, simulate
from .research import config
from .research_v3 import FOLDS, SEEDS, daily_schedule, desired


def main():
    out = Path('artifacts/study004_audit')
    out.mkdir(parents=True, exist_ok=True)
    old = json.loads(Path('artifacts/experiment_003/selection.json').read_text())
    expected = next(r for r in old['development'] if r['name']=='neat_growth')
    bars = {s: load(s) for s in ('BTCUSDT', 'ETHUSDT')}
    results, all_orders, all_states = [], [], []
    for fold, (a,z) in enumerate(FOLDS):
        members = []
        for seed in SEEDS:
            with Path(f'artifacts/experiment_003/jobs/fold{fold}_growth_{seed}.pkl').open('rb') as f:
                members.append(pickle.load(f))
        candidate = {'kind':'ensemble','members':members}
        for symbol,b in bars.items():
            m = make_market(symbol,b,a,z)
            result = simulate(m,desired(candidate,m,config()),trade_allowed=daily_schedule(m),trace=True)
            np.testing.assert_allclose(result['return'],expected['folds'][fold]['by_symbol'][symbol]['return'],atol=1e-12)
            frame = pd.DataFrame(result['ledger'])
            reconciliation = frame.price_pnl.sum()-frame.funding_paid.sum()-frame.trade_cost.sum()-frame.terminal_exit_cost.sum()
            np.testing.assert_allclose(reconciliation,result['return'],atol=1e-12)
            frame.to_csv(out/f'fold{fold}_{symbol}.csv',index=False)
            grouped = frame.groupby('order_category').agg(orders=('trade_cost',lambda x:int((x>0).sum())),
                        cost_initial_equity=('trade_cost','sum'),turnover=('traded_equity','sum')).reset_index()
            grouped['fold'],grouped['symbol'] = fold,symbol
            all_orders.append(grouped)
            frame['direction'] = np.where(frame.units>0,'long',np.where(frame.units<0,'short','flat'))
            frame['trend_state'] = np.where(frame.slow_trend>0,'positive','nonpositive')
            state = frame.groupby(['direction','trend_state']).agg(price_pnl=('price_pnl','sum'),
                    funding_paid=('funding_paid','sum'),bars=('units','size')).reset_index()
            state['fold'],state['symbol'] = fold,symbol
            all_states.append(state)
            small = frame.order_category.eq('small_adjustment')
            results.append({'fold':fold,'symbol':symbol,'return':result['return'],'reconciled':True,
                'small_adjustment_order_fraction':float(small.sum()/max(1,(frame.order_category!='none').sum())),
                'small_adjustment_cost_fraction':float(frame.loc[small,'trade_cost'].sum()/max(1e-12,frame.trade_cost.sum())),
                'price_pnl':float(frame.price_pnl.sum()),'execution_cost':float(frame.trade_cost.sum()+frame.terminal_exit_cost.sum()),
                'funding_paid':float(frame.funding_paid.sum())})
    pd.concat(all_orders).to_csv(out/'orders_by_category.csv',index=False)
    pd.concat(all_states).to_csv(out/'pnl_by_direction_and_trend.csv',index=False)
    (out/'audit.json').write_text(json.dumps({'scope':'post-hoc diagnosis of frozen study003; no reselection', 'results':results},indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
