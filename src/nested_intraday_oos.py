from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.intraday_direction_strategy import FEATURES, barrier_return, build_direction_targets

@dataclass
class NestedOOS:
    predictions: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict
    selections: pd.DataFrame

def _fit_predict(train: pd.DataFrame, test: pd.DataFrame):
    if len(train) == 0 or train.direction_label.nunique() < 2 or len(test) == 0:
        return None
    clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.035, max_leaf_nodes=15, l2_regularization=1.0, random_state=42)
    clf.fit(train[FEATURES], train.direction_label)
    prob = clf.predict_proba(test[FEATURES]); classes = list(clf.classes_)
    pl = prob[:, classes.index(1)] if 1 in classes else np.zeros(len(test))
    ps = prob[:, classes.index(-1)] if -1 in classes else np.zeros(len(test))
    return pl, ps

def _candidate_frame(frame: pd.DataFrame, target: float, stop: float) -> pd.DataFrame:
    x = build_direction_targets(frame, target, stop).sort_values('date')
    return x.dropna(subset=FEATURES + ['next_open', 'next_open_to_high', 'next_open_to_low', 'next_open_to_close'])

def _validation_score(frame: pd.DataFrame, train_dates: np.ndarray, target: float, stop: float, gate: float, inner_test_days: int = 21):
    if len(train_dates) < 2 * inner_test_days:
        return -np.inf, 0
    split = len(train_dates) - inner_test_days
    tr_dates, va_dates = train_dates[:split], train_dates[split:]
    candidate = _candidate_frame(frame, target, stop)
    tr, va = candidate[candidate.date.isin(tr_dates)], candidate[candidate.date.isin(va_dates)].copy()
    pred = _fit_predict(tr, va)
    if pred is None:
        return -np.inf, 0
    pl, ps = pred
    side = np.where((pl >= gate) & (pl > ps), 1, np.where((ps >= gate) & (ps > pl), -1, 0))
    rets = []
    for i, (_, row) in enumerate(va.iterrows()):
        if side[i] != 0:
            r, _ = barrier_return(row, int(side[i]), target, stop); rets.append(r)
    if not rets:
        return -np.inf, 0
    arr = np.asarray(rets, dtype=float)
    compounded = float(np.prod(1 + arr) - 1)
    return compounded - 0.0005 * max(0, 10 - len(arr)), len(arr)

def _choose_params(frame, train_dates, target_grid, stop_grid, gate_grid, inner_test_days):
    best = (-np.inf, target_grid[0], stop_grid[0], gate_grid[0], 0)
    for target, stop, gate in product(target_grid, stop_grid, gate_grid):
        score, n = _validation_score(frame, train_dates, target, stop, gate, inner_test_days)
        if score > best[0] or (score == best[0] and n > best[4]):
            best = (score, target, stop, gate, n)
    return best[1], best[2], best[3], best[0], best[4]

def _max_consecutive(flags):
    best = cur = 0
    for flag in flags:
        cur = cur + 1 if flag else 0; best = max(best, cur)
    return best

def _capital_metrics(trades, initial_capital, cost_bps_side, slippage_bps_side, risk_fraction, max_position_fraction):
    capital = float(initial_capital); peak = capital; rows = []
    for _, t in trades.iterrows():
        stop = float(t.stop_return); gross = float(t['return'])
        position_fraction = min(max_position_fraction, risk_fraction / max(stop, 1e-9))
        notional = capital * position_fraction
        round_trip_cost = 2.0 * (cost_bps_side + slippage_bps_side) / 10000.0
        net_return = gross - round_trip_cost; pnl = notional * net_return; capital += pnl; peak = max(peak, capital)
        rows.append({**t.to_dict(), 'position_fraction': position_fraction, 'notional': notional, 'gross_return': gross, 'net_return': net_return, 'pnl': pnl, 'equity': capital, 'drawdown': capital / peak - 1.0})
    eq = pd.DataFrame(rows)
    if eq.empty:
        return eq, {'initial_capital': initial_capital, 'final_capital': initial_capital, 'net_return': 0.0, 'max_drawdown': 0.0, 'trade_count': 0}
    months = eq.assign(month=pd.to_datetime(eq.date).dt.to_period('M'))
    monthly = months.groupby('month').equity.last().pct_change().dropna()
    wins = int((eq.net_return > 0).sum()); gross_profit = eq.loc[eq.pnl > 0, 'pnl'].sum(); gross_loss = -eq.loc[eq.pnl < 0, 'pnl'].sum()
    return eq, {'initial_capital': initial_capital, 'final_capital': float(eq.equity.iloc[-1]), 'net_return': float(eq.equity.iloc[-1] / initial_capital - 1), 'max_drawdown': float(eq.drawdown.min()), 'trade_count': int(len(eq)), 'hit_rate': float(wins / len(eq)), 'profit_factor': float(gross_profit / gross_loss) if gross_loss > 0 else None, 'avg_net_trade_return': float(eq.net_return.mean()), 'median_net_trade_return': float(eq.net_return.median()), 'monthly_avg_return': float(monthly.mean()) if len(monthly) else 0.0, 'monthly_median_return': float(monthly.median()) if len(monthly) else 0.0, 'profitable_month_fraction': float((monthly > 0).mean()) if len(monthly) else 0.0, 'worst_trade': float(eq.net_return.min()), 'consecutive_losses_max': _max_consecutive((eq.net_return < 0).tolist())}

def nested_walk_forward_direction(frame, train_days=504, test_days=21, step_days=21, target_grid=(.015,.020,.025,.0275,.030,.035), stop_grid=(.0075,.010,.0125,.015,.020), gate_grid=(.55,.60,.65,.70,.75,.80), inner_test_days=21, initial_capital=100000, cost_bps_side=8.0, slippage_bps_side=3.0, risk_fraction=.01, max_position_fraction=.95):
    dates = np.array(sorted(pd.to_datetime(frame.date).unique())); preds=[]; trades=[]; selections=[]
    for end in range(train_days, len(dates), step_days):
        tr_dates=dates[end-train_days:end]; te_dates=dates[end:min(end+test_days,len(dates))]
        base=_candidate_frame(frame,max(target_grid),min(stop_grid)); test_base=base[base.date.isin(te_dates)].copy()
        if test_base.empty: continue
        target,stop,gate,score,inner_n=_choose_params(frame,tr_dates,target_grid,stop_grid,gate_grid,inner_test_days)
        labeled=_candidate_frame(frame,target,stop); train=labeled[labeled.date.isin(tr_dates)]; test=labeled[labeled.date.isin(te_dates)].copy()
        pred=_fit_predict(train,test)
        if pred is None: continue
        pl,ps=pred; side=np.where((pl>=gate)&(pl>ps),1,np.where((ps>=gate)&(ps>pl),-1,0)); p=np.maximum(pl,ps)
        out=test[['date','next_open']].copy(); out['p_long']=pl; out['p_short']=ps; out['side']=side; out['trade']=side!=0; out['selected_target']=target; out['selected_stop']=stop; out['selected_gate']=gate; preds.append(out)
        selections.append({'test_start':str(te_dates[0]),'test_end':str(te_dates[-1]),'target':target,'stop':stop,'gate':gate,'inner_score':score,'inner_trade_count':inner_n})
        for i,(_,row) in enumerate(test.iterrows()):
            if side[i]==0: continue
            gross,reason=barrier_return(row,int(side[i]),target,stop)
            trades.append({'date':row.date,'side':'LONG' if side[i]==1 else 'SHORT','entry':row.next_open,'return':gross,'exit_reason':reason,'probability':float(p[i]),'target_return':target,'stop_return':stop,'probability_gate':gate})
    predictions=pd.concat(preds,ignore_index=True) if preds else pd.DataFrame(); raw=pd.DataFrame(trades)
    equity,metrics=_capital_metrics(raw,initial_capital,cost_bps_side,slippage_bps_side,risk_fraction,max_position_fraction)
    if not raw.empty:
        metrics['long_trades']=int((raw.side=='LONG').sum()); metrics['short_trades']=int((raw.side=='SHORT').sum()); metrics['gross_compounded_return']=float(np.prod(1+raw['return'])-1)
    metrics.update({'cost_bps_per_side':cost_bps_side,'slippage_bps_per_side':slippage_bps_side,'risk_fraction':risk_fraction,'max_position_fraction':max_position_fraction,'nested_selection':True})
    return NestedOOS(predictions,equity,metrics,pd.DataFrame(selections))