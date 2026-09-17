from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from src.nse_data import load_prices
from src.nse_fo import load_fo
from src.single_stock_context import load_context_dir
from src.single_stock_strategy import build_single_stock_frame, walk_forward_oos
from src.intraday_direction_strategy import walk_forward_direction
ROOT=Path(__file__).resolve().parents[1]
def parse_args():
 p=argparse.ArgumentParser(); p.add_argument('--symbol',default='RELIANCE'); p.add_argument('--lookback-days',type=int,default=1800); p.add_argument('--train-days',type=int,default=504); p.add_argument('--test-days',type=int,default=21); p.add_argument('--step-days',type=int,default=21); p.add_argument('--target',type=float,default=.0275); p.add_argument('--stop',type=float,default=.0125); p.add_argument('--probability-gate',type=float,default=.60); return p.parse_args()
def add_fo_context(stock,fo,symbol):
 if fo.empty:return stock
 z=fo[fo.symbol.eq(symbol)].sort_values('date').copy()
 if z.empty:return stock
 for c in ['fut_oi','fut_oi_change','fut_volume','pcr']: z[c]=pd.to_numeric(z[c],errors='coerce')
 z['fo_oi_z20']=(z.fut_oi-z.fut_oi.rolling(20,min_periods=5).mean())/z.fut_oi.rolling(20,min_periods=5).std().replace(0,pd.NA)
 z['fo_oi_change_z20']=(z.fut_oi_change-z.fut_oi_change.rolling(20,min_periods=5).mean())/z.fut_oi_change.rolling(20,min_periods=5).std().replace(0,pd.NA)
 z['fo_volume_rel20']=z.fut_volume/z.fut_volume.rolling(20,min_periods=5).mean(); z['pcr_dev20']=z.pcr-z.pcr.rolling(20,min_periods=5).mean()
 return stock.merge(z[['date','fo_oi_z20','fo_oi_change_z20','fo_volume_rel20','pcr_dev20']],on='date',how='left')
def main():
 a=parse_args(); cache=ROOT/'data'/'cache'; prices=load_prices(str(cache),a.lookback_days); prices.date=pd.to_datetime(prices.date)
 stock=prices[(prices.symbol==a.symbol)&prices.series.astype(str).str.upper().eq('EQ')].copy().sort_values('date')
 if stock.empty: raise SystemExit(f'No EQ history found for {a.symbol}')
 stock['prev_day_high']=stock.high.shift(1); stock['prev_day_low']=stock.low.shift(1); stock=add_fo_context(stock,load_fo(str(cache),a.lookback_days),a.symbol)
 frame=build_single_stock_frame(stock,load_context_dir(str(ROOT/'data'),a.symbol))
 long=walk_forward_oos(frame,a.train_days,a.test_days,a.step_days,a.target,a.stop,a.probability_gate); both=walk_forward_direction(frame,a.train_days,a.test_days,a.step_days,a.target,a.stop,a.probability_gate)
 out=ROOT/'reports'/'single_stock'; out.mkdir(parents=True,exist_ok=True)
 long.predictions.to_csv(out/f'{a.symbol}_long_oos_predictions.csv',index=False); long.trades.to_csv(out/f'{a.symbol}_long_oos_trades.csv',index=False); both.predictions.to_csv(out/f'{a.symbol}_long_short_oos_predictions.csv',index=False); both.trades.to_csv(out/f'{a.symbol}_long_short_oos_trades.csv',index=False)
 summary={'symbol':a.symbol,'intraday_only':True,'entry':'next session open','exit':'target/stop or session close','target':a.target,'stop':a.stop,'probability_gate':a.probability_gate,'long_only_oos':long.metrics,'long_short_oos':both.metrics,'context':'symbol-aware dated context; future labels excluded from features'}
 (out/f'{a.symbol}_oos_summary.json').write_text(json.dumps(summary,indent=2,default=str)); print(json.dumps(summary,indent=2,default=str))
if __name__=='__main__': main()
