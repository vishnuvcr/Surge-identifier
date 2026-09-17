from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
CFG=yaml.safe_load((ROOT/'v6.1/config.yaml').read_text()); TH=float(CFG['regime_threshold'])
FEATURES=['ret1','ret5','ret10','ret20','ema10_gap','ema20_gap','ema50_gap','ema10_20_gap','ema20_50_gap','rsi14','vol10','vol20','breakout20','breakdown20','range60','slope20','atm_straddle_pct','skew_proxy','dte','india_vix','india_vix_ret5','spx_ret5','usd_inr_ret5','brent_ret5','bank_rel5','midcap_rel5','news_sentiment','news_volume_log','news_neg_share']
MODELS=['ridge','hgbr','rf','et','ensemble']
def regime(x): return 'BULLISH' if x>=TH else ('BEARISH' if x<=-TH else 'RANGE_BOUND')
def load_market():
 from v5.run import Market
 o=pd.read_parquet(ROOT/'data/cache/nifty_options_long.parquet'); f=pd.read_parquet(ROOT/'data/cache/nifty_futures_long.parquet'); c=pd.read_parquet(ROOT/'data/cache/v5_market_context.parquet'); n=pd.read_csv(ROOT/'data/cache/v5_news_daily.csv'); n.columns=[str(x).strip().lower() for x in n.columns]
 if 'date' not in c.columns and c.index.name=='date': c=c.reset_index()
 return Market(o,f,c,n)
def expiry_list(m): return sorted(set(sum((list(v) for v in m.expiries.values()),[])))
def spot_on_or_before(m,d):
 d=pd.Timestamp(d)
 if d in m.spot: return float(m.spot[d]),d
 z=[x for x in m.dates if x<=d]
 if not z:return np.nan,None
 x=pd.Timestamp(z[-1]); return float(m.spot[x]),x
def completed_cycles(m):
 exps=expiry_list(m); rows=[]
 for i in range(1,len(exps)):
  prev,e=pd.Timestamp(exps[i-1]),pd.Timestamp(exps[i]); dates=[d for d in m.dates if prev<d<e]
  if not dates:continue
  signal=pd.Timestamp(dates[0]); target,td=spot_on_or_before(m,e)
  if td is None or td<=signal:continue
  spot=float(m.spot[signal])
  if np.isfinite(spot) and np.isfinite(target): rows.append({'signal_date':signal,'expiry':e,'target_date':td,'spot':spot,'target_close':target,'target_return':target/spot-1})
 return pd.DataFrame(rows).drop_duplicates('expiry').sort_values('signal_date').reset_index(drop=True)
def latest_open_cycle(m):
 latest=pd.Timestamp(m.dates[-1]); exps=expiry_list(m); future=[e for e in exps if pd.Timestamp(e)>latest]; prev=[e for e in exps if pd.Timestamp(e)<latest]
 if not future or not prev:return None
 sig=[d for d in m.dates if pd.Timestamp(prev[-1])<d<=latest]
 if not sig:return None
 return {'signal_date':pd.Timestamp(sig[0]),'latest_feature_date':latest,'expiry':pd.Timestamp(future[0]),'spot':float(m.spot[latest])}
def feature_row(m,d):
 from v5 import run as base
 f=base.make_features(m,pd.Timestamp(d))
 if f is None:return None
 c=m.asof_context(pd.Timestamp(d)); f['usd_inr_ret5']=c.get('usd_inr_ret5',c.get('INR=X_ret5',np.nan)); f['brent_ret5']=c.get('brent_ret5',c.get('BZ=F_ret5',np.nan))
 return {k:(float(f[k]) if pd.notna(f.get(k,np.nan)) else np.nan) for k in FEATURES}
def build_dataset(m,cycles):
 rows=[]
 for r in cycles.itertuples(index=False):
  f=feature_row(m,r.signal_date)
  if f is not None: rows.append(dict(f,signal_date=pd.Timestamp(r.signal_date),expiry=pd.Timestamp(r.expiry),target_date=pd.Timestamp(r.target_date),spot=float(r.spot),target_close=float(r.target_close),target_return=float(r.target_return)))
 if not rows:raise RuntimeError('No completed NIFTY expiry cycles available')
 return pd.DataFrame(rows).sort_values('signal_date').reset_index(drop=True)
def regressors():
 seed=int(CFG['random_state']); n=int(CFG['n_estimators']); depth=int(CFG['max_depth']); leaf=int(CFG['min_samples_leaf'])
 return {'ridge':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=10)),'hgbr':make_pipeline(SimpleImputer(strategy='median'),HistGradientBoostingRegressor(max_iter=250,learning_rate=.035,max_leaf_nodes=15,l2_regularization=.5,random_state=seed)),'rf':make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(n_estimators=n,max_depth=depth,min_samples_leaf=leaf,random_state=seed,n_jobs=-1)),'et':make_pipeline(SimpleImputer(strategy='median'),ExtraTreesRegressor(n_estimators=n,max_depth=depth,min_samples_leaf=leaf,random_state=seed,n_jobs=-1))}
def calibration(train):
 cut=max(40,int(len(train)*.8)); fit=train.iloc[:cut]; cal=train.iloc[cut:]; ps=[]
 for model in regressors().values(): model.fit(fit[FEATURES],fit.target_return.astype(float)); ps.append(model.predict(cal[FEATURES]))
 r=cal.target_return.to_numpy()-np.mean(np.column_stack(ps),axis=1); a=1-float(CFG['prediction_interval'])
 return float(np.nanquantile(r,a/2)),float(np.nanquantile(r,1-a/2))
def fit_predict(train,test):
 y=train.target_return.astype(float); models=regressors(); pred={}
 for name,model in models.items(): model.fit(train[FEATURES],y); pred[name]=model.predict(test[FEATURES])
 pred['ensemble']=np.mean(np.column_stack([pred[k] for k in models]),axis=1); qlo,qhi=calibration(train)
 out=test[['signal_date','expiry','target_date','spot','target_close','target_return']].copy()
 for name,arr in pred.items(): out[f'pred_return_{name}']=arr; out[f'pred_close_{name}']=out.spot.to_numpy()*(1+arr); out[f'pred_regime_{name}']=[regime(x) for x in arr]
 out['pred_low']=out.spot.to_numpy()*(1+pred['ensemble']+qlo); out['pred_high']=out.spot.to_numpy()*(1+pred['ensemble']+qhi); out['residual_q_low']=qlo; out['residual_q_high']=qhi
 return out
def metrics(p,m):
 pr=p[f'pred_return_{m}'].to_numpy(float); y=p.target_return.to_numpy(float)
 return {'mae':float(np.mean(np.abs(pr-y))),'rmse':float(np.sqrt(np.mean((pr-y)**2))),'direction_accuracy':float(np.mean((pr>=0)==(y>=0))),'regime_accuracy':float(np.mean([regime(a)==regime(b) for a,b in zip(pr,y)])),'correlation':float(np.corrcoef(pr,y)[0,1]) if len(pr)>1 and np.std(pr)>0 and np.std(y)>0 else np.nan,'samples':len(p)}
def run_wfo(ds):
 windows=[('2021-01-01','2021-12-31'),('2022-01-01','2022-12-31'),('2023-01-01','2023-12-31'),('2024-01-01','2024-12-31'),('2025-01-01','2025-12-31'),('2026-01-01','2026-03-31')]; outs=[]; rows=[]; years=int(CFG['rolling_train_years'])
 for i,(a,b) in enumerate(windows,1):
  ts,te=pd.Timestamp(a),pd.Timestamp(b); tr=ds[(ds.signal_date>=ts-pd.DateOffset(years=years))&(ds.signal_date<ts)]; test=ds[(ds.signal_date>=ts)&(ds.signal_date<=te)]
  if len(tr)<60 or test.empty:continue
  p=fit_predict(tr,test); p['window']=i; outs.append(p)
  for m in MODELS:rows.append({'window':i,'model':m,**metrics(p,m)})
 return pd.concat(outs,ignore_index=True),pd.DataFrame(rows)
def run_split(ds):
 cut=ds.signal_date.min()+(ds.signal_date.max()-ds.signal_date.min())*.7; tr=ds[ds.signal_date<=cut]; test=ds[ds.signal_date>cut]; p=fit_predict(tr,test); return p,pd.DataFrame([{'model':m,**metrics(p,m)} for m in MODELS])
def run_new_oos(ds):
 start=pd.Timestamp(CFG['new_oos_start']); tr=ds[ds.signal_date<start]; test=ds[ds.signal_date>=start]
 if test.empty:raise RuntimeError('New OOS dataset is empty')
 p=fit_predict(tr,test); return p,pd.DataFrame([{'model':m,**metrics(p,m)} for m in MODELS])
def latest_forecast(m,ds,model_name):
 c=latest_open_cycle(m); train=ds[ds.signal_date<pd.Timestamp(CFG['new_oos_start'])].copy()
 if c is None or len(train)<60:raise RuntimeError('Insufficient data for latest forecast')
 row=feature_row(m,c['latest_feature_date']); model=regressors()[model_name]; model.fit(train[FEATURES],train.target_return.astype(float)); pred=float(model.predict(pd.DataFrame([row]))[0]); qlo,qhi=calibration(train); s=c['spot']
 return {'available':True,'signal_date':str(c['signal_date'].date()),'latest_feature_date':str(c['latest_feature_date'].date()),'next_expiry':str(c['expiry'].date()),'spot':s,'production_model':model_name,'predicted_return':pred,'predicted_close':s*(1+pred),'predicted_low':s*(1+pred+qlo),'predicted_high':s*(1+pred+qhi),'regime':regime(pred),'prediction_interval':float(CFG['prediction_interval']),'calibration_q_low':qlo,'calibration_q_high':qhi}
def main():
 outdir=ROOT/'v6.1/research'; outdir.mkdir(parents=True,exist_ok=True); m=load_market(); ds=build_dataset(m,completed_cycles(m)); end=pd.Timestamp(CFG['history_end'])
 if pd.Timestamp(m.dates[-1])<end:raise RuntimeError('Market data ends before configured history_end')
 ds=ds[ds.signal_date<end].copy()
 if len(ds)<150:raise RuntimeError(f'Only {len(ds)} completed cycles; need >=150')
 wfo_p,wfo_r=run_wfo(ds); split_p,split_r=run_split(ds[ds.signal_date<pd.Timestamp(CFG['new_oos_start'])]); oos_p,oos_r=run_new_oos(ds)
 wfo_p.to_csv(outdir/'walk_forward_predictions.csv',index=False); split_p.to_csv(outdir/'split_70_30_predictions.csv',index=False); oos_p.to_csv(outdir/'new_oos_predictions.csv',index=False); wfo_r.to_csv(outdir/'walk_forward_model_metrics.csv',index=False); split_r.to_csv(outdir/'split_70_30_model_metrics.csv',index=False); oos_r.to_csv(outdir/'new_oos_model_metrics.csv',index=False)
 production=str(wfo_r.groupby('model').rmse.mean().sort_values().index[0]); latest=latest_forecast(m,ds,production); json.dump(latest,open(outdir/'latest_forecast.json','w'),indent=2)
 summary={'sample_count':int(len(ds)),'target':'NIFTY next-expiry close and return','history_start':str(ds.signal_date.min().date()),'history_end':str(end.date()),'new_oos_start':str(pd.Timestamp(CFG['new_oos_start']).date()),'selected_production_model':production,'walk_forward':{'best_model_by_mean_rmse':production},'split_70_30':split_r.to_dict(orient='records'),'new_oos':oos_r.to_dict(orient='records')}; json.dump(summary,open(outdir/'summary.json','w'),indent=2); print(json.dumps({'samples':len(ds),'selected_model':production,'latest_forecast':latest},indent=2))
def target_return():return 'next-expiry NIFTY return'
def target_close():return 'next-expiry NIFTY close'
if __name__=='__main__':main()
