from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'v5') not in sys.path:
    sys.path.insert(0, str(ROOT / 'v5'))
from run import Market  # noqa: E402

CFG = yaml.safe_load((ROOT / 'v6.2/config.yaml').read_text())
OUT = ROOT / 'v6.2/research'
OUT.mkdir(parents=True, exist_ok=True)
SEED = int(CFG['seed'])
np.random.seed(SEED); torch.manual_seed(SEED)
TH = float(CFG['regime_threshold'])

ALIASES = {'INFOSYSTCH': 'INFY', 'HEROHONDA': 'HEROMOTOCO', 'BAJAJ-AUTO': 'BAJAJAUTO', 'ZOMATO': 'ETERNAL', 'MINDTREE': 'LTIM', 'TATAMTRDVR': 'TATAMOTORS'}


def regime(x: float) -> str:
    if x >= TH: return 'BULLISH'
    if x <= -TH: return 'BEARISH'
    return 'RANGE_BOUND'


def load_inputs():
    options = pd.read_parquet(ROOT / 'data/cache/nifty_options_long.parquet')
    futures = pd.read_parquet(ROOT / 'data/cache/nifty_futures_long.parquet')
    context = pd.read_parquet(ROOT / 'data/cache/v5_market_context.parquet')
    news = pd.read_csv(ROOT / 'data/cache/v5_news_daily.csv')
    context['date'] = pd.to_datetime(context['date']).dt.normalize() if 'date' in context.columns else pd.to_datetime(context.index).normalize()
    news.columns = [str(c).strip().lower() for c in news.columns]
    idx = pd.read_parquet(ROOT / 'data/cache/v62_nifty50_index.parquet')
    idx['date'] = pd.to_datetime(idx['date']).dt.normalize(); idx = idx.sort_values('date').drop_duplicates('date', keep='last')
    prices = pd.read_parquet(ROOT / 'data/cache/v62_constituent_prices.parquet')
    prices['symbol'] = prices['symbol'].astype(str).str.upper().replace(ALIASES)
    prices['date'] = pd.to_datetime(prices['date']).dt.normalize(); prices['close'] = pd.to_numeric(prices['close'], errors='coerce')
    prices = prices.dropna(subset=['date','close']).drop_duplicates(['symbol','date'], keep='last')
    weights = pd.read_csv(ROOT / 'data/cache/v62_constituent_weights.csv')
    weights.columns = [str(c).strip().upper() for c in weights.columns]; weights['DATE'] = pd.to_datetime(weights['DATE']).dt.normalize()
    for c in weights.columns:
        if c != 'DATE': weights[c] = pd.to_numeric(weights[c], errors='coerce').fillna(0.0)
    membership = pd.read_csv(ROOT / 'data/cache/v62_nifty50_membership.csv')
    membership.columns = [str(c).strip().lower() for c in membership.columns]
    membership['symbol'] = membership['symbol'].astype(str).str.upper().replace(ALIASES)
    membership['valid_from'] = pd.to_datetime(membership['valid_from']).dt.normalize(); membership['valid_to'] = pd.to_datetime(membership['valid_to'], errors='coerce').dt.normalize()
    flows = pd.read_parquet(ROOT / 'data/cache/v62_fii_dii.parquet')
    if len(flows): flows['date'] = pd.to_datetime(flows['date']).dt.normalize()
    return options, futures, context, news, idx, prices, weights, membership, flows


def cycles(index: pd.DataFrame, options: pd.DataFrame) -> pd.DataFrame:
    exps = sorted(pd.to_datetime(options['expiry']).dt.normalize().dropna().unique())
    dates = pd.DatetimeIndex(index['date'].sort_values().unique()); ix = index.set_index('date')['close']
    rows=[]
    for i in range(len(exps)-1):
        prev = pd.Timestamp(exps[i]); expiry = pd.Timestamp(exps[i+1])
        sig_dates = dates[(dates > prev) & (dates < expiry)]
        if not len(sig_dates): continue
        signal = pd.Timestamp(sig_dates[0]); target_dates = dates[dates <= expiry]
        if not len(target_dates): continue
        td = pd.Timestamp(target_dates[-1]);
        if td <= signal or signal not in ix.index or td not in ix.index: continue
        spot=float(ix.loc[signal]); target=float(ix.loc[td])
        rows.append({'signal_date':signal,'expiry':expiry,'target_date':td,'spot':spot,'target_close':target,'target_return':target/spot-1})
    return pd.DataFrame(rows).drop_duplicates('expiry').sort_values('signal_date').reset_index(drop=True)


def rsi(x, n=14):
    d=x.diff(); up=d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1+up/(dn+1e-12))


def asof_series(s: pd.Series, d: pd.Timestamp) -> float:
    z=s.loc[s.index < d]
    return float(z.iloc[-1]) if len(z) else np.nan


def index_features(idx: pd.DataFrame, d: pd.Timestamp, options: pd.DataFrame) -> dict:
    z=idx[idx.date<=d].copy().set_index('date').sort_index(); c=z.close.astype(float); ret=c.pct_change(); row={}
    for n in [1,5,10,20]: row[f'idx_ret{n}']=float(c.pct_change(n).iloc[-1])
    for n in [10,20,50]: row[f'idx_ema{n}_gap']=float(c.iloc[-1]/c.ewm(span=n,adjust=False).mean().iloc[-1]-1)
    row['idx_rsi14']=float(rsi(c).iloc[-1]/100); row['idx_vol10']=float(ret.rolling(10).std().iloc[-1]*math.sqrt(252)); row['idx_vol20']=float(ret.rolling(20).std().iloc[-1]*math.sqrt(252))
    row['idx_atr14']=float(((z.high-z.low).abs().rolling(14).mean()/c).iloc[-1]) if {'high','low'}.issubset(z.columns) else np.nan
    row['idx_range20']=float((c.iloc[-1]-c.shift(1).rolling(20).min().iloc[-1])/(c.shift(1).rolling(20).max().iloc[-1]-c.shift(1).rolling(20).min().iloc[-1]+1e-9))
    pivot = (z['high'].iloc[-2]+z['low'].iloc[-2]+z['close'].iloc[-2])/3 if {'high','low'}.issubset(z.columns) and len(z)>1 else np.nan
    bc=(z['high'].iloc[-2]+z['low'].iloc[-2])/2 if {'high','low'}.issubset(z.columns) and len(z)>1 else np.nan
    tc=2*pivot-bc if pd.notna(pivot) and pd.notna(bc) else np.nan
    row['cpr_width']=abs(tc-bc)/c.iloc[-1] if pd.notna(tc) else np.nan; row['cpr_position']=(c.iloc[-1]-min(bc,tc))/max(abs(tc-bc),1e-9) if pd.notna(tc) else np.nan
    row['idx_slope20']=float(np.polyfit(np.arange(min(20,len(c))),c.iloc[-min(20,len(c)):].values,1)[0]/c.iloc[-1])
    return row


def context_features(context: pd.DataFrame, news: pd.DataFrame, flows: pd.DataFrame, d: pd.Timestamp) -> dict:
    out={}; c=context[context.date < d].copy();
    if len(c):
        p=c.pivot_table(index='date',columns='series',values='close',aggfunc='last').sort_index()
        for sym in ['india_vix','nifty_bank','nifty_midcap','spx','usd_inr','brent']:
            col=sym if sym in p.columns else ({'usd_inr':'USDINR','brent':'BRENT','nifty_bank':'nifty_bank','nifty_midcap':'nifty_midcap'}.get(sym,sym))
            if col in p.columns: out[f'{sym}_ret5']=float(p[col].pct_change(5).iloc[-1])
    if len(flows):
        f=flows[flows.date < d].sort_values('date').set_index('date');
        for col in ['fii_net','dii_net']:
            if col in f:
                s=f[col].astype(float); out[col+'_1']=float(s.iloc[-1]); out[col+'_5']=float(s.tail(5).sum()); out[col+'_20']=float(s.tail(20).sum());
                base=s.tail(120); out[col+'_z']=float((s.iloc[-1]-base.mean())/(base.std()+1e-9)) if len(base)>20 else np.nan
    if len(news):
        n=news[pd.to_datetime(news['date']) < d].copy(); n['date']=pd.to_datetime(n['date']).dt.normalize();
        neg=[c for c in n.columns if 'neg' in c and pd.api.types.is_numeric_dtype(n[c])]; pos=[c for c in n.columns if 'pos' in c and pd.api.types.is_numeric_dtype(n[c])]; cnt=[c for c in n.columns if 'count' in c and pd.api.types.is_numeric_dtype(n[c])]
        if pos or neg or cnt:
            ps=float(n.tail(5)[pos].sum().sum()) if pos else 0.; ns=float(n.tail(5)[neg].sum().sum()) if neg else 0.; cc=float(n.tail(5)[cnt].sum().sum()) if cnt else 0.; out['news_sentiment_5']=(ps-ns)/max(cc,1.); out['news_volume_5']=math.log1p(max(cc,0)); out['news_neg_share_5']=ns/max(cc,1.)
    return out


def option_features(options: pd.DataFrame, d: pd.Timestamp, expiry: pd.Timestamp, spot: float) -> dict:
    o=options[(pd.to_datetime(options.date).dt.normalize()==d)&(pd.to_datetime(options.expiry).dt.normalize()==expiry)].copy()
    if o.empty:return {'opt_pcr_oi':np.nan,'opt_pcr_volume':np.nan,'opt_maxpain_gap':np.nan,'opt_call_wall_gap':np.nan,'opt_put_wall_gap':np.nan,'opt_atm_straddle_pct':np.nan,'opt_oi_concentration':np.nan,'opt_skew':np.nan}
    o['strike']=pd.to_numeric(o.strike,errors='coerce'); o['open_interest']=pd.to_numeric(o.open_interest,errors='coerce').fillna(0); o['volume']=pd.to_numeric(o.volume,errors='coerce').fillna(0); o['close']=pd.to_numeric(o.close,errors='coerce')
    ce=o[o.option_type.eq('CE')].dropna(subset=['strike']); pe=o[o.option_type.eq('PE')].dropna(subset=['strike'])
    call_oi=float(ce.open_interest.sum()); put_oi=float(pe.open_interest.sum()); call_v=float(ce.volume.sum()); put_v=float(pe.volume.sum()); res={'opt_pcr_oi':put_oi/max(call_oi,1),'opt_pcr_volume':put_v/max(call_v,1)}
    strikes=np.sort(pd.concat([ce.strike,pe.strike]).unique())
    pain=[]
    for k in strikes:
        pain.append(float((ce.open_interest*np.maximum(k-ce.strike,0)).sum()+(pe.open_interest*np.maximum(pe.strike-k,0)).sum()))
    maxpain=float(strikes[int(np.argmin(pain))]) if len(pain) else np.nan
    callwall=float(ce.groupby('strike').open_interest.sum().idxmax()) if len(ce) else np.nan; putwall=float(pe.groupby('strike').open_interest.sum().idxmax()) if len(pe) else np.nan
    atm=float(strikes[np.argmin(abs(strikes-spot))]) if len(strikes) else np.nan
    cp=float(ce.loc[ce.strike.eq(atm),'close'].mean()) if np.isfinite(atm) and len(ce.loc[ce.strike.eq(atm)]) else np.nan; pp=float(pe.loc[pe.strike.eq(atm),'close'].mean()) if np.isfinite(atm) and len(pe.loc[pe.strike.eq(atm)]) else np.nan
    near5=spot*1.03; nearp=spot*0.97; kc=float(ce.strike.iloc[np.argmin(abs(ce.strike-near5))]) if len(ce) else np.nan; kp=float(pe.strike.iloc[np.argmin(abs(pe.strike-nearp))]) if len(pe) else np.nan; cv=float(ce.loc[ce.strike.eq(kc),'close'].mean()) if np.isfinite(kc) else np.nan; pv=float(pe.loc[pe.strike.eq(kp),'close'].mean()) if np.isfinite(kp) else np.nan
    total_oi=o.open_interest.sum(); max_oi=o.open_interest.max();
    res.update({'opt_maxpain_gap':maxpain/spot-1 if np.isfinite(maxpain) else np.nan,'opt_call_wall_gap':callwall/spot-1 if np.isfinite(callwall) else np.nan,'opt_put_wall_gap':putwall/spot-1 if np.isfinite(putwall) else np.nan,'opt_atm_straddle_pct':(cp+pp)/spot if np.isfinite(cp) and np.isfinite(pp) else np.nan,'opt_oi_concentration':max_oi/max(total_oi,1),'opt_skew':(pv-cv)/max(pv+cv,1e-9) if np.isfinite(pv) and np.isfinite(cv) else np.nan})
    return res


def futures_features(futures: pd.DataFrame, d: pd.Timestamp, expiry: pd.Timestamp, spot: float) -> dict:
    f=futures[(pd.to_datetime(futures.date).dt.normalize()==d)&(pd.to_datetime(futures.expiry).dt.normalize()==expiry)]
    if f.empty:return {'fut_basis':np.nan,'fut_oi':np.nan,'fut_volume':np.nan}
    z=f.iloc[0]; return {'fut_basis':float(z.close)/spot-1,'fut_oi':float(z.open_interest),'fut_volume':float(z.volume)}


def constituent_features(prices: pd.DataFrame, weights: pd.DataFrame, membership: pd.DataFrame, d: pd.Timestamp) -> dict:
    wide=prices.pivot(index='date',columns='symbol',values='close').sort_index(); ret5=wide.pct_change(5); ret20=wide.pct_change(20); vol20=wide.pct_change().rolling(20).std()*math.sqrt(252)
    wr=weights[weights.DATE<=d].sort_values('DATE'); vec=wr.iloc[-1].drop(labels=['DATE']).astype(float) if len(wr) else pd.Series(dtype=float)
    if len(wr) and (d-wr.iloc[-1].DATE).days>int(CFG['constituents']['weights_max_staleness_days']): vec=pd.Series(0.0,index=wide.columns)
    active=membership[(membership.valid_from<=d)&((membership.valid_to.isna())|(membership.valid_to>d))].symbol.drop_duplicates().tolist(); active=[ALIASES.get(x,x) for x in active]
    cols=[c for c in wide.columns if c in set(active)]
    if not cols:return {'con_n':0}
    vec=vec.reindex(cols).fillna(0); vec[vec<0]=0
    if vec.sum()<=0: vec=pd.Series(1.0,index=cols)
    vec=vec/vec.sum(); r5=ret5[cols].loc[:d].iloc[-1]; r20=ret20[cols].loc[:d].iloc[-1]; v20=vol20[cols].loc[:d].iloc[-1]
    valid=r5.notna()&r20.notna(); vec=vec.where(valid,0); vec=vec/vec.sum() if vec.sum()>0 else vec
    signed=(vec*r5.fillna(0)); out={'con_n':int(valid.sum()),'con_weighted_ret5':float(signed.sum()),'con_weighted_ret20':float((vec*r20.fillna(0)).sum()),'con_weighted_vol20':float((vec*v20.fillna(0)).sum()),'con_breadth5':float((r5[valid]>0).mean()),'con_dispersion5':float(r5[valid].std()),'con_hhi':float((vec**2).sum()),'con_top10_shock':float(signed.abs().sort_values(ascending=False).head(10).sum())}
    top=vec.sort_values(ascending=False).head(5)
    for i,(sym,w) in enumerate(top.items(),1): out[f'con_top{i}_ret5']=float(r5.get(sym,np.nan))
    return out


def build_dataset(inp):
    options,futures,context,news,index,prices,weights,membership,flows=inp; cyc=cycles(index,options); rows=[]
    for r in cyc.itertuples(index=False):
        f=index_features(index,r.signal_date,options); f.update(option_features(options,r.signal_date,r.expiry,r.spot)); f.update(futures_features(futures,r.signal_date,r.expiry,r.spot)); f.update(context_features(context,news,flows,r.signal_date)); f.update(constituent_features(prices,weights,membership,r.signal_date)); f.update({'signal_date':r.signal_date,'expiry':r.expiry,'target_date':r.target_date,'spot':r.spot,'target_close':r.target_close,'target_return':r.target_return}); rows.append(f)
    df=pd.DataFrame(rows).sort_values('signal_date').reset_index(drop=True)
    return df


BASE_EXCLUDE={'signal_date','expiry','target_date','spot','target_close','target_return','con_n'}


def model_features(df):
    cols=[c for c in df.columns if c not in BASE_EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]
    return cols


def builders():
    return {
        'ridge': make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=5.0)),
        'rf': make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(n_estimators=300,max_depth=6,min_samples_leaf=3,random_state=SEED,n_jobs=-1)),
        'extra_trees': make_pipeline(SimpleImputer(strategy='median'),ExtraTreesRegressor(n_estimators=300,max_depth=7,min_samples_leaf=2,random_state=SEED,n_jobs=-1)),
        'hgbr': make_pipeline(SimpleImputer(strategy='median'),HistGradientBoostingRegressor(max_iter=250,learning_rate=.035,max_leaf_nodes=15,l2_regularization=1.0,random_state=SEED)),
        'xgboost': make_pipeline(SimpleImputer(strategy='median'),XGBRegressor(n_estimators=300,max_depth=3,learning_rate=.035,subsample=.8,colsample_bytree=.8,reg_lambda=2.0,objective='reg:squarederror',random_state=SEED,n_jobs=2)),
    }


def fit_tabular(train,test,features):
    preds={}; y=train.target_return.astype(float); Xtr=train[features]; Xte=test[features]
    for name,model in builders().items():
        if not CFG['models'].get(name,True): continue
        model.fit(Xtr,y); preds[name]=model.predict(Xte)
    return preds


class LSTMReg(nn.Module):
    def __init__(self,nf,hidden=48):
        super().__init__(); self.lstm=nn.LSTM(nf,hidden,batch_first=True); self.head=nn.Sequential(nn.Linear(hidden,24),nn.ReLU(),nn.Dropout(.1),nn.Linear(24,1))
    def forward(self,x): y,_=self.lstm(x); return self.head(y[:,-1,:]).squeeze(-1)


def lstm_predict(df,train_mask,test_mask,features):
    if not CFG['models'].get('lstm',True): return np.full(test_mask.sum(),np.nan)
    seq_len=int(CFG['sequence_length']); n=len(df); cut=np.where(train_mask)[0]
    if len(cut)<seq_len+40: return np.full(test_mask.sum(),np.nan)
    scaler=StandardScaler(); scaler.fit(SimpleImputer(strategy='median').fit_transform(df.loc[train_mask,features]))
    imp=SimpleImputer(strategy='median'); imp.fit(df.loc[train_mask,features]); mat=scaler.transform(imp.transform(df[features])).astype('float32')
    X=[]; y=[]; end=[]
    yy=df.target_return.to_numpy(float)
    for i in range(seq_len-1,n): X.append(mat[i-seq_len+1:i+1]); y.append(yy[i]); end.append(i)
    X=np.asarray(X,dtype='float32'); y=np.asarray(y,dtype='float32'); end=np.asarray(end)
    tr=end<cut[-1]+1; te=np.array([test_mask[i] for i in end],dtype=bool)
    if tr.sum()<50 or te.sum()==0:return np.full(test_mask.sum(),np.nan)
    tx=torch.tensor(X[tr]); ty=torch.tensor(y[tr]); model=LSTMReg(X.shape[-1]); opt=torch.optim.Adam(model.parameters(),lr=0.003,weight_decay=1e-4); lossfn=nn.HuberLoss(); best=np.inf; bad=0; state=None
    model.train()
    for _ in range(160):
        opt.zero_grad(); loss=lossfn(model(tx),ty); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); val=float(loss.detach());
        if val<best-1e-5: best=val; bad=0; state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else: bad+=1
        if bad>=15: break
    if state: model.load_state_dict(state)
    model.eval(); with_pred=[]
    with torch.no_grad(): with_pred=model(torch.tensor(X[te])).cpu().numpy()
    return with_pred


def predict(train,test,features):
    preds=fit_tabular(train,test,features)
    tm=np.zeros(len(train)+len(test),dtype=bool); tm[:len(train)]=True; mm=pd.concat([train,test],ignore_index=True); testmask=np.zeros(len(mm),dtype=bool); testmask[len(train):]=True
    lp=lstm_predict(mm,tm,testmask,features)
    preds['lstm']=lp
    if CFG['models'].get('ensemble',True):
        arr=[v for k,v in preds.items() if np.isfinite(v).all()]
        if arr: preds['ensemble']=np.mean(np.column_stack(arr),axis=1)
    return preds


def metrics(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); m=np.isfinite(y)&np.isfinite(p); y=y[m]; p=p[m]
    if len(y)<2:return {'mae':np.nan,'rmse':np.nan,'direction_accuracy':np.nan,'regime_accuracy':np.nan,'correlation':np.nan,'samples':int(len(y))}
    return {'mae':float(mean_absolute_error(y,p)),'rmse':float(math.sqrt(mean_squared_error(y,p))),'direction_accuracy':float(np.mean((y>=0)==(p>=0))),'regime_accuracy':float(np.mean([regime(a)==regime(b) for a,b in zip(y,p)])),'correlation':float(np.corrcoef(y,p)[0,1]) if np.std(y)>0 and np.std(p)>0 else np.nan,'samples':int(len(y))}


def purge_train(ds,test_start):
    start=pd.Timestamp(test_start); purge=pd.Timedelta(days=int(CFG['purge_days'])+int(CFG['embargo_days']))
    return ds[(ds.signal_date < start-purge)&(ds.target_date < start-purge)].copy()


def run_wfo(ds,features):
    windows=[('2021-01-01','2021-12-31'),('2022-01-01','2022-12-31'),('2023-01-01','2023-12-31'),('2024-01-01','2024-12-31'),('2025-01-01','2025-12-31'),('2026-01-01','2026-03-31')]; allp=[]; rows=[]; yrs=int(CFG['rolling_training_years'])
    for wi,(a,b) in enumerate(windows,1):
        ts,te=pd.Timestamp(a),pd.Timestamp(b); train=purge_train(ds[(ds.signal_date>=ts-pd.DateOffset(years=yrs))&(ds.signal_date<ts)],ts); test=ds[(ds.signal_date>=ts)&(ds.signal_date<=te)]
        if len(train)<int(CFG['min_samples']) or test.empty: continue
        pr=predict(train,test,features); base=test[['signal_date','expiry','target_date','spot','target_close','target_return']].copy(); base['window']=wi
        for name,p in pr.items():
            base[f'pred_return_{name}']=p; base[f'pred_close_{name}']=base.spot.to_numpy()*(1+p); base[f'pred_regime_{name}']=[regime(x) if np.isfinite(x) else 'NA' for x in p]; rows.append({'window':wi,'model':name,**metrics(test.target_return,p)})
        allp.append(base)
    return pd.concat(allp,ignore_index=True),pd.DataFrame(rows)


def run_split(ds,features):
    cut=ds.signal_date.min()+(ds.signal_date.max()-ds.signal_date.min())*.7; train=purge_train(ds[ds.signal_date<=cut],ds[ds.signal_date>cut].signal_date.min()); test=ds[ds.signal_date>cut]; pr=predict(train,test,features); base=test[['signal_date','expiry','target_date','spot','target_close','target_return']].copy();
    for name,p in pr.items(): base[f'pred_return_{name}']=p; base[f'pred_close_{name}']=base.spot.to_numpy()*(1+p); base[f'pred_regime_{name}']=[regime(x) if np.isfinite(x) else 'NA' for x in p]
    return base,pd.DataFrame([{'model':k,**metrics(test.target_return,v)} for k,v in pr.items()])


def run_new_oos(ds,features):
    start=pd.Timestamp(CFG['new_oos_start']); train=purge_train(ds[ds.signal_date<start],start); test=ds[ds.signal_date>=start];
    if test.empty or len(train)<int(CFG['min_samples']): raise RuntimeError('Insufficient training or OOS observations')
    pr=predict(train,test,features); base=test[['signal_date','expiry','target_date','spot','target_close','target_return']].copy();
    for name,p in pr.items(): base[f'pred_return_{name}']=p; base[f'pred_close_{name}']=base.spot.to_numpy()*(1+p); base[f'pred_regime_{name}']=[regime(x) if np.isfinite(x) else 'NA' for x in p]
    return base,pd.DataFrame([{'model':k,**metrics(test.target_return,v)} for k,v in pr.items()])


def calibration(train,features,model_name):
    n=max(30,int(len(train)*.2)); fit=train.iloc[:-n] if len(train)>n+50 else train.iloc[:max(30,len(train)-30)]; cal=train.iloc[len(fit):]
    if model_name not in builders() or cal.empty:return (-0.02,0.02)
    model=builders()[model_name]; model.fit(fit[features],fit.target_return); p=model.predict(cal[features]); r=cal.target_return.to_numpy()-p; alpha=1-float(CFG['prediction_interval']); return float(np.nanquantile(r,alpha/2)),float(np.nanquantile(r,1-alpha/2))


def latest_forecast(ds,features,production):
    inp=load_inputs(); options,futures,context,news,index,prices,weights,membership,flows=inp; index=index[index.date<=pd.Timestamp(CFG['history_end'])].copy(); signal=index.date.iloc[-1]; future_exps=sorted(pd.to_datetime(options.loc[pd.to_datetime(options.date).dt.normalize().eq(signal),'expiry']).dropna().unique())
    if future_exps: expiry=pd.Timestamp(future_exps[0])
    else:
        all_exps=sorted(pd.to_datetime(options.expiry).dropna().unique()); expiry=next((pd.Timestamp(x) for x in all_exps if pd.Timestamp(x)>signal),pd.Timestamp(all_exps[-1]))
    train=ds.copy(); row=pd.DataFrame([ds.iloc[-1][features].to_dict()]); pred=None
    if production=='lstm':
        mm=ds.copy(); mask=np.zeros(len(mm),bool); mask[-1]=True; p=lstm_predict(mm,np.ones(len(mm),bool),mask,features); pred=float(p[-1]) if len(p) else np.nan
    elif production=='ensemble':
        pr=fit_tabular(train,train.iloc[[-1]],features); vals=[v[-1] for k,v in pr.items() if k!='ensemble' and np.isfinite(v[-1])]; pred=float(np.mean(vals))
    else:
        model=builders()[production]; model.fit(train[features],train.target_return); pred=float(model.predict(row)[0])
    if not np.isfinite(pred): pred=0.0
    qlo,qhi=calibration(train,features,production if production in builders() else 'rf'); spot=float(index.iloc[-1].close)
    return {'available':True,'signal_date':str(signal.date()),'latest_feature_date':str(signal.date()),'next_expiry':str(expiry.date()),'spot':spot,'production_model':production,'predicted_return':pred,'predicted_close':spot*(1+pred),'predicted_low':spot*(1+pred+qlo),'predicted_high':spot*(1+pred+qhi),'regime':regime(pred),'prediction_interval':float(CFG['prediction_interval']),'calibration_q_low':qlo,'calibration_q_high':qhi}


def paper_strategy(row, options):
    pred=float(row.get('pred_return',0)); spot=float(row.spot); expiry=pd.Timestamp(row.expiry); d=pd.Timestamp(row.signal_date)
    o=options[(pd.to_datetime(options.date).dt.normalize()==d)&(pd.to_datetime(options.expiry).dt.normalize()==expiry)].copy()
    if o.empty:return {'action':'NO_DATA'}
    o['strike']=pd.to_numeric(o.strike); o['close']=pd.to_numeric(o.close)
    def near(typ,target):
        z=o[o.option_type.eq(typ)&(o.close>0)].copy()
        if z.empty:return None
        k=float(z.iloc[np.argmin(abs(z.strike.to_numpy()-target))].strike); p=float(z[z.strike.eq(k)].iloc[0].close); return k,p
    lot=25 if expiry<pd.Timestamp('2024-11-20') else (75 if expiry<pd.Timestamp('2026-01-01') else 65)
    if pred>TH:
        k1,p1=near('CE',spot); k2,p2=near('CE',max(spot*(1+pred),spot+250)); strategy='BULL_CALL_SPREAD'
        if k1 is None or k2 is None or k2<=k1:return {'action':'NO_TRADE'}
        max_loss=(p1-p2)*lot; entry=(p1-p2)*lot; return {'action':strategy,'lot_size':lot,'legs':[['BUY','CE',k1,p1],['SELL','CE',k2,p2]],'estimated_max_loss':max_loss,'entry_debit':entry,'capital_requirement':max_loss}
    if pred<-TH:
        k1,p1=near('PE',spot); k2,p2=near('PE',min(spot*(1+pred),spot-250)); strategy='BEAR_PUT_SPREAD'
        if k1 is None or k2 is None or k2>=k1:return {'action':'NO_TRADE'}
        max_loss=(p1-p2)*lot; return {'action':strategy,'lot_size':lot,'legs':[['BUY','PE',k1,p1],['SELL','PE',k2,p2]],'estimated_max_loss':max_loss,'entry_debit':max_loss,'capital_requirement':max_loss}
    lp,pp=near('PE',spot+(-abs(float(CFG['options']['otm_points'])))); lc,pc=near('CE',spot+abs(float(CFG['options']['otm_points']))); lpb,ppb=near('PE',spot-abs(float(CFG['options']['otm_points']))-float(CFG['options']['wing_width_points'])); lcb,pcb=near('CE',spot+abs(float(CFG['options']['otm_points']))+float(CFG['options']['wing_width_points']))
    if any(x is None for x in [lp,pp,lc,pc,lpb,ppb,lcb,pcb]): return {'action':'NO_TRADE'}
    credit=(pp+pc-ppb-pcb)*lot; max_loss=(float(CFG['options']['wing_width_points'])*lot)-credit
    return {'action':'IRON_CONDOR','lot_size':lot,'legs':[['SELL','PE',lp,pp],['BUY','PE',lpb,ppb],['SELL','CE',lc,pc],['BUY','CE',lcb,pcb]],'estimated_credit':credit,'estimated_max_loss':max_loss,'capital_requirement':max(0,max_loss)}


def write_dashboard(latest, summary):
    site=ROOT/'_site'; site.mkdir(exist_ok=True)
    forecast=json.dumps(latest,indent=2); sm=json.dumps(summary,indent=2)
    (site/'index.html').write_text(f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>V6.2 NIFTY Research</title><style>body{{font-family:Arial;max-width:1100px;margin:30px auto;padding:0 18px}}pre{{background:#f4f4f4;padding:14px;overflow:auto}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:6px}}</style></head><body><h1>V6.2 Multimodal NIFTY Expiry Research</h1><h2>Latest forecast</h2><pre>{forecast}</pre><h2>Research summary</h2><pre>{sm}</pre></body></html>''')
    (site/'latest_forecast.json').write_text(forecast); (site/'summary.json').write_text(sm)


def main():
    inp=load_inputs(); options,futures,context,news,index,prices,weights,membership,flows=inp; ds=build_dataset(inp); hist_end=pd.Timestamp(CFG['history_end']); ds=ds[ds.signal_date<hist_end].copy()
    if len(ds)<int(CFG['min_samples']): raise RuntimeError(f'Only {len(ds)} samples; need >= {CFG["min_samples"]}')
    features=model_features(ds); ds[features]=ds[features].replace([np.inf,-np.inf],np.nan); wfo_p,wfo_m=run_wfo(ds,features); split_p,split_m=run_split(ds[ds.signal_date<pd.Timestamp(CFG['new_oos_start'])],features); oos_p,oos_m=run_new_oos(ds,features)
    wfo_p.to_csv(OUT/'walk_forward_predictions.csv',index=False); wfo_m.to_csv(OUT/'walk_forward_model_metrics.csv',index=False); split_p.to_csv(OUT/'split_70_30_predictions.csv',index=False); split_m.to_csv(OUT/'split_70_30_model_metrics.csv',index=False); oos_p.to_csv(OUT/'new_oos_predictions.csv',index=False); oos_m.to_csv(OUT/'new_oos_model_metrics.csv',index=False)
    production=str(wfo_m.groupby('model').rmse.mean().sort_values().index[0]); latest=latest_forecast(ds,features,production); json.dump(latest,open(OUT/'latest_forecast.json','w'),indent=2)
    latest_trade=paper_strategy(pd.Series({'pred_return':latest['predicted_return'],'spot':latest['spot'],'expiry':pd.Timestamp(latest['next_expiry']),'signal_date':pd.Timestamp(latest['signal_date'])}),options)
    summary={'version':'V6.2','sample_count':int(len(ds)),'feature_count':len(features),'features':features,'selected_production_model':production,'history_end':str(hist_end.date()),'new_oos_start':str(pd.Timestamp(CFG['new_oos_start']).date()),'constituent_price_symbols':int(prices.symbol.nunique()),'membership_latest':str(membership.valid_from.max().date()),'weights_latest':str(weights.DATE.max().date()),'wfo':wfo_m.to_dict(orient='records'),'split_70_30':split_m.to_dict(orient='records'),'new_oos':oos_m.to_dict(orient='records'),'latest_forecast':latest,'latest_paper_trade':latest_trade}
    json.dump(summary,open(OUT/'summary.json','w'),indent=2); pd.DataFrame([latest_trade]).to_csv(OUT/'latest_paper_trade.csv',index=False); write_dashboard(latest,summary); print(json.dumps({'samples':len(ds),'feature_count':len(features),'selected_model':production,'latest_forecast':latest,'latest_paper_trade':latest_trade},indent=2))

if __name__=='__main__': main()
