from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CFG = yaml.safe_load((ROOT / 'v6.1/config.yaml').read_text())
TH = float(CFG['regime_threshold'])

FEATURES = [
    'ret1','ret5','ret10','ret20','ema10_gap','ema20_gap','ema50_gap',
    'ema10_20_gap','ema20_50_gap','rsi14','vol10','vol20','breakout20',
    'breakdown20','range60','slope20','atm_straddle_pct','skew_proxy','dte',
    'india_vix','india_vix_ret5','spx_ret5','usd_inr_ret5','brent_ret5',
    'bank_rel5','midcap_rel5','fii_net_z','dii_net_z','fii_fut_long_short',
    'pcr','flow_sentiment','news_sentiment','news_volume_log','news_neg_share'
]
REGIMES = ['BEARISH', 'RANGE_BOUND', 'BULLISH']


def regime(ret: float) -> str:
    if ret >= TH:
        return 'BULLISH'
    if ret <= -TH:
        return 'BEARISH'
    return 'RANGE_BOUND'


def load_market():
    from v5.run import Market

    o = pd.read_parquet(ROOT / 'data/cache/nifty_options_long.parquet')
    f = pd.read_parquet(ROOT / 'data/cache/nifty_futures_long.parquet')
    c = pd.read_parquet(ROOT / 'data/cache/v5_market_context.parquet') if (ROOT / 'data/cache/v5_market_context.parquet').exists() else None
    q = pd.read_parquet(ROOT / 'data/cache/v5_flow_context.parquet') if (ROOT / 'data/cache/v5_flow_context.parquet').exists() else None
    n = pd.read_csv(ROOT / 'data/cache/v5_news_daily.csv') if (ROOT / 'data/cache/v5_news_daily.csv').exists() else None
    if n is not None:
        n.columns = [str(x).strip().lower() for x in n.columns]
    if c is not None and 'date' not in c.columns and c.index.name == 'date':
        c = c.reset_index()
    if q is not None and 'date' not in q.columns and q.index.name == 'date':
        q = q.reset_index()
    context = pd.concat([x for x in (c, q) if x is not None], ignore_index=True) if (c is not None or q is not None) else None
    return Market(o, f, context, n)


def spot_on_or_before(m, d):
    d = pd.Timestamp(d)
    if d in m.spot:
        return float(m.spot[d]), d
    z = [x for x in m.dates if x <= d]
    if not z:
        return np.nan, None
    x = pd.Timestamp(z[-1])
    return float(m.spot[x]), x


def next_expiry(m, d):
    d = pd.Timestamp(d)
    exps = sorted(set(sum((list(v) for v in m.expiries.values()), [])))
    for e in exps:
        if e > d:
            return pd.Timestamp(e)
    return None


def completed_cycles(m):
    exps = sorted(set(sum((list(v) for v in m.expiries.values()), [])))
    out = []
    for i, e in enumerate(exps):
        if i == 0:
            continue
        prev = pd.Timestamp(exps[i - 1]); e = pd.Timestamp(e)
        candidates = [d for d in m.dates if prev < d < e]
        if not candidates:
            continue
        signal_date = pd.Timestamp(candidates[0])
        target, target_date = spot_on_or_before(m, e)
        if target_date is None or target_date <= signal_date:
            continue
        spot = float(m.spot[signal_date])
        if np.isfinite(spot) and np.isfinite(target):
            out.append({
                'signal_date': signal_date,
                'expiry': e,
                'target_date': target_date,
                'spot': spot,
                'target_close': target,
                'target_return': target / spot - 1.0,
            })
    return pd.DataFrame(out).drop_duplicates('expiry').sort_values('signal_date').reset_index(drop=True)


def latest_open_cycle(m):
    latest = pd.Timestamp(m.dates[-1])
    prev_candidates = [e for e in sorted(set(sum((list(v) for v in m.expiries.values()), []))) if e < latest]
    if not prev_candidates:
        return None
    prev = pd.Timestamp(prev_candidates[-1])
    e = next_expiry(m, latest)
    if e is None or e <= latest:
        return None
    signal_dates = [d for d in m.dates if prev < d <= latest]
    if not signal_dates:
        return None
    signal_date = pd.Timestamp(signal_dates[0])
    return {
        'signal_date': signal_date,
        'latest_feature_date': latest,
        'expiry': pd.Timestamp(e),
        'spot': float(m.spot[latest]),
    }


def feature_row(m, d):
    from v5 import run as base
    f = base.make_features(m, pd.Timestamp(d))
    if f is None:
        return None
    c = m.asof_context(pd.Timestamp(d))
    f['usd_inr_ret5'] = c.get('usd_inr_ret5', c.get('INR=X_ret5', np.nan))
    f['brent_ret5'] = c.get('brent_ret5', c.get('BZ=F_ret5', np.nan))
    f['fii_net_z'] = c.get('fii_net_z', np.nan)
    f['dii_net_z'] = c.get('dii_net_z', np.nan)
    f['fii_fut_long_short'] = c.get('fii_fut_long_short', np.nan)
    f['pcr'] = c.get('pcr', np.nan)
    f['flow_sentiment'] = c.get('sentiment_score', np.nan)
    return {k: (float(f[k]) if pd.notna(f.get(k, np.nan)) else np.nan) for k in FEATURES}


def build_dataset(m, cycles):
    rows = []
    for r in cycles.itertuples(index=False):
        feats = feature_row(m, r.signal_date)
        if feats is None:
            continue
        rows.append(dict(
            feats,
            signal_date=pd.Timestamp(r.signal_date),
            expiry=pd.Timestamp(r.expiry),
            target_date=pd.Timestamp(r.target_date),
            spot=float(r.spot),
            target_close=float(r.target_close),
            target_return=float(r.target_return),
        ))
    return pd.DataFrame(rows).sort_values('signal_date').reset_index(drop=True)


def build_latest_row(m, cycle):
    feats = feature_row(m, cycle['latest_feature_date'])
    if feats is None:
        raise RuntimeError('Unable to construct latest NIFTY prediction features')
    return pd.DataFrame([dict(feats, signal_date=cycle['signal_date'], expiry=cycle['expiry'], spot=cycle['spot'])])


def make_regressors():
    seed = int(CFG['random_state'])
    n = int(CFG['n_estimators']); depth = int(CFG['max_depth']); leaf = int(CFG['min_samples_leaf'])
    return {
        'RIDGE': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=10.0)),
        'HGBR': make_pipeline(SimpleImputer(strategy='median'), HistGradientBoostingRegressor(max_iter=250, learning_rate=0.035, max_leaf_nodes=15, l2_regularization=0.5, random_state=seed)),
        'RF': make_pipeline(SimpleImputer(strategy='median'), RandomForestRegressor(n_estimators=n, max_depth=depth, min_samples_leaf=leaf, random_state=seed, n_jobs=-1)),
        'ET': make_pipeline(SimpleImputer(strategy='median'), ExtraTreesRegressor(n_estimators=n, max_depth=depth, min_samples_leaf=leaf, random_state=seed, n_jobs=-1)),
    }


def make_classifiers():
    seed = int(CFG['random_state'])
    n = int(CFG['n_estimators']); depth = int(CFG['max_depth']); leaf = int(CFG['min_samples_leaf'])
    return {
        'LOGIT': make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(max_iter=2000, C=0.5, multi_class='auto')),
        'HGC': make_pipeline(SimpleImputer(strategy='median'), HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.5, random_state=seed)),
        'RFC': make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(n_estimators=n, max_depth=depth, min_samples_leaf=leaf, random_state=seed, n_jobs=-1, class_weight='balanced_subsample')),
    }


def fit_predict(train, test):
    Xtr = train[FEATURES]; Xte = test[FEATURES]; y = train['target_return'].astype(float)
    models = make_regressors(); pred = {}
    for name, model in models.items():
        model.fit(Xtr, y); pred[name] = model.predict(Xte)
    pred['ENSEMBLE'] = np.mean(np.column_stack([pred[k] for k in models]), axis=1)

    labels = y.apply(regime)
    class_pred = {}
    proba_stack = []
    for name, model in make_classifiers().items():
        model.fit(Xtr, labels)
        classes = list(model.classes_)
        p = model.predict_proba(Xte)
        full = np.zeros((len(Xte), len(REGIMES)))
        for j, c in enumerate(classes):
            full[:, REGIMES.index(c)] = p[:, j]
        proba_stack.append(full)
        class_pred[name] = [REGIMES[i] for i in np.argmax(full, axis=1)]
    mean_prob = np.mean(proba_stack, axis=0)
    class_pred['CLASSIFIER_ENSEMBLE'] = [REGIMES[i] for i in np.argmax(mean_prob, axis=1)]

    out = test[['signal_date','expiry','target_date','spot','target_close','target_return']].copy()
    for name, arr in pred.items():
        out[f'pred_return_{name.lower()}'] = arr
        out[f'pred_close_{name.lower()}'] = out['spot'] * (1 + arr)
        out[f'pred_regime_{name.lower()}'] = [regime(x) for x in arr]
    out['pred_regime_classifier_ensemble'] = class_pred['CLASSIFIER_ENSEMBLE']
    for i, rg in enumerate(REGIMES):
        out[f'prob_{rg.lower()}'] = mean_prob[:, i]

    # Prediction interval comes from in-sample residual calibration of the ensemble.
    ensemble_train = np.mean(np.column_stack([
        make_regression_train_prediction(train, make_regressors()[k], y)[0] for k in make_regressors()
    ]), axis=1)
    resid = y.to_numpy() - ensemble_train
    alpha = 1.0 - float(CFG['prediction_interval'])
    lo = float(np.nanquantile(resid, alpha / 2))
    hi = float(np.nanquantile(resid, 1 - alpha / 2))
    ens = out['pred_return_ensemble'].to_numpy()
    out['pred_low'] = out['spot'] * (1 + ens + lo)
    out['pred_high'] = out['spot'] * (1 + ens + hi)
    out['residual_q_low'] = lo; out['residual_q_high'] = hi
    return out


def make_regression_train_prediction(train, model, y):
    model.fit(train[FEATURES], y)
    return model.predict(train[FEATURES]), model


def metrics(pred_df, model_name='ensemble'):
    pr = pred_df[f'pred_return_{model_name}'].astype(float).to_numpy()
    y = pred_df['target_return'].astype(float).to_numpy()
    mae = float(np.mean(np.abs(pr - y)))
    rmse = float(np.sqrt(np.mean((pr - y) ** 2)))
    dir_acc = float(np.mean((pr >= 0) == (y >= 0)))
    regime_acc = float(np.mean(np.array([regime(x) for x in pr]) == np.array([regime(x) for x in y])))
    corr = float(np.corrcoef(pr, y)[0, 1]) if len(pr) > 1 and np.std(pr) > 0 and np.std(y) > 0 else np.nan
    return {'mae': mae, 'rmse': rmse, 'direction_accuracy': dir_acc, 'regime_accuracy': regime_acc, 'correlation': corr, 'samples': int(len(pred_df))}


def run_wfo(ds):
    windows = [
        ('2021-01-01','2021-12-31'),('2022-01-01','2022-12-31'),('2023-01-01','2023-12-31'),
        ('2024-01-01','2024-12-31'),('2025-01-01','2025-12-31'),('2026-01-01','2026-03-31')
    ]
    outputs=[]; rows=[]
    for idx,(start,end) in enumerate(windows,1):
        ts=pd.Timestamp(start); te=pd.Timestamp(end)
        train=ds[(ds.signal_date>=ts-pd.DateOffset(years=int(CFG['rolling_train_years'])))&(ds.signal_date<ts)].copy()
        test=ds[(ds.signal_date>=ts)&(ds.signal_date<=te)].copy()
        if len(train)<60 or test.empty:
            continue
        p=fit_predict(train,test); p['window']=idx; outputs.append(p)
        for model in ['ensemble','ridge','hgbr','rf','et']:
            m=metrics(p,model); rows.append({'window':idx,'model':model,**m})
        cm=metrics(p,'ensemble'); rows.append({'window':idx,'model':'classifier_ensemble','mae':np.nan,'rmse':np.nan,'direction_accuracy':float((p['pred_regime_classifier_ensemble'].to_numpy()==np.array([regime(x) for x in p.target_return])) .mean()),'regime_accuracy':float((p['pred_regime_classifier_ensemble'].to_numpy()==np.array([regime(x) for x in p.target_return])).mean()),'correlation':np.nan,'samples':len(p)})
    all_p=pd.concat(outputs,ignore_index=True) if outputs else pd.DataFrame()
    report=pd.DataFrame(rows)
    return all_p, report


def run_split(ds):
    cut=ds.signal_date.min()+(ds.signal_date.max()-ds.signal_date.min())*0.7
    train=ds[ds.signal_date<=cut].copy(); test=ds[ds.signal_date>cut].copy()
    p=fit_predict(train,test)
    return p, pd.DataFrame([{'model':m,**metrics(p,m)} for m in ['ensemble','ridge','hgbr','rf','et']]+[{'model':'classifier_ensemble','mae':np.nan,'rmse':np.nan,'direction_accuracy':float((p['pred_regime_classifier_ensemble'].to_numpy()==np.array([regime(x) for x in p.target_return])).mean()),'regime_accuracy':float((p['pred_regime_classifier_ensemble'].to_numpy()==np.array([regime(x) for x in p.target_return])).mean()),'correlation':np.nan,'samples':len(p)}])


def run_new_oos(ds):
    start=pd.Timestamp(CFG['new_oos_start'])
    train=ds[ds.signal_date<start].copy(); test=ds[ds.signal_date>=start].copy()
    p=fit_predict(train,test)
    return p, pd.DataFrame([{'model':m,**metrics(p,m)} for m in ['ensemble','ridge','hgbr','rf','et']]+[{'model':'classifier_ensemble','mae':np.nan,'rmse':np.nan,'direction_accuracy':float((p['pred_regime_classifier_ensemble'].to_numpy()==np.array([regime(x) for x in p.target_return])).mean()),'regime_accuracy':float((p['pred_regime_classifier_ensemble'].to_numpy()==np.array([regime(x) for x in p.target_return])).mean()),'correlation':np.nan,'samples':len(p)}])


def select_production_model(wfo_report, split_report):
    pool=pd.concat([wfo_report,split_report],ignore_index=True)
    reg=pool[pool.model.isin(['ensemble','ridge','hgbr','rf','et'])].groupby('model').agg(mae=('mae','mean'),rmse=('rmse','mean'),direction_accuracy=('direction_accuracy','mean'),regime_accuracy=('regime_accuracy','mean')).reset_index()
    reg['score_rank_mae']=reg['mae'].rank(method='min',ascending=True)
    reg['score_rank_dir']=reg['direction_accuracy'].rank(method='min',ascending=False)
    reg['score']=reg['score_rank_mae']+reg['score_rank_dir']
    return reg.sort_values(['score','mae','rmse']).iloc[0]['model'], reg.sort_values(['score','mae','rmse'])


def latest_forecast(m, ds, selected_model):
    cycle=latest_open_cycle(m)
    if cycle is None:
        return {'available':False,'reason':'No incomplete next-expiry cycle found'}
    row=build_latest_row(m,cycle)
    train=ds[ds.signal_date<=ds.signal_date.max()].copy()
    models=make_regressors()
    pred_by={}
    for name, model in models.items():
        model.fit(train[FEATURES],train.target_return)
        pred_by[name]=float(model.predict(row[FEATURES])[0])
    if selected_model == 'ensemble':
        pred_return=float(np.mean(list(pred_by.values())))
    else:
        pred_return=pred_by[selected_model.upper()]
    labels=train.target_return.apply(regime)
    probs=[]
    for model in make_classifiers().values():
        model.fit(train[FEATURES],labels)
        full=np.zeros(len(REGIMES))
        for j,c in enumerate(model.classes_):
            full[REGIMES.index(c)]=model.predict_proba(row[FEATURES])[0,j]
        probs.append(full)
    pp=np.mean(probs,axis=0); regime_cls=REGIMES[int(np.argmax(pp))]
    # Calibration interval from ensemble in-sample residuals.
    trpred=[]
    for model in make_regressors().values():
        model.fit(train[FEATURES],train.target_return); trpred.append(model.predict(train[FEATURES]))
    ens_train=np.mean(np.column_stack(trpred),axis=1); resid=train.target_return.to_numpy()-ens_train
    alpha=1-float(CFG['prediction_interval'])
    lo=float(np.nanquantile(resid,alpha/2)); hi=float(np.nanquantile(resid,1-alpha/2))
    return {
        'available':True,
        'signal_date':str(cycle['signal_date'].date()),
        'feature_date':str(cycle['latest_feature_date'].date()),
        'next_expiry':str(cycle['expiry'].date()),
        'current_nifty':cycle['spot'],
        'selected_model':selected_model,
        'pred_return':pred_return,
        'pred_close':cycle['spot']*(1+pred_return),
        'pred_low':cycle['spot']*(1+pred_return+lo),
        'pred_high':cycle['spot']*(1+pred_return+hi),
        'pred_regime_regression':regime(pred_return),
        'pred_regime_classifier':regime_cls,
        'prob_bullish':float(pp[REGIMES.index('BULLISH')]),
        'prob_range_bound':float(pp[REGIMES.index('RANGE_BOUND')]),
        'prob_bearish':float(pp[REGIMES.index('BEARISH')]),
        'interval_level':float(CFG['prediction_interval']),
        'residual_q_low':lo,
        'residual_q_high':hi,
    }


def main():
    outdir=ROOT/'v6.1/research'; outdir.mkdir(parents=True,exist_ok=True)
    m=load_market(); cycles=completed_cycles(m); cycles=cycles[(cycles.signal_date>=pd.Timestamp(CFG['pre_oos_start']))&(cycles.signal_date<=pd.Timestamp(CFG['history_end']))].copy()
    ds=build_dataset(m,cycles)
    if len(ds)<150:
        raise RuntimeError(f'V6.1 dataset too small: {len(ds)}')
    if (ds.expiry>pd.Timestamp(CFG['history_end'])).any():
        raise RuntimeError('Completed dataset contains expiry beyond history_end')
    wfo_p,wfo_r=run_wfo(ds); split_p,split_r=run_split(ds[ds.signal_date<pd.Timestamp(CFG['new_oos_start'])]); oos_p,oos_r=run_new_oos(ds)
    selected,model_table=select_production_model(wfo_r,split_r)
    latest=latest_forecast(m,ds,selected)
    wfo_p.to_csv(outdir/'walk_forward_predictions.csv',index=False); split_p.to_csv(outdir/'split_70_30_predictions.csv',index=False); oos_p.to_csv(outdir/'new_oos_predictions.csv',index=False)
    wfo_r.to_csv(outdir/'walk_forward_model_metrics.csv',index=False); split_r.to_csv(outdir/'split_70_30_model_metrics.csv',index=False); oos_r.to_csv(outdir/'new_oos_model_metrics.csv',index=False); model_table.to_csv(outdir/'production_model_selection.csv',index=False)
    pd.DataFrame([latest]).to_json(outdir/'latest_forecast.json',orient='records',indent=2)
    summary={
        'sample_count':int(len(ds)),
        'feature_count':len(FEATURES),
        'target':'NIFTY next-expiry close and return',
        'completed_cycle_policy':'Only cycles whose expiry target is observable by history_end are used for evaluation; incomplete current cycle is excluded from OOS metrics.',
        'new_oos_start':str(CFG['new_oos_start']),
        'selected_production_model':selected,
        'walk_forward_aggregate':wfo_r.groupby('model').mean(numeric_only=True).reset_index().to_dict('records'),
        '70_30':split_r.to_dict('records'),
        'new_oos':oos_r.to_dict('records'),
        'latest_forecast':latest,
    }
    (outdir/'summary.json').write_text(json.dumps(summary,indent=2,default=str))

if __name__=='__main__':
    main()
