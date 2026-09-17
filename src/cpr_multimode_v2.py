from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

STRATEGIES = (
    'camarilla_inside_reversal',
    'mean_reversion_r1_pdh_s1_pdl',
    'narrow_cpr_breakout',
    'virgin_cpr_reversal',
    'mtf_cpr_daytrade',
    'inside_ascending_descending',
    'masterclass_regime_open',
    'weekly_cpr_masterclass',
)

@dataclass(frozen=True)
class CPRMultiModeConfig:
    narrow_pct: float = 0.15
    medium_pct: float = 0.30
    break_buffer_bps: float = 2.0
    virgin_age: int = 3
    max_swing_days: int = 10
    btst_long_only: bool = True


def levels(src: pd.DataFrame, prefix: str) -> pd.DataFrame:
    z = src.copy()
    p = prefix
    z[f'{p}pivot'] = (z.high + z.low + z.close) / 3.0
    z[f'{p}bc'] = (z.high + z.low) / 2.0
    z[f'{p}tc'] = 2*z[f'{p}pivot'] - z[f'{p}bc']
    z[f'{p}cpr_low'] = z[[f'{p}bc', f'{p}tc']].min(axis=1)
    z[f'{p}cpr_high'] = z[[f'{p}bc', f'{p}tc']].max(axis=1)
    z[f'{p}r1'] = 2*z[f'{p}pivot'] - z.low
    z[f'{p}s1'] = 2*z[f'{p}pivot'] - z.high
    z[f'{p}r2'] = z[f'{p}pivot'] + (z.high-z.low)
    z[f'{p}s2'] = z[f'{p}pivot'] - (z.high-z.low)
    z[f'{p}r3'] = z.high + 2*(z[f'{p}pivot']-z.low)
    z[f'{p}s3'] = z.low - 2*(z.high-z[f'{p}pivot'])
    z[f'{p}cam_r3'] = z.close + (z.high-z.low)*1.1/4.0
    z[f'{p}cam_s3'] = z.close - (z.high-z.low)*1.1/4.0
    z[f'{p}width'] = z[f'{p}cpr_high'] - z[f'{p}cpr_low']
    z[f'{p}range'] = z.high-z.low
    z[f'{p}width_ratio'] = z[f'{p}width'] / z[f'{p}range'].replace(0, np.nan)
    return z


def prior_period(daily: pd.DataFrame, period_col: str, prefix: str) -> pd.DataFrame:
    p = (daily.groupby(['symbol', period_col], as_index=False)
         .agg(open=('open','first'), high=('high','max'), low=('low','min'), close=('close','last'))
         .sort_values(['symbol', period_col]))
    for c in ['open','high','low','close']:
        p[c] = p.groupby('symbol')[c].shift(1)
    p = levels(p, prefix)
    p[f'{prefix}prev_high'] = p.high
    p[f'{prefix}prev_low'] = p.low
    return p


def width_class(r: pd.Series, cfg: CPRMultiModeConfig) -> pd.Series:
    return pd.Series(np.select([r<=0.03, r<=cfg.narrow_pct, r<=cfg.medium_pct],
                               ['single_line','narrow','medium'], default='wide'), index=r.index)


def prepare_context(intraday: pd.DataFrame, cfg: CPRMultiModeConfig = CPRMultiModeConfig()):
    x = intraday.copy()
    x.timestamp = pd.to_datetime(x.timestamp)
    x['session'] = x.timestamp.dt.normalize()
    x = x.sort_values(['symbol','timestamp']).reset_index(drop=True)
    daily = (x.groupby(['symbol','session'], as_index=False)
               .agg(open=('open','first'), high=('high','max'), low=('low','min'), close=('close','last'))
               .sort_values(['symbol','session']))
    daily['week'] = daily.session.dt.to_period('W-FRI')
    daily['month'] = daily.session.dt.to_period('M')

    pdsrc = daily[['symbol','session','open','high','low','close']].copy()
    for c in ['open','high','low','close']:
        pdsrc[c] = pdsrc.groupby('symbol')[c].shift(1)
    d = levels(pdsrc, 'd_')
    d['d_pd_high'] = pdsrc.high
    d['d_pd_low'] = pdsrc.low
    d['d_ascending'] = d.groupby('symbol').d_cpr_low.diff() > 0
    d['d_descending'] = d.groupby('symbol').d_cpr_low.diff() < 0
    d['d_touched'] = (daily.high >= d.d_cpr_low) & (daily.low <= d.d_cpr_high)
    for lag in range(1, cfg.virgin_age+1):
        virgin = ~d.groupby('symbol').d_touched.shift(lag).fillna(True)
        d[f'd_virgin_{lag}_low'] = d.groupby('symbol').d_cpr_low.shift(lag).where(virgin)
        d[f'd_virgin_{lag}_high'] = d.groupby('symbol').d_cpr_high.shift(lag).where(virgin)

    w = prior_period(daily, 'week', 'w_')
    w['w_ascending'] = w.groupby('symbol').w_cpr_low.diff() > 0
    w['w_descending'] = w.groupby('symbol').w_cpr_low.diff() < 0
    w_touch = (daily.groupby(['symbol','week'], as_index=False)
                    .agg(high=('high','max'), low=('low','min'))
                    .merge(w[['symbol','week','w_cpr_low','w_cpr_high']], on=['symbol','week'], how='left'))
    w_touch['w_touched'] = (w_touch.high >= w_touch.w_cpr_low) & (w_touch.low <= w_touch.w_cpr_high)
    w = w.merge(w_touch[['symbol','week','w_touched']], on=['symbol','week'], how='left')
    w['w_virgin_low'] = w.groupby('symbol').w_cpr_low.shift(1).where(~w.groupby('symbol').w_touched.shift(1).fillna(True))
    w['w_virgin_high'] = w.groupby('symbol').w_cpr_high.shift(1).where(~w.groupby('symbol').w_touched.shift(1).fillna(True))

    m = prior_period(daily, 'month', 'm_')

    dcols = ['symbol','session'] + [c for c in d.columns if c not in ('symbol','session','open','high','low','close')]
    daily = daily.merge(d[dcols], on=['symbol','session'], how='left')
    wcols = ['symbol','week'] + [c for c in w.columns if c.startswith('w_') or c == 'w_touched']
    daily = daily.merge(w[list(dict.fromkeys(wcols))], on=['symbol','week'], how='left')
    mcols = ['symbol','month'] + [c for c in m.columns if c.startswith('m_')]
    daily = daily.merge(m[list(dict.fromkeys(mcols))], on=['symbol','month'], how='left')
    daily['d_width_class'] = width_class(daily.d_width_ratio, cfg)
    daily['w_width_class'] = width_class(daily.w_width_ratio, cfg)

    context_cols = ['symbol','session'] + [c for c in daily.columns if c.startswith(('d_','w_','m_'))] + ['d_width_class','w_width_class']
    context_cols = list(dict.fromkeys(context_cols))
    x = x.merge(daily[context_cols], on=['symbol','session'], how='left')
    return x, daily


def cross_up(z: pd.DataFrame, level: pd.Series) -> pd.Series:
    pc = z.groupby('symbol').close.shift(1)
    pl = level.groupby(z.symbol).shift(1)
    return (z.close > level) & (pc <= pl)


def cross_down(z: pd.DataFrame, level: pd.Series) -> pd.Series:
    pc = z.groupby('symbol').close.shift(1)
    pl = level.groupby(z.symbol).shift(1)
    return (z.close < level) & (pc >= pl)


def blank(index, timeframe):
    out = pd.DataFrame(index=index)
    out['signal'] = 0
    out['stop_level'] = np.nan
    out['target_level'] = np.nan
    out['setup_timeframe'] = timeframe
    out['setup_reason'] = ''
    return out


def put(out, mask, side, stop, target, reason):
    mask = mask.fillna(False)
    out.loc[mask,'signal'] = side
    out.loc[mask,'stop_level'] = stop.loc[mask]
    out.loc[mask,'target_level'] = target.loc[mask]
    out.loc[mask,'setup_reason'] = reason


def generate_intraday(x, strategy, cfg):
    z = x.sort_values(['symbol','timestamp']).reset_index(drop=True)
    out = blank(z.index, '5min')
    b = cfg.break_buffer_bps/10000.0
    near = z.d_width_class.isin(['single_line','narrow'])
    wabove = z.close > z.w_cpr_high
    wbelow = z.close < z.w_cpr_low
    upper = z[['d_r1','d_pd_high']].max(axis=1)
    lower = z[['d_s1','d_pd_low']].min(axis=1)
    if strategy == 'camarilla_inside_reversal':
        rin = z.d_cam_r3.between(z.d_cpr_low,z.d_cpr_high)
        sin = z.d_cam_s3.between(z.d_cpr_low,z.d_cpr_high)
        put(out, rin & (z.high>=z.d_cam_r3) & (z.close<z.d_cam_r3), -1, z.high*(1+b), z.d_pivot, 'Daily Cam R3 inside CPR rejection')
        put(out, sin & (z.low<=z.d_cam_s3) & (z.close>z.d_cam_s3), 1, z.low*(1-b), z.d_pivot, 'Daily Cam S3 inside CPR rejection')
    elif strategy == 'mean_reversion_r1_pdh_s1_pdl':
        put(out, (z.high>upper) & cross_down(z,upper), -1, z.high*(1+b), z.d_pivot, 'R1/PDH reversal')
        put(out, (z.low<lower) & cross_up(z,lower), 1, z.low*(1-b), z.d_pivot, 'S1/PDL reversal')
    elif strategy == 'narrow_cpr_breakout':
        put(out, near & cross_up(z,upper), 1, z.d_cpr_high*(1-b), z.d_r2, 'Narrow daily CPR breakout')
        put(out, near & cross_down(z,lower), -1, z.d_cpr_low*(1+b), z.d_s2, 'Narrow daily CPR breakdown')
    elif strategy == 'virgin_cpr_reversal':
        for lag in range(1,cfg.virgin_age+1):
            lo, hi = z[f'd_virgin_{lag}_low'], z[f'd_virgin_{lag}_high']
            put(out, (z.low<=lo) & (z.close>lo), 1, lo*(1-b), z.d_cpr_high, f'Virgin CPR lower rejection age {lag}')
            put(out, (z.high>=hi) & (z.close<hi), -1, hi*(1+b), z.d_cpr_low, f'Virgin CPR upper rejection age {lag}')
    elif strategy == 'mtf_cpr_daytrade':
        put(out, wabove & cross_up(z,upper), 1, z.d_cpr_high*(1-b), z.d_r2, 'Weekly-above daily breakout')
        put(out, wbelow & cross_down(z,lower), -1, z.d_cpr_low*(1+b), z.d_s2, 'Weekly-below daily breakdown')
    elif strategy == 'inside_ascending_descending':
        prev_lo = z.groupby('symbol').d_cpr_low.shift(1); prev_hi = z.groupby('symbol').d_cpr_high.shift(1)
        inside = (z.d_cpr_low>=prev_lo) & (z.d_cpr_high<=prev_hi)
        put(out, (near|inside) & z.d_ascending.fillna(False) & cross_up(z,upper), 1, z.d_cpr_high*(1-b), z.d_r2, 'Ascending/inside CPR breakout')
        put(out, (near|inside) & z.d_descending.fillna(False) & cross_down(z,lower), -1, z.d_cpr_low*(1+b), z.d_s2, 'Descending/inside CPR breakdown')
    elif strategy == 'masterclass_regime_open':
        t = z.timestamp.dt.time
        first30 = (t>=pd.Timestamp('09:15').time()) & (t<=pd.Timestamp('09:45').time())
        put(out, first30 & wabove & (z.open>z.d_r1) & (z.close<z.d_cpr_high), -1, z.high*(1+b), z.d_pivot, 'Opening excess fade above daily R1')
        put(out, first30 & wbelow & (z.open<z.d_s1) & (z.close>z.d_cpr_low), 1, z.low*(1-b), z.d_pivot, 'Opening excess fade below daily S1')
        put(out, first30 & near & wabove & cross_up(z,z.d_r1), 1, z.d_cpr_high*(1-b), z.d_r2, 'Narrow CPR continuation')
        put(out, first30 & near & wbelow & cross_down(z,z.d_s1), -1, z.d_cpr_low*(1+b), z.d_s2, 'Narrow CPR downside continuation')
    elif strategy == 'weekly_cpr_masterclass':
        valid = z.w_width_class.isin(['single_line','narrow'])
        put(out, valid & near & wabove & cross_up(z,upper), 1, z.d_cpr_high*(1-b), z.w_r2, 'Narrow daily + weekly CPR breakout')
        put(out, valid & near & wbelow & cross_down(z,lower), -1, z.d_cpr_low*(1+b), z.w_s2, 'Narrow daily + weekly CPR breakdown')
    else:
        raise ValueError(strategy)
    return pd.concat([z,out],axis=1)


def generate_daily(daily, strategy, mode, cfg):
    z = daily.sort_values(['symbol','session']).reset_index(drop=True)
    out = blank(z.index, '1D')
    b = cfg.break_buffer_bps/10000.0
    above_m = z.close > z.m_cpr_high
    below_m = z.close < z.m_cpr_low
    upper = z[['w_r1','w_prev_high']].max(axis=1)
    lower = z[['w_s1','w_prev_low']].min(axis=1)
    near_w = z.w_width_class.isin(['single_line','narrow'])
    prev_r1 = z.groupby('symbol').w_r1.shift(1)
    prev_s1 = z.groupby('symbol').w_s1.shift(1)
    if strategy == 'camarilla_inside_reversal':
        rin = z.w_cam_r3.between(z.w_cpr_low,z.w_cpr_high)
        sin = z.w_cam_s3.between(z.w_cpr_low,z.w_cpr_high)
        put(out, rin&(z.high>=z.w_cam_r3)&(z.close<z.w_cam_r3), -1, z.high*(1+b), z.w_pivot, 'Weekly Cam R3 rejection')
        put(out, sin&(z.low<=z.w_cam_s3)&(z.close>z.w_cam_s3), 1, z.low*(1-b), z.w_pivot, 'Weekly Cam S3 rejection')
    elif strategy == 'mean_reversion_r1_pdh_s1_pdl':
        put(out,(z.high>upper)&(z.close<upper),-1,z.high*(1+b),z.w_pivot,'Weekly R1/previous-week-high reversal')
        put(out,(z.low<lower)&(z.close>lower),1,z.low*(1-b),z.w_pivot,'Weekly S1/previous-week-low reversal')
    elif strategy == 'narrow_cpr_breakout':
        pc = z.groupby('symbol').close.shift(1)
        put(out,near_w&(z.close>upper)&(pc<=prev_r1),1,z.w_cpr_high*(1-b),z.w_r2,'Narrow weekly CPR breakout')
        put(out,near_w&(z.close<lower)&(pc>=prev_s1),-1,z.w_cpr_low*(1+b),z.w_s2,'Narrow weekly CPR breakdown')
    elif strategy == 'virgin_cpr_reversal':
        put(out,z.low<=z.w_virgin_low,1,z.w_virgin_low*(1-b),z.w_cpr_high,'Virgin weekly CPR lower reversal')
        put(out,z.high>=z.w_virgin_high,-1,z.w_virgin_high*(1+b),z.w_cpr_low,'Virgin weekly CPR upper reversal')
    elif strategy == 'mtf_cpr_daytrade':
        put(out,above_m&(z.close>upper),1,z.w_cpr_high*(1-b),z.w_r2,'Monthly-above + weekly breakout')
        put(out,below_m&(z.close<lower),-1,z.w_cpr_low*(1+b),z.w_s2,'Monthly-below + weekly breakdown')
    elif strategy == 'inside_ascending_descending':
        put(out,z.w_ascending.fillna(False)&(z.close>upper),1,z.w_cpr_high*(1-b),z.w_r2,'Ascending weekly CPR breakout')
        put(out,z.w_descending.fillna(False)&(z.close<lower),-1,z.w_cpr_low*(1+b),z.w_s2,'Descending weekly CPR breakdown')
    elif strategy == 'masterclass_regime_open':
        first_week = z.groupby('symbol').session.diff().isna() | z.week.ne(z.groupby('symbol').week.shift(1))
        put(out,first_week&(z.open>z.w_r1)&(z.close<z.w_cpr_high),-1,z.high*(1+b),z.w_pivot,'Weekly opening excess fade')
        put(out,first_week&(z.open<z.w_s1)&(z.close>z.w_cpr_low),1,z.low*(1-b),z.w_pivot,'Weekly opening downside fade')
    elif strategy == 'weekly_cpr_masterclass':
        valid = z.w_width_class.isin(['single_line','narrow'])
        put(out,valid&above_m&(z.close>upper),1,z.w_cpr_high*(1-b),z.w_r2,'Narrow weekly CPR + monthly breakout')
        put(out,valid&below_m&(z.close<lower),-1,z.w_cpr_low*(1+b),z.w_s2,'Narrow weekly CPR + monthly breakdown')
    else:
        raise ValueError(strategy)
    if mode=='btst' and cfg.btst_long_only:
        neg = out.signal<0
        out.loc[neg,['signal','stop_level','target_level']] = [0,np.nan,np.nan]
    return pd.concat([z,out],axis=1)
