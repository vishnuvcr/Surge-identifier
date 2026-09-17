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
class MultiModeConfig:
    narrow_pct: float = 0.15
    medium_pct: float = 0.30
    virgin_age_days: int = 3
    break_buffer_bps: float = 2.0
    max_swing_days: int = 10
    btst_long_only: bool = True


def _levels(x: pd.DataFrame, prefix: str = '') -> pd.DataFrame:
    z = x.copy()
    p = prefix
    z[f'{p}pivot'] = (z['high'] + z['low'] + z['close']) / 3.0
    z[f'{p}bc'] = (z['high'] + z['low']) / 2.0
    z[f'{p}tc'] = 2 * z[f'{p}pivot'] - z[f'{p}bc']
    z[f'{p}cpr_low'] = z[[f'{p}bc', f'{p}tc']].min(axis=1)
    z[f'{p}cpr_high'] = z[[f'{p}bc', f'{p}tc']].max(axis=1)
    z[f'{p}r1'] = 2 * z[f'{p}pivot'] - z['low']
    z[f'{p}s1'] = 2 * z[f'{p}pivot'] - z['high']
    z[f'{p}r2'] = z[f'{p}pivot'] + (z['high'] - z['low'])
    z[f'{p}s2'] = z[f'{p}pivot'] - (z['high'] - z['low'])
    z[f'{p}r3'] = z['high'] + 2 * (z[f'{p}pivot'] - z['low'])
    z[f'{p}s3'] = z['low'] - 2 * (z[f'{p}high'] - z[f'{p}pivot']) if f'{p}high' in z else z['low'] - 2 * (z['high'] - z[f'{p}pivot'])
    z[f'{p}width'] = z[f'{p}cpr_high'] - z[f'{p}cpr_low']
    z[f'{p}range'] = z['high'] - z['low']
    z[f'{p}width_ratio'] = z[f'{p}width'] / z[f'{p}range'].replace(0, np.nan)
    return z


def _cam(x: pd.DataFrame, prefix: str = '') -> pd.DataFrame:
    z = x.copy()
    r = z['high'] - z['low']
    z[f'{prefix}cam_r3'] = z['close'] + r * 1.1 / 4.0
    z[f'{prefix}cam_s3'] = z['close'] - r * 1.1 / 4.0
    return z


def _prev_period(daily: pd.DataFrame, period_col: str, prefix: str) -> pd.DataFrame:
    p = (daily.groupby(['symbol', period_col], as_index=False)
         .agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last'))
         .sort_values(['symbol', period_col]))
    for c in ('open', 'high', 'low', 'close'):
        p[c] = p.groupby('symbol')[c].shift(1)
    p = _levels(p, prefix)
    p = _cam(p, prefix)
    p[f'{prefix}prev_high'] = p['high']
    p[f'{prefix}prev_low'] = p['low']
    return p


def _width_class(ratio: pd.Series, cfg: MultiModeConfig) -> pd.Series:
    return pd.Series(np.select([
        ratio.le(0.03),
        ratio.le(cfg.narrow_pct),
        ratio.le(cfg.medium_pct),
    ], ['single_line', 'narrow', 'medium'], default='wide'), index=ratio.index)


def prepare_levels(intraday: pd.DataFrame, cfg: MultiModeConfig = MultiModeConfig()):
    x = intraday.copy()
    x['timestamp'] = pd.to_datetime(x['timestamp'])
    x['session'] = x['timestamp'].dt.normalize()
    x = x.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    daily = (x.groupby(['symbol', 'session'], as_index=False)
             .agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last'))
             .sort_values(['symbol', 'session']))

    # Prior-day CPR.
    dsrc = daily[['symbol', 'session', 'open', 'high', 'low', 'close']].copy()
    for c in ('open', 'high', 'low', 'close'):
        dsrc[c] = dsrc.groupby('symbol')[c].shift(1)
    d = _levels(dsrc, 'd_')
    d = _cam(d, 'd_')
    d['d_pd_high'] = dsrc['high']
    d['d_pd_low'] = dsrc['low']

    # A CPR is virgin on the next session when the session that formed it did not touch that CPR.
    prev_touch = ((daily['high'] >= d['d_cpr_low']) & (daily['low'] <= d['d_cpr_high']))
    d['d_prev_cpr_touched'] = prev_touch.groupby(d['symbol']).shift(1)
    d['d_virgin'] = ~d['d_prev_cpr_touched'].fillna(True)
    for lag in range(1, cfg.virgin_age_days + 1):
        flag = d.groupby('symbol')['d_virgin'].shift(lag).fillna(False)
        d[f'd_virgin_{lag}_low'] = d.groupby('symbol')['d_cpr_low'].shift(lag).where(flag)
        d[f'd_virgin_{lag}_high'] = d.groupby('symbol')['d_cpr_high'].shift(lag).where(flag)

    daily['week'] = daily['session'].dt.to_period('W-FRI')
    daily['month'] = daily['session'].dt.to_period('M')
    w = _prev_period(daily, 'week', 'w_')
    m = _prev_period(daily, 'month', 'm_')

    # Previous-period widths and directional CPRs.
    for frame, prefix in ((d, 'd'), (w, 'w'), (m, 'm')):
        frame[f'{prefix}_ascending'] = frame.groupby('symbol')[f'{prefix}_cpr_low'].diff() > 0
        frame[f'{prefix}_descending'] = frame.groupby('symbol')[f'{prefix}_cpr_low'].diff() < 0

    # Weekly virgin CPR: whether the prior week did not touch its own CPR.
    week_touch = daily.groupby(['symbol', 'week'], as_index=False).agg(high=('high', 'max'), low=('low', 'min'))
    w_touch = week_touch.merge(w[['symbol', 'week', 'w_cpr_low', 'w_cpr_high']], on=['symbol', 'week'], how='left')
    w_touch['touched'] = (w_touch['high'] >= w_touch['w_cpr_low']) & (w_touch['low'] <= w_touch['w_cpr_high'])
    w_touch['virgin'] = ~w_touch['touched'].fillna(True)
    w = w.merge(w_touch[['symbol', 'week', 'virgin']], on=['symbol', 'week'], how='left')
    w['w_virgin_low'] = w.groupby('symbol')['w_cpr_low'].shift(1).where(w.groupby('symbol')['virgin'].shift(1).fillna(False))
    w['w_virgin_high'] = w.groupby('symbol')['w_cpr_high'].shift(1).where(w.groupby('symbol')['virgin'].shift(1).fillna(False))

    daily = daily.merge(d.drop(columns=['symbol', 'session']), left_on=['symbol', 'session'], right_on=['symbol', 'session'], how='left')
    daily = daily.merge(w[['symbol', 'week', 'w_pivot', 'w_cpr_low', 'w_cpr_high', 'w_r1', 'w_s1', 'w_r2', 'w_s2', 'w_r3', 'w_s3', 'w_cam_r3', 'w_cam_s3', 'w_prev_high', 'w_prev_low', 'w_width_ratio', 'w_ascending', 'w_descending', 'w_virgin_low', 'w_virgin_high']], on=['symbol', 'week'], how='left')
    daily = daily.merge(m[['symbol', 'month', 'm_pivot', 'm_cpr_low', 'm_cpr_high', 'm_r1', 'm_s1', 'm_r2', 'm_s2', 'm_width_ratio']], on=['symbol', 'month'], how='left')
    daily['d_width_class'] = _width_class(daily['d_width_ratio'], cfg)
    daily['w_width_class'] = _width_class(daily['w_width_ratio'], cfg)

    x = x.merge(d, on=['symbol', 'session'], how='left', suffixes=('', '_d'))
    x['week'] = x['session'].dt.to_period('W-FRI')
    x['month'] = x['session'].dt.to_period('M')
    x = x.merge(daily[['symbol', 'session', 'w_pivot', 'w_cpr_low', 'w_cpr_high', 'w_r1', 'w_s1', 'w_r2', 'w_s2', 'w_r3', 'w_s3', 'w_cam_r3', 'w_cam_s3', 'w_prev_high', 'w_prev_low', 'w_width_ratio', 'w_ascending', 'w_descending', 'w_virgin_low', 'w_virgin_high', 'm_pivot', 'm_cpr_low', 'm_cpr_high', 'm_r1', 'm_s1', 'm_r2', 'm_s2', 'm_width_ratio', 'd_width_class', 'w_width_class']], on=['symbol', 'session'], how='left')
    return x, daily


def _cross_up(x, level):
    pc = x.groupby('symbol')['close'].shift(1)
    pl = level.groupby(x['symbol']).shift(1)
    return (x['close'] > level) & (pc <= pl)


def _cross_down(x, level):
    pc = x.groupby('symbol')['close'].shift(1)
    pl = level.groupby(x['symbol']).shift(1)
    return (x['close'] < level) & (pc >= pl)


def _set(out, mask, side, stop, target, reason):
    out.loc[mask, 'signal'] = side
    out.loc[mask, 'stop_level'] = stop[mask]
    out.loc[mask, 'target_level'] = target[mask]
    out.loc[mask, 'setup_reason'] = reason


def generate_intraday_signals(x: pd.DataFrame, strategy: str, cfg: MultiModeConfig):
    z = x.copy().sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    out = pd.DataFrame(index=z.index)
    out['signal'] = 0
    out['stop_level'] = np.nan
    out['target_level'] = np.nan
    out['setup_timeframe'] = '5min'
    out['setup_reason'] = ''
    buf = cfg.break_buffer_bps / 10000.0
    dwidth = z['d_width_class']
    near = dwidth.isin(['single_line', 'narrow'])
    weekly_above = z['close'] > z['w_cpr_high']
    weekly_below = z['close'] < z['w_cpr_low']

    if strategy == 'camarilla_inside_reversal':
        r = z['d_cam_r3'].between(z['d_cpr_low'], z['d_cpr_high'])
        s = z['d_cam_s3'].between(z['d_cpr_low'], z['d_cpr_high'])
        _set(out, r & (z['high'] >= z['d_cam_r3']) & (z['close'] < z['d_cam_r3']), -1, z['high'] * (1 + buf), z['d_pivot'], 'Camarilla R3 inside daily CPR rejection')
        _set(out, s & (z['low'] <= z['d_cam_s3']) & (z['close'] > z['d_cam_s3']), 1, z['low'] * (1 - buf), z['d_pivot'], 'Camarilla S3 inside daily CPR rejection')
    elif strategy == 'mean_reversion_r1_pdh_s1_pdl':
        upper = z[['d_r1', 'd_pd_high']].max(axis=1)
        lower = z[['d_s1', 'd_pd_low']].min(axis=1)
        _set(out, (z['high'] > upper) & _cross_down(z, upper), -1, z['high'] * (1 + buf), z['d_pivot'], 'R1/PDH rejection')
        _set(out, (z['low'] < lower) & _cross_up(z, lower), 1, z['low'] * (1 - buf), z['d_pivot'], 'S1/PDL rejection')
    elif strategy == 'narrow_cpr_breakout':
        upper = z[['d_r1', 'd_pd_high']].max(axis=1)
        lower = z[['d_s1', 'd_pd_low']].min(axis=1)
        _set(out, near & _cross_up(z, upper), 1, z['d_cpr_high'] * (1 - buf), z['d_r2'], 'Narrow daily CPR upside breakout')
        _set(out, near & _cross_down(z, lower), -1, z['d_cpr_low'] * (1 + buf), z['d_s2'], 'Narrow daily CPR downside breakout')
    elif strategy == 'virgin_cpr_reversal':
        for lag in range(1, cfg.virgin_age_days + 1):
            lo = z[f'd_virgin_{lag}_low']; hi = z[f'd_virgin_{lag}_high']
            _set(out, (z['low'] <= lo) & (z['close'] > lo), 1, lo * (1 - buf), z['d_cpr_high'], f'Virgin CPR lower-edge reversal age {lag}')
            _set(out, (z['high'] >= hi) & (z['close'] < hi), -1, hi * (1 + buf), z['d_cpr_low'], f'Virgin CPR upper-edge reversal age {lag}')
    elif strategy == 'mtf_cpr_daytrade':
        upper = z[['d_r1', 'd_pd_high']].max(axis=1); lower = z[['d_s1', 'd_pd_low']].min(axis=1)
        _set(out, weekly_above & _cross_up(z, upper), 1, z['d_cpr_high'] * (1 - buf), z['d_r2'], 'Weekly-above + daily breakout')
        _set(out, weekly_below & _cross_down(z, lower), -1, z['d_cpr_low'] * (1 + buf), z['d_s2'], 'Weekly-below + daily breakdown')
    elif strategy == 'inside_ascending_descending':
        prev_low = z.groupby('symbol')['d_cpr_low'].shift(1); prev_high = z.groupby('symbol')['d_cpr_high'].shift(1)
        inside = (z['d_cpr_low'] >= prev_low) & (z['d_cpr_high'] <= prev_high)
        upper = z[['d_r1', 'd_pd_high']].max(axis=1); lower = z[['d_s1', 'd_pd_low']].min(axis=1)
        _set(out, (near | inside) & z['d_ascending'].fillna(False) & _cross_up(z, upper), 1, z['d_cpr_high'] * (1 - buf), z['d_r2'], 'Ascending/inside CPR breakout')
        _set(out, (near | inside) & z['d_descending'].fillna(False) & _cross_down(z, lower), -1, z['d_cpr_low'] * (1 + buf), z['d_s2'], 'Descending/inside CPR breakdown')
    elif strategy == 'masterclass_regime_open':
        first30 = z['timestamp'].dt.time.between(pd.Timestamp('09:15').time(), pd.Timestamp('09:45').time())
        far_up = z['open'] > z['d_r1']; far_dn = z['open'] < z['d_s1']
        _set(out, first30 & weekly_above & far_up & (z['close'] < z['d_cpr_high']), -1, z['high'] * (1 + buf), z['d_pivot'], 'Opening excess above R1 fades toward CPR')
        _set(out, first30 & weekly_below & far_dn & (z['close'] > z['d_cpr_low']), 1, z['low'] * (1 - buf), z['d_pivot'], 'Opening excess below S1 fades toward CPR')
        _set(out, first30 & near & weekly_above & _cross_up(z, z['d_r1']), 1, z['d_cpr_high'] * (1 - buf), z['d_r2'], 'Narrow CPR continuation')
        _set(out, first30 & near & weekly_below & _cross_down(z, z['d_s1']), -1, z['d_cpr_low'] * (1 + buf), z['d_s2'], 'Narrow CPR downside continuation')
    elif strategy == 'weekly_cpr_masterclass':
        upper = z[['d_r1', 'd_pd_high']].max(axis=1); lower = z[['d_s1', 'd_pd_low']].min(axis=1)
        valid = ~z['w_width_ratio'].isna()
        _set(out, valid & near & weekly_above & _cross_up(z, upper), 1, z['d_cpr_high'] * (1 - buf), z['w_r2'], 'Daily narrow CPR + weekly context breakout')
        _set(out, valid & near & weekly_below & _cross_down(z, lower), -1, z['d_cpr_low'] * (1 + buf), z['w_s2'], 'Daily narrow CPR + weekly context breakdown')
    else:
        raise ValueError(strategy)
    return pd.concat([z, out], axis=1)


def generate_daily_signals(daily: pd.DataFrame, strategy: str, mode: str, cfg: MultiModeConfig):
    z = daily.copy().sort_values(['symbol', 'session']).reset_index(drop=True)
    out = pd.DataFrame(index=z.index)
    out['signal'] = 0
    out['stop_level'] = np.nan
    out['target_level'] = np.nan
    out['setup_timeframe'] = '1D'
    out['setup_reason'] = ''
    buf = cfg.break_buffer_bps / 10000.0
    above_m = z['close'] > z['m_cpr_high']; below_m = z['close'] < z['m_cpr_low']
    near_w = z['w_width_class'].isin(['single_line', 'narrow'])
    upper_w = z[['w_r1', 'w_prev_high']].max(axis=1); lower_w = z[['w_s1', 'w_prev_low']].min(axis=1)

    if strategy == 'camarilla_inside_reversal':
        r = z['w_cam_r3'].between(z['w_cpr_low'], z['w_cpr_high']); s = z['w_cam_s3'].between(z['w_cpr_low'], z['w_cpr_high'])
        _set(out, r & (z['high'] >= z['w_cam_r3']) & (z['close'] < z['w_cam_r3']), -1, z['high'] * (1 + buf), z['w_pivot'], 'Weekly Camarilla R3 rejection')
        _set(out, s & (z['low'] <= z['w_cam_s3']) & (z['close'] > z['w_cam_s3']), 1, z['low'] * (1 - buf), z['w_pivot'], 'Weekly Camarilla S3 rejection')
    elif strategy == 'mean_reversion_r1_pdh_s1_pdl':
        _set(out, (z['high'] > upper_w) & (z['close'] < upper_w), -1, z['high'] * (1 + buf), z['w_pivot'], 'Weekly R1/previous-week-high rejection')
        _set(out, (z['low'] < lower_w) & (z['close'] > lower_w), 1, z['low'] * (1 - buf), z['w_pivot'], 'Weekly S1/previous-week-low rejection')
    elif strategy == 'narrow_cpr_breakout':
        _set(out, near_w & (z['close'] > upper_w) & (z.groupby('symbol')['close'].shift(1) <= z.groupby('symbol')[upper_w.name if False else 'w_r1'].shift(1)), 1, z['w_cpr_high'] * (1 - buf), z['w_r2'], 'Narrow weekly CPR breakout')
        _set(out, near_w & (z['close'] < lower_w), -1, z['w_cpr_low'] * (1 + buf), z['w_s2'], 'Narrow weekly CPR breakdown')
    elif strategy == 'virgin_cpr_reversal':
        _set(out, z['low'] <= z['w_virgin_low'], 1, z['w_virgin_low'] * (1 - buf), z['w_cpr_high'], 'Virgin weekly CPR lower-edge reversal')
        _set(out, z['high'] >= z['w_virgin_high'], -1, z['w_virgin_high'] * (1 + buf), z['w_cpr_low'], 'Virgin weekly CPR upper-edge reversal')
    elif strategy == 'mtf_cpr_daytrade':
        _set(out, above_m & (z['close'] > upper_w), 1, z['w_cpr_high'] * (1 - buf), z['w_r2'], 'Monthly-above + weekly breakout')
        _set(out, below_m & (z['close'] < lower_w), -1, z['w_cpr_low'] * (1 + buf), z['w_s2'], 'Monthly-below + weekly breakdown')
    elif strategy == 'inside_ascending_descending':
        _set(out, z['w_ascending'].fillna(False) & (z['close'] > upper_w), 1, z['w_cpr_high'] * (1 - buf), z['w_r2'], 'Ascending weekly CPR breakout')
        _set(out, z['w_descending'].fillna(False) & (z['close'] < lower_w), -1, z['w_cpr_low'] * (1 + buf), z['w_s2'], 'Descending weekly CPR breakdown')
    elif strategy == 'masterclass_regime_open':
        first = z.groupby('symbol').cumcount() == 0
        # first trading day of each weekly period
        first_week = z.groupby(['symbol', z['session'].dt.to_period('W-FRI')]).cumcount().eq(0)
        _set(out, first_week & (z['open'] > z['w_r1']) & (z['close'] < z['w_cpr_high']), -1, z['high'] * (1 + buf), z['w_pivot'], 'Weekly opening excess fades')
        _set(out, first_week & (z['open'] < z['w_s1']) & (z['close'] > z['w_cpr_low']), 1, z['low'] * (1 - buf), z['w_pivot'], 'Weekly opening downside excess fades')
    elif strategy == 'weekly_cpr_masterclass':
        valid = z['w_width_class'].isin(['single_line', 'narrow'])
        _set(out, valid & above_m & (z['close'] > upper_w), 1, z['w_cpr_high'] * (1 - buf), z['w_r2'], 'Narrow weekly CPR + monthly context breakout')
        _set(out, valid & below_m & (z['close'] < lower_w), -1, z['w_cpr_low'] * (1 + buf), z['w_s2'], 'Narrow weekly CPR + monthly context breakdown')
    else:
        raise ValueError(strategy)

    if mode == 'btst' and cfg.btst_long_only:
        out.loc[out['signal'] < 0, 'signal'] = 0
        out.loc[out['signal'] < 0, ['stop_level', 'target_level']] = np.nan
    return pd.concat([z, out], axis=1)
