from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'v8/deep_iron_condor_engine.py'
s = ENGINE.read_text()

start = s.index('def strict_asof(')
end = s.index('def make_constituent_features(', start)
new_strict = '''def strict_asof(left_dates, right, date_col='date'):\n    """Strict point-in-time backward lookup without pandas merge_asof dtype fragility."""\n    left = pd.DatetimeIndex(pd.to_datetime(left_dates, errors='coerce'))\n    if left.tz is not None:\n        left = left.tz_localize(None)\n    left = left.normalize()\n    r = right.copy()\n    r[date_col] = pd.to_datetime(r[date_col], errors='coerce')\n    if hasattr(r[date_col].dt, 'tz') and r[date_col].dt.tz is not None:\n        r[date_col] = r[date_col].dt.tz_localize(None)\n    r[date_col] = r[date_col].dt.normalize()\n    r = r.dropna(subset=[date_col]).sort_values(date_col)\n    r = r.drop_duplicates(date_col, keep='last').rename(columns={date_col: '_date'})\n    out = pd.DataFrame(index=left)\n    out.index.name = 'date'\n    if r.empty:\n        return out\n    right_ns = r['_date'].to_numpy(dtype='datetime64[ns]').astype('int64')\n    left_ns = left.to_numpy(dtype='datetime64[ns]').astype('int64')\n    pos = np.searchsorted(right_ns, left_ns, side='left') - 1\n    valid = pos >= 0\n    safe = np.clip(pos, 0, len(r) - 1)\n    picked = r.iloc[safe].copy()\n    picked.index = left\n    if (~valid).any():\n        picked.loc[~valid, '_date'] = pd.NaT\n        for col in picked.columns:\n            if col != '_date':\n                picked.loc[~valid, col] = np.nan\n    return picked\n\n'''
s = s[:start] + new_strict + s[end:]

old = "wd=pd.DataFrame({'date':pd.DatetimeIndex(dates)}); wd['date']=pd.to_datetime(wd['date']).dt.normalize(); w=pd.merge_asof(wd.sort_values('date'),weights[['DATE']+symbols].sort_values('DATE'),left_on='date',right_on='DATE',direction='backward',allow_exact_matches=False).set_index('date'); W=w[symbols].fillna(0.0); R=r.reindex(pd.DatetimeIndex(dates))[symbols]"
new = "wsrc=weights[['DATE']+symbols].rename(columns={'DATE':'date'}); w=strict_asof(dates,wsrc).drop(columns=['_date'],errors='ignore'); W=w[symbols].fillna(0.0); R=r.reindex(pd.DatetimeIndex(dates))[symbols]"
if old not in s:
    raise SystemExit('expected constituent merge block not found')
s = s.replace(old, new, 1)

# Ensure the patched engine explicitly advertises the robust implementation for inspection.
if 'def strict_asof(left_dates, right, date_col=' not in s or 'np.searchsorted(right_ns, left_ns, side=' not in s:
    raise SystemExit('strict_asof patch not present')

ENGINE.write_text(s)
print('Patched', ENGINE)
