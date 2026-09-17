from __future__ import annotations

import math

import numpy as np
import pandas as pd

from multimodal_engine import main as engine_main
import multimodal_engine as engine


def _safe_context_features(context: pd.DataFrame, news: pd.DataFrame, flows: pd.DataFrame, d: pd.Timestamp) -> dict:
    """Build strictly point-in-time context features without assuming prior observations exist."""
    d = pd.Timestamp(d)
    out: dict[str, float] = {}

    c = context.copy()
    if not c.empty:
        c['date'] = pd.to_datetime(c['date'], errors='coerce').dt.normalize()
        c = c[c['date'] < d]
        if not c.empty and {'date', 'series', 'close'}.issubset(c.columns):
            p = c.pivot_table(index='date', columns='series', values='close', aggfunc='last').sort_index()
            aliases = {
                'usd_inr': 'USDINR',
                'brent': 'BRENT',
                'nifty_bank': 'nifty_bank',
                'nifty_midcap': 'nifty_midcap',
            }
            for sym in ['india_vix', 'nifty_bank', 'nifty_midcap', 'spx', 'usd_inr', 'brent']:
                col = sym if sym in p.columns else aliases.get(sym, sym)
                if col in p.columns and len(p[col].dropna()) > 0:
                    s = pd.to_numeric(p[col], errors='coerce').dropna()
                    out[f'{sym}_ret5'] = float(s.pct_change(5).iloc[-1]) if len(s) >= 6 else np.nan

    f = flows.copy()
    if not f.empty and 'date' in f.columns:
        f['date'] = pd.to_datetime(f['date'], errors='coerce').dt.normalize()
        f = f[f['date'] < d].sort_values('date')
        # Historical FII/DII data may begin much later than the price history.
        # Missing prior flow observations remain NaN rather than aborting the sample.
        for col in ['fii_net', 'dii_net']:
            if col not in f.columns:
                continue
            s = pd.to_numeric(f[col], errors='coerce').dropna()
            if s.empty:
                continue
            out[f'{col}_1'] = float(s.iloc[-1])
            out[f'{col}_5'] = float(s.tail(5).sum())
            out[f'{col}_20'] = float(s.tail(20).sum())
            base = s.tail(120)
            out[f'{col}_z'] = float((s.iloc[-1] - base.mean()) / (base.std() + 1e-9)) if len(base) > 20 else np.nan

    n = news.copy()
    if not n.empty and 'date' in n.columns:
        n['date'] = pd.to_datetime(n['date'], errors='coerce').dt.normalize()
        n = n[n['date'] < d].sort_values('date')
        if not n.empty:
            neg = [c for c in n.columns if 'neg' in c and pd.api.types.is_numeric_dtype(n[c])]
            pos = [c for c in n.columns if 'pos' in c and pd.api.types.is_numeric_dtype(n[c])]
            cnt = [c for c in n.columns if 'count' in c and pd.api.types.is_numeric_dtype(n[c])]
            if pos or neg or cnt:
                ps = float(n.tail(5)[pos].sum().sum()) if pos else 0.0
                ns = float(n.tail(5)[neg].sum().sum()) if neg else 0.0
                cc = float(n.tail(5)[cnt].sum().sum()) if cnt else 0.0
                out['news_sentiment_5'] = (ps - ns) / max(cc, 1.0)
                out['news_volume_5'] = math.log1p(max(cc, 0.0))
                out['news_neg_share_5'] = ns / max(cc, 1.0)

    return out


# Replace only the fragile context helper; all model, WFO, OOS and trading logic remains in the main engine.
engine.context_features = _safe_context_features

if __name__ == '__main__':
    print('V6.2 safe runner: sparse-history context guard enabled')
    engine_main()
