from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / 'v6.2/config.yaml').read_text())
OUT = ROOT / 'v6.2/research'
OUT.mkdir(parents=True, exist_ok=True)
SEED = int(CFG['seed'])
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)

ALIASES = {
    'INFOSYSTCH': 'INFY', 'HEROHONDA': 'HEROMOTOCO', 'BAJAJ-AUTO': 'BAJAJAUTO',
    'ZOMATO': 'ETERNAL', 'MINDTREE': 'LTIM', 'TATAMTRDVR': 'TATAMOTORS'
}

REGIMES = ['BEARISH', 'RANGE_BOUND', 'BULLISH']
Q = np.array([0.10, 0.50, 0.90], dtype=np.float32)


def regime(x: float) -> str:
    th = float(CFG['regime_threshold'])
    if x >= th:
        return 'BULLISH'
    if x <= -th:
        return 'BEARISH'
    return 'RANGE_BOUND'


def _clean_dates(df: pd.DataFrame, col: str = 'date') -> pd.DataFrame:
    z = df.copy()
    z[col] = pd.to_datetime(z[col], errors='coerce').dt.normalize()
    return z.dropna(subset=[col]).sort_values(col)


def load_inputs():
    options = pd.read_parquet(ROOT / 'data/cache/nifty_options_long.parquet')
    futures = pd.read_parquet(ROOT / 'data/cache/nifty_futures_long.parquet')
    context = pd.read_parquet(ROOT / 'data/cache/v5_market_context.parquet')
    news = pd.read_csv(ROOT / 'data/cache/v5_news_daily.csv')
    index = pd.read_parquet(ROOT / 'data/cache/v62_nifty50_index.parquet')
    prices = pd.read_parquet(ROOT / 'data/cache/v62_constituent_prices.parquet')
    weights = pd.read_csv(ROOT / 'data/cache/v62_constituent_weights.csv')
    membership = pd.read_csv(ROOT / 'data/cache/v62_nifty50_membership.csv')
    flows = pd.read_parquet(ROOT / 'data/cache/v62_fii_dii.parquet')

    index = _clean_dates(index)
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in index.columns:
            index[c] = pd.to_numeric(index[c], errors='coerce')
    index = index.dropna(subset=['close']).drop_duplicates('date', keep='last')

    options['date'] = pd.to_datetime(options['date'], errors='coerce').dt.normalize()
    options['expiry'] = pd.to_datetime(options['expiry'], errors='coerce').dt.normalize()
    options['option_type'] = options['option_type'].astype(str).str.upper()
    for c in ['strike', 'open_interest', 'volume', 'close']:
        options[c] = pd.to_numeric(options[c], errors='coerce').fillna(0.0)
    options = options.dropna(subset=['date', 'expiry'])

    futures['date'] = pd.to_datetime(futures['date'], errors='coerce').dt.normalize()
    futures['expiry'] = pd.to_datetime(futures['expiry'], errors='coerce').dt.normalize()
    for c in ['close', 'open_interest', 'volume']:
        futures[c] = pd.to_numeric(futures[c], errors='coerce').fillna(0.0)
    futures = futures.dropna(subset=['date', 'expiry'])

    context = _clean_dates(context)
    context['series'] = context['series'].astype(str).str.strip()
    context['close'] = pd.to_numeric(context['close'], errors='coerce')

    news.columns = [str(c).strip().lower() for c in news.columns]
    news = _clean_dates(news)
    for c in news.columns:
        if c != 'date':
            news[c] = pd.to_numeric(news[c], errors='coerce')

    prices.columns = [str(c).strip().lower() for c in prices.columns]
    prices['symbol'] = prices['symbol'].astype(str).str.upper().replace(ALIASES)
    prices['date'] = pd.to_datetime(prices['date'], errors='coerce').dt.normalize()
    prices['close'] = pd.to_numeric(prices['close'], errors='coerce')
    prices = prices.dropna(subset=['symbol', 'date', 'close']).drop_duplicates(['symbol', 'date'], keep='last')

    weights.columns = [str(c).strip().upper() for c in weights.columns]
    weights['DATE'] = pd.to_datetime(weights['DATE'], errors='coerce').dt.normalize()
    for c in weights.columns:
        if c != 'DATE':
            weights[c] = pd.to_numeric(weights[c], errors='coerce').fillna(0.0)
    weights = weights.dropna(subset=['DATE']).sort_values('DATE').drop_duplicates('DATE', keep='last')
    weights = weights.rename(columns={c: ALIASES.get(c, c) for c in weights.columns})

    membership.columns = [str(c).strip().lower() for c in membership.columns]
    membership['symbol'] = membership['symbol'].astype(str).str.upper().replace(ALIASES)
    membership['valid_from'] = pd.to_datetime(membership['valid_from'], errors='coerce').dt.normalize()
    membership['valid_to'] = pd.to_datetime(membership['valid_to'], errors='coerce').dt.normalize()
    membership = membership.dropna(subset=['symbol', 'valid_from'])

    flows['date'] = pd.to_datetime(flows['date'], errors='coerce').dt.normalize()
    for c in ['fii_net', 'dii_net']:
        if c in flows.columns:
            flows[c] = pd.to_numeric(flows[c], errors='coerce')
    flows = flows.dropna(subset=['date']).sort_values('date')
    return options, futures, context, news, index, prices, weights, membership, flows


def _expiry_maps(index_dates: pd.DatetimeIndex, options: pd.DataFrame):
    exps = np.array(sorted(pd.to_datetime(options['expiry']).dropna().unique()), dtype='datetime64[ns]')
    dates = pd.DatetimeIndex(index_dates)
    pos = np.searchsorted(exps, dates.values.astype('datetime64[ns]'), side='right')
    next_exp = [pd.Timestamp(exps[p]) if p < len(exps) else pd.NaT for p in pos]
    return pd.Series(next_exp, index=dates)


def _strict_asof(left_dates: pd.DatetimeIndex, right: pd.DataFrame, right_date_col: str = 'date') -> pd.DataFrame:
    l = pd.DataFrame({'date': pd.DatetimeIndex(left_dates)})
    r = right.copy().sort_values(right_date_col).rename(columns={right_date_col: '_date'})
    return pd.merge_asof(l.sort_values('date'), r, left_on='date', right_on='_date', direction='backward', allow_exact_matches=False).set_index('date')


def _index_features(index: pd.DataFrame) -> pd.DataFrame:
    x = index.set_index('date').sort_index().copy()
    c = x['close'].astype(float)
    r = c.pct_change(fill_method=None)
    out = pd.DataFrame(index=x.index)
    out['idx_ret1'] = r
    out['idx_ret5'] = c.pct_change(5, fill_method=None)
    out['idx_ret20'] = c.pct_change(20, fill_method=None)
    out['idx_vol5'] = r.rolling(5).std() * math.sqrt(252)
    out['idx_vol20'] = r.rolling(20).std() * math.sqrt(252)
    out['idx_ema20_gap'] = c / c.ewm(span=20, adjust=False).mean() - 1.0
    out['idx_range20'] = (c - c.shift(1).rolling(20).min()) / (c.shift(1).rolling(20).max() - c.shift(1).rolling(20).min() + 1e-9)
    if {'high', 'low'}.issubset(x.columns):
        out['idx_atr14'] = (x['high'] - x['low']).abs().rolling(14).mean() / c
    else:
        out['idx_atr14'] = np.nan
    out['idx_rsi14'] = _rsi(c) / 100.0
    return out.replace([np.inf, -np.inf], np.nan)


def _rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def _context_features(index_dates: pd.DatetimeIndex, context: pd.DataFrame, flows: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=index_dates)
    p = context.pivot_table(index='date', columns='series', values='close', aggfunc='last').sort_index()
    ctx = pd.DataFrame(index=p.index)
    aliases = {'usd_inr': 'USDINR', 'brent': 'BRENT'}
    for sym in ['india_vix', 'nifty_bank', 'nifty_midcap', 'spx', 'usd_inr', 'brent']:
        col = sym if sym in p.columns else aliases.get(sym, sym)
        if col in p.columns:
            ctx[f'{sym}_ret5'] = p[col].pct_change(5, fill_method=None)
    if len(ctx):
        out = out.join(_strict_asof(index_dates, ctx.reset_index(), 'date').drop(columns=['_date'], errors='ignore'))

    if {'fii_net', 'dii_net'}.issubset(flows.columns) and len(flows):
        f = flows.groupby('date', as_index=True)[['fii_net', 'dii_net']].sum().sort_index()
        ff = pd.DataFrame(index=f.index)
        for col in ['fii_net', 'dii_net']:
            s = f[col].astype(float)
            roll = s.rolling(60, min_periods=10)
            ff[f'{col}_z'] = (s - roll.mean()) / (roll.std() + 1e-9)
            ff[f'{col}_5z'] = s.rolling(5).sum() / (s.rolling(60, min_periods=10).std() + 1e-9)
            ff[f'{col}_20z'] = s.rolling(20).sum() / (s.rolling(60, min_periods=10).std() + 1e-9)
        tmp = _strict_asof(index_dates, ff.reset_index(), 'date').drop(columns=['_date'], errors='ignore')
        out = out.join(tmp)

    if len(news):
        pos = [c for c in news.columns if 'pos' in c]
        neg = [c for c in news.columns if 'neg' in c]
        cnt = [c for c in news.columns if 'count' in c]
        if pos or neg or cnt:
            n = news.set_index('date').sort_index()
            ps = n[pos].sum(axis=1) if pos else pd.Series(0.0, index=n.index)
            ns = n[neg].sum(axis=1) if neg else pd.Series(0.0, index=n.index)
            cc = n[cnt].sum(axis=1) if cnt else pd.Series(0.0, index=n.index)
            nd = pd.DataFrame(index=n.index)
            nd['news_sentiment_5'] = (ps.rolling(5).sum() - ns.rolling(5).sum()) / cc.rolling(5).sum().clip(lower=1)
            nd['news_volume_5'] = np.log1p(cc.rolling(5).sum().clip(lower=0))
            nd['news_neg_share_5'] = ns.rolling(5).sum() / cc.rolling(5).sum().clip(lower=1)
            tmp = _strict_asof(index_dates, nd.reset_index(), 'date').drop(columns=['_date'], errors='ignore')
            out = out.join(tmp)
    return out.replace([np.inf, -np.inf], np.nan)


def _option_features(index: pd.DataFrame, options: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DatetimeIndex(index['date'])
    spot_map = index.set_index('date')['close'].astype(float)
    exp_map = _expiry_maps(dates, options)
    base = pd.DataFrame({'date': dates, 'next_expiry': exp_map.values})
    o = options.merge(base, on='date', how='inner')
    o = o[o['expiry'].eq(o['next_expiry'])].copy()
    if o.empty:
        return pd.DataFrame(index=dates)
    o['spot'] = o['date'].map(spot_map)
    o['m_bin'] = np.rint((o['strike'] / o['spot'] - 1.0) * 100).clip(-5, 5).astype('Int64')
    o = o.dropna(subset=['m_bin'])
    rows = pd.DataFrame(index=dates)
    for typ in ['CE', 'PE']:
        for stat in ['open_interest', 'volume']:
            g = o[o['option_type'].eq(typ)].groupby(['date', 'm_bin'])[stat].sum().unstack(fill_value=0)
            g = g.reindex(dates).fillna(0.0)
            den = g.sum(axis=1).replace(0, 1.0)
            g = g.div(den, axis=0)
            for b in range(-5, 6):
                rows[f'opt_{typ.lower()}_{stat[:2]}_{b:+d}'] = g[b] if b in g.columns else 0.0
    ce = o[o['option_type'].eq('CE')].groupby('date').open_interest.sum()
    pe = o[o['option_type'].eq('PE')].groupby('date').open_interest.sum()
    cev = o[o['option_type'].eq('CE')].groupby('date').volume.sum()
    pev = o[o['option_type'].eq('PE')].groupby('date').volume.sum()
    rows['opt_pcr_oi'] = pe / ce.replace(0, np.nan)
    rows['opt_pcr_volume'] = pev / cev.replace(0, np.nan)
    atm = o[o['m_bin'].eq(0)].pivot_table(index='date', columns='option_type', values='close', aggfunc='mean')
    atm['spot'] = atm.index.map(spot_map)
    rows['opt_atm_straddle_pct'] = (atm.get('CE', pd.Series(index=atm.index, dtype=float)) + atm.get('PE', pd.Series(index=atm.index, dtype=float))) / atm['spot']
    otm = o[o['m_bin'].isin([-3, 3])].pivot_table(index='date', columns=['m_bin', 'option_type'], values='close', aggfunc='mean')
    put3 = otm.get((-3, 'PE'), pd.Series(index=otm.index, dtype=float))
    call3 = otm.get((3, 'CE'), pd.Series(index=otm.index, dtype=float))
    rows['opt_skew'] = (put3 - call3) / (put3 + call3).replace(0, np.nan)
    return rows.reindex(dates).replace([np.inf, -np.inf], np.nan)


def _futures_features(index: pd.DataFrame, futures: pd.DataFrame, options: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DatetimeIndex(index['date'])
    spot = index.set_index('date')['close']
    exp_map = _expiry_maps(dates, options)
    f = futures.copy()
    f['next_expiry'] = f['date'].map(exp_map)
    f = f[f['expiry'].eq(f['next_expiry'])].copy()
    if f.empty:
        return pd.DataFrame(index=dates)
    f = f.groupby('date').agg(close=('close', 'last'), open_interest=('open_interest', 'last'), volume=('volume', 'last'))
    out = pd.DataFrame(index=dates)
    out['fut_basis'] = f['close'] / spot - 1.0
    out['fut_oi'] = np.log1p(f['open_interest'].clip(lower=0))
    out['fut_volume'] = np.log1p(f['volume'].clip(lower=0))
    return out


def _stock_features(index_dates: pd.DatetimeIndex, prices: pd.DataFrame, weights: pd.DataFrame, membership: pd.DataFrame):
    wide = prices.pivot(index='date', columns='symbol', values='close').sort_index()
    symbols = sorted(wide.columns.astype(str).tolist())
    wide = wide.reindex(columns=symbols)
    r1 = wide.pct_change(fill_method=None)
    r5 = wide.pct_change(5, fill_method=None)
    r20 = wide.pct_change(20, fill_method=None)
    vol20 = wide.pct_change(fill_method=None).rolling(20).std() * math.sqrt(252)
    wd = weights.copy().sort_values('DATE')
    wdates = wd['DATE'].to_numpy(dtype='datetime64[ns]')
    wmat = wd.reindex(columns=symbols, fill_value=0.0).drop(columns=['DATE'], errors='ignore').to_numpy(dtype=np.float32) if len(wd) else np.zeros((0, len(symbols)), dtype=np.float32)
    tensor = np.zeros((len(index_dates), len(symbols), 6), dtype=np.float32)
    for i, d in enumerate(index_dates):
        active = membership[(membership['valid_from'] <= d) & (membership['valid_to'].isna() | (membership['valid_to'] > d))]['symbol'].drop_duplicates().tolist()
        active_mask = np.array([s in active for s in symbols], dtype=np.float32)
        if len(wd):
            pos = np.searchsorted(wdates, np.datetime64(d), side='right') - 1
            gap = (d - pd.Timestamp(wdates[pos])).days if pos >= 0 else 10_000
            w = wmat[pos].astype(np.float32) if pos >= 0 else np.zeros(len(symbols), dtype=np.float32)
            if gap > int(CFG['constituents']['weights_max_staleness_days']):
                w = np.zeros(len(symbols), dtype=np.float32)
        else:
            w = np.zeros(len(symbols), dtype=np.float32)
        w = np.where(active_mask > 0, np.maximum(w, 0), 0)
        if w.sum() <= 0 and active_mask.sum() > 0:
            w = active_mask / active_mask.sum()
        else:
            w = w / max(w.sum(), 1e-9)
        for j, s in enumerate(symbols):
            tensor[i, j, 0] = float(r1.loc[d, s]) if d in r1.index and np.isfinite(r1.loc[d, s]) else 0.0
            tensor[i, j, 1] = float(r5.loc[d, s]) if d in r5.index and np.isfinite(r5.loc[d, s]) else 0.0
            tensor[i, j, 2] = float(r20.loc[d, s]) if d in r20.index and np.isfinite(r20.loc[d, s]) else 0.0
            tensor[i, j, 3] = float(vol20.loc[d, s]) if d in vol20.index and np.isfinite(vol20.loc[d, s]) else 0.0
            tensor[i, j, 4] = float(w[j])
            tensor[i, j, 5] = float(active_mask[j])
    return tensor, symbols


def build_daily_modalities(inp):
    options, futures, context, news, index, prices, weights, membership, flows = inp
    dates = pd.DatetimeIndex(index['date'].sort_values().unique())
    idxf = _index_features(index).reindex(dates)
    ctx = _context_features(dates, context, flows, news).reindex(dates)
    opt = _option_features(index, options).reindex(dates)
    fut = _futures_features(index, futures, options).reindex(dates)
    global_df = idxf.join(ctx, how='left').join(opt, how='left').join(fut, how='left')
    global_df = global_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    stock_tensor, symbols = _stock_features(dates, prices, weights, membership)
    return dates, global_df, stock_tensor, symbols, options


def build_samples(inp):
    dates, global_df, stock_tensor, symbols, options = build_daily_modalities(inp)
    index = inp[4].set_index('date').sort_index()
    exps = pd.DatetimeIndex(sorted(pd.to_datetime(options['expiry']).dropna().unique()))
    seq = int(CFG['sequence_length'])
    stride = int(CFG.get('sample_stride', 2))
    canonical = {}
    prev = None
    for e in exps:
        if prev is not None:
            z = dates[(dates > prev) & (dates < e)]
            if len(z):
                canonical[pd.Timestamp(e)] = pd.Timestamp(z[0])
        prev = pd.Timestamp(e)
    rows = []
    xs_g = []
    xs_s = []
    for i in range(seq - 1, len(dates), stride):
        d = pd.Timestamp(dates[i])
        pos = np.searchsorted(exps.values, np.datetime64(d), side='right')
        if pos >= len(exps):
            continue
        expiry = pd.Timestamp(exps[pos])
        target_dates = dates[dates <= expiry]
        if not len(target_dates):
            continue
        td = pd.Timestamp(target_dates[-1])
        if td <= d or td not in index.index:
            continue
        spot = float(index.loc[d, 'close'])
        tgt = index.loc[td]
        high = float(tgt['high']) if 'high' in tgt.index and np.isfinite(tgt['high']) else float(tgt['close'])
        low = float(tgt['low']) if 'low' in tgt.index and np.isfinite(tgt['low']) else float(tgt['close'])
        y = [float(tgt['close']) / spot - 1.0, high / spot - 1.0, low / spot - 1.0]
        sl = slice(i - seq + 1, i + 1)
        xs_g.append(global_df.iloc[sl].to_numpy(dtype=np.float32))
        xs_s.append(stock_tensor[sl].astype(np.float32))
        rows.append({
            'signal_date': d, 'expiry': expiry, 'target_date': td, 'spot': spot,
            'target_return': y[0], 'target_high_return': y[1], 'target_low_return': y[2],
            'canonical': bool(canonical.get(expiry) == d)
        })
    meta = pd.DataFrame(rows).sort_values('signal_date').reset_index(drop=True)
    return {
        'global': np.stack(xs_g).astype(np.float32),
        'stock': np.stack(xs_s).astype(np.float32),
        'meta': meta,
        'global_columns': list(global_df.columns),
        'stock_symbols': symbols,
        'dates': dates,
        'latest_global': global_df,
        'latest_stock': stock_tensor,
        'index': index,
        'options': options,
    }


class DeepExpiryTransformer(nn.Module):
    def __init__(self, global_dim: int, stock_dim: int, n_stock: int):
        super().__init__()
        c = CFG['v7']
        sd = int(c['stock_embed_dim'])
        d = int(c['d_model'])
        heads = int(c['transformer_heads'])
        layers = int(c['transformer_layers'])
        drop = float(c['dropout'])
        self.stock_mlp = nn.Sequential(nn.Linear(stock_dim, sd), nn.GELU(), nn.Dropout(drop), nn.Linear(sd, sd))
        self.stock_score = nn.Linear(sd, 1)
        self.global_mlp = nn.Sequential(nn.Linear(global_dim, d), nn.GELU(), nn.Dropout(drop))
        self.fusion = nn.Sequential(nn.Linear(sd + d, d), nn.GELU(), nn.Dropout(drop))
        self.pos = nn.Parameter(torch.zeros(1, int(CFG['sequence_length']), d))
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=4 * d, dropout=drop, batch_first=True, norm_first=True, activation='gelu')
        self.temporal = nn.TransformerEncoder(enc, num_layers=layers)
        self.norm = nn.LayerNorm(d)
        self.quantile_head = nn.Linear(d, 9)
        self.regime_head = nn.Linear(d, 3)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, global_x: torch.Tensor, stock_x: torch.Tensor):
        # stock_x: [B,T,N,features], final feature is the PIT active mask.
        sh = self.stock_mlp(stock_x)
        scores = self.stock_score(sh).squeeze(-1)
        mask = stock_x[..., -1] > 0.5
        scores = scores.masked_fill(~mask, -1e4)
        attn = torch.softmax(scores, dim=-1)
        pooled = (sh * attn.unsqueeze(-1)).sum(dim=2)
        gh = self.global_mlp(global_x)
        h = self.fusion(torch.cat([pooled, gh], dim=-1))
        h = h + self.pos[:, :h.shape[1]]
        h = self.temporal(h)
        h = self.norm(h[:, -1])
        return self.quantile_head(h), self.regime_head(h)


def _pinball(pred, target, quantiles):
    loss = 0.0
    for k, q in enumerate(quantiles):
        e = target - pred[:, k]
        loss = loss + torch.maximum(torch.tensor(q - 1.0, device=pred.device) * e, torch.tensor(q, device=pred.device) * e).mean()
    return loss / len(quantiles)


def _normalizer(train_g, train_s):
    gm = train_g.reshape(-1, train_g.shape[-1]).mean(0)
    gs = train_g.reshape(-1, train_g.shape[-1]).std(0)
    gs = np.where(gs < 1e-6, 1.0, gs)
    sm = train_s.reshape(-1, train_s.shape[-1]).mean(0)
    ss = train_s.reshape(-1, train_s.shape[-1]).std(0)
    ss = np.where(ss < 1e-6, 1.0, ss)
    return gm.astype(np.float32), gs.astype(np.float32), sm.astype(np.float32), ss.astype(np.float32)


def _apply_norm(g, s, norm):
    gm, gs, sm, ss = norm
    g = np.clip((g - gm) / gs, -8, 8).astype(np.float32)
    s = np.clip((s - sm) / ss, -8, 8).astype(np.float32)
    # Preserve PIT activity semantics after scaling.
    s[..., -1] = (s[..., -1] > -0.5).astype(np.float32)
    return g, s


def _fit_model(train_g, train_s, y, epochs=None):
    epochs = int(epochs or CFG['v7']['epochs'])
    norm = _normalizer(train_g, train_s)
    g, s = _apply_norm(train_g, train_s, norm)
    n = len(y)
    split = max(1, int(n * 0.9))
    tr = np.arange(split)
    va = np.arange(split, n) if split < n else tr
    model = DeepExpiryTransformer(g.shape[-1], s.shape[-1], s.shape[-2]).float()
    opt = torch.optim.AdamW(model.parameters(), lr=float(CFG['v7']['learning_rate']), weight_decay=float(CFG['v7']['weight_decay']))
    loss_fn = nn.CrossEntropyLoss()
    ds = TensorDataset(torch.from_numpy(g[tr]), torch.from_numpy(s[tr]), torch.from_numpy(y[tr].astype(np.float32)))
    dl = DataLoader(ds, batch_size=int(CFG['v7']['batch_size']), shuffle=False)
    best = float('inf'); best_state = None; bad = 0
    for _ in range(epochs):
        model.train()
        for xb, sb, yb in dl:
            opt.zero_grad(set_to_none=True)
            qhat, logit = model(xb, sb)
            close_q = qhat[:, 0:3]
            high_q = qhat[:, 3:6]
            low_q = qhat[:, 6:9]
            loss = _pinball(close_q, yb[:, 0], Q) + _pinball(high_q, yb[:, 1], Q) + _pinball(low_q, yb[:, 2], Q)
            cls = np.where(yb[:, 0].numpy() > float(CFG['regime_threshold']), 2, np.where(yb[:, 0].numpy() < -float(CFG['regime_threshold']), 0, 1))
            cls_t = torch.from_numpy(cls).long()
            loss = loss + 0.20 * loss_fn(logit, cls_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(CFG['v7']['gradient_clip']))
            opt.step()
        model.eval()
        with torch.no_grad():
            vg, vs, vy = torch.from_numpy(g[va]), torch.from_numpy(s[va]), torch.from_numpy(y[va].astype(np.float32))
            qv, lv = model(vg, vs)
            val = float(_pinball(qv[:, :3], vy[:, 0], Q) + _pinball(qv[:, 3:6], vy[:, 1], Q) + _pinball(qv[:, 6:9], vy[:, 2], Q))
        if val < best - 1e-5:
            best = val; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= int(CFG['v7']['patience']):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, norm


def _predict(model, norm, g, s):
    g, s = _apply_norm(g, s, norm)
    model.eval()
    with torch.no_grad():
        qhat, logit = model(torch.from_numpy(g), torch.from_numpy(s))
        q = qhat.numpy(); p = torch.softmax(logit, dim=1).numpy()
    return q, p


def _metrics(pred_q, pred_p, meta: pd.DataFrame):
    actual = meta['target_return'].to_numpy(float)
    pred = pred_q[:, 1]
    mae = float(np.mean(np.abs(pred - actual)))
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    direction = float(np.mean((pred >= 0) == (actual >= 0)))
    actual_reg = np.where(actual > float(CFG['regime_threshold']), 2, np.where(actual < -float(CFG['regime_threshold']), 0, 1))
    model_reg = np.argmax(pred_p, axis=1)
    regacc = float(np.mean(actual_reg == model_reg))
    corr = float(np.corrcoef(pred, actual)[0, 1]) if len(pred) > 1 and np.std(pred) > 1e-9 and np.std(actual) > 1e-9 else 0.0
    return {'mae': mae, 'rmse': rmse, 'direction_accuracy': direction, 'regime_accuracy': regacc, 'correlation': corr, 'samples': int(len(meta))}


def _canonical(samples):
    return samples['meta']['canonical'].to_numpy(bool)


def _eligible(meta, expiries):
    return meta['expiry'].isin(set(pd.to_datetime(expiries)))


def run_wfo(samples):
    meta = samples['meta']; exps = pd.DatetimeIndex(sorted(meta['expiry'].unique()))
    nwin = int(CFG['v7']['wfo_windows']); train_n = int(CFG['v7']['wfo_train_expiries']); test_n = int(CFG['v7']['wfo_test_expiries'])
    needed = train_n + test_n
    if len(exps) < needed:
        raise RuntimeError(f'Not enough expiry groups for WFO: {len(exps)}')
    starts = np.linspace(0, len(exps) - needed, nwin, dtype=int)
    rows=[]
    for w, start in enumerate(starts, 1):
        train_exp = exps[start:start+train_n]
        test_exp = exps[start+train_n:start+needed]
        first_test = pd.Timestamp(test_exp[0])
        train_exp = train_exp[train_exp <= first_test - pd.Timedelta(days=int(CFG['purge_days']))]
        tr = np.where(_eligible(meta, train_exp))[0]
        te = np.where(_eligible(meta, test_exp) & _canonical(samples))[0]
        if len(tr) < int(CFG['min_samples']) or len(te) < 2:
            continue
        model, norm = _fit_model(samples['global'][tr], samples['stock'][tr], meta.loc[tr, ['target_return','target_high_return','target_low_return']].to_numpy())
        q, p = _predict(model, norm, samples['global'][te], samples['stock'][te])
        m = _metrics(q, p, meta.loc[te].reset_index(drop=True)); m.update({'window': w, 'model': 'deep_transformer'})
        rows.append(m)
        nq = np.zeros((len(te), 9), dtype=np.float32); npreds = np.full((len(te), 3), 1/3, dtype=np.float32)
        nm = _metrics(nq, npreds, meta.loc[te].reset_index(drop=True)); nm.update({'window': w, 'model': 'naive'})
        rows.append(nm)
    return pd.DataFrame(rows)


def run_split(samples, mode='70_30'):
    meta = samples['meta']; exps = pd.DatetimeIndex(sorted(meta['expiry'].unique()))
    if mode == '70_30':
        cut = max(1, int(len(exps) * 0.70)); train_exp, test_exp = exps[:cut], exps[cut:]
    else:
        cutoff = pd.Timestamp(CFG['new_oos_start']); train_exp, test_exp = exps[exps < cutoff], exps[exps >= cutoff]
    first_test = pd.Timestamp(test_exp[0])
    train_exp = train_exp[train_exp <= first_test - pd.Timedelta(days=int(CFG['purge_days']))]
    tr = np.where(_eligible(meta, train_exp))[0]
    te = np.where(_eligible(meta, test_exp) & _canonical(samples))[0]
    if len(tr) < int(CFG['min_samples']) or len(te) == 0:
        raise RuntimeError(f'{mode} split insufficient: train={len(tr)} test={len(te)}')
    model, norm = _fit_model(samples['global'][tr], samples['stock'][tr], meta.loc[tr, ['target_return','target_high_return','target_low_return']].to_numpy())
    q, p = _predict(model, norm, samples['global'][te], samples['stock'][te])
    m = _metrics(q, p, meta.loc[te].reset_index(drop=True)); m.update({'model': 'deep_transformer'})
    nq = np.zeros((len(te), 9), dtype=np.float32); npreds = np.full((len(te), 3), 1/3, dtype=np.float32)
    nm = _metrics(nq, npreds, meta.loc[te].reset_index(drop=True)); nm.update({'model': 'naive'})
    return pd.DataFrame([m, nm]), (model, norm, tr, te, q, p)


def _latest_forecast(samples, production_model, norm):
    dates = samples['dates']
    gdf = samples['latest_global']
    st = samples['latest_stock']
    latest_date = pd.Timestamp(dates[-1])
    index = samples['index']
    options = samples['options']
    exps = pd.DatetimeIndex(sorted(options['expiry'].unique()))
    pos = np.searchsorted(exps.values, np.datetime64(latest_date), side='right')
    expiry = pd.Timestamp(exps[pos])
    seq = int(CFG['sequence_length'])
    g = gdf.iloc[-seq:].to_numpy(dtype=np.float32)[None, ...]
    s = st[-seq:].astype(np.float32)[None, ...]
    q, p = _predict(production_model, norm, g, s)
    q = q[0]; p = p[0]
    close_q = q[0:3].copy(); high_q = q[3:6].copy(); low_q = q[6:9].copy()
    pred_return = float(close_q[1])
    low_ret = float(min(low_q[0], close_q[1]))
    high_ret = float(max(high_q[2], close_q[1]))
    spot = float(index.loc[latest_date, 'close'])
    probs = {'BEARISH': float(p[0]), 'RANGE_BOUND': float(p[1]), 'BULLISH': float(p[2])}
    model_regime = REGIMES[int(np.argmax(p))]
    return {
        'available': True,
        'version': 'V7',
        'signal_date': str(latest_date.date()),
        'latest_feature_date': str(latest_date.date()),
        'next_expiry': str(expiry.date()),
        'spot': spot,
        'production_model': 'deep_transformer',
        'predicted_return': pred_return,
        'predicted_close': spot * (1 + pred_return),
        'predicted_low': spot * (1 + low_ret),
        'predicted_high': spot * (1 + high_ret),
        'regime': model_regime,
        'regime_probabilities': probs,
        'prediction_interval': float(CFG['prediction_interval']),
        'close_quantiles_return': {'q10': float(close_q[0]), 'q50': float(close_q[1]), 'q90': float(close_q[2])},
        'low_quantiles_return': {'q10': float(low_q[0]), 'q50': float(low_q[1]), 'q90': float(low_q[2])},
        'high_quantiles_return': {'q10': float(high_q[0]), 'q50': float(high_q[1]), 'q90': float(high_q[2])},
        'forecast_feature_date_verified': True,
        'forecast_type': 'fresh_current_cutoff_deep_multimodal',
    }


def paper_strategy(forecast: dict, options: pd.DataFrame):
    spot = float(forecast['spot']); expiry = pd.Timestamp(forecast['next_expiry']); d = pd.Timestamp(forecast['signal_date'])
    o = options[(options['date'] == d) & (options['expiry'] == expiry)].copy()
    lot = 65
    if o.empty:
        return {'action': 'NO_TRADE', 'lot_size': lot, 'reason': 'No current expiry option quote'}
    def q(typ, strike):
        z = o[o['option_type'].eq(typ)].copy()
        if z.empty: return np.nan, float(strike)
        z['dist'] = abs(z['strike'] - strike)
        r = z.sort_values('dist').iloc[0]
        return float(r['close']), float(r['strike'])
    width = float(CFG['options']['spread_width_points'])
    reg = forecast['regime']
    if reg == 'BEARISH':
        lp, k1 = q('PE', spot); sp, k2 = q('PE', spot - width)
        debit = max(lp - sp, 0.0); max_loss = debit * lot; max_profit = max(width - debit, 0.0) * lot
        return {'action': 'BEAR_PUT_SPREAD', 'lot_size': lot, 'legs': [['BUY','PE',k1,lp],['SELL','PE',k2,sp]], 'entry_debit_per_unit': debit, 'estimated_max_loss': max_loss, 'estimated_max_profit': max_profit, 'capital_requirement': max_loss}
    if reg == 'BULLISH':
        lc, k1 = q('CE', spot); sc, k2 = q('CE', spot + width)
        debit = max(lc - sc, 0.0); max_loss = debit * lot; max_profit = max(width - debit, 0.0) * lot
        return {'action': 'BULL_CALL_SPREAD', 'lot_size': lot, 'legs': [['BUY','CE',k1,lc],['SELL','CE',k2,sc]], 'entry_debit_per_unit': debit, 'estimated_max_loss': max_loss, 'estimated_max_profit': max_profit, 'capital_requirement': max_loss}
    lowk = min(spot - width, float(forecast['predicted_low'])); highk = max(spot + width, float(forecast['predicted_high']))
    sp, ks = q('PE', lowk); sc, kc = q('CE', highk); lp, kl = q('PE', ks - width); lc, ku = q('CE', kc + width)
    credit = max(sp + sc - lp - lc, 0.0); max_loss = max(width - credit, 0.0) * lot
    return {'action': 'IRON_CONDOR', 'lot_size': lot, 'legs': [['BUY','PE',kl,lp],['SELL','PE',ks,sp],['SELL','CE',kc,sc],['BUY','CE',ku,lc]], 'entry_credit_per_unit': credit, 'estimated_max_loss': max_loss, 'capital_requirement': max_loss}


def main():
    inp = load_inputs()
    samples = build_samples(inp)
    meta = samples['meta']
    hist_end = pd.Timestamp(CFG['history_end'])
    known = meta[pd.to_datetime(meta['target_date']) <= hist_end].copy().reset_index(drop=True)
    # Reindex arrays after removing any sample whose expiry target is still in the future.
    keep = np.where(pd.to_datetime(meta['target_date']) <= hist_end)[0]
    for k in ['global', 'stock']:
        samples[k] = samples[k][keep]
    samples['meta'] = meta.loc[keep].reset_index(drop=True)
    meta = samples['meta']

    wfo = run_wfo(samples)
    split70, split_obj = run_split(samples, '70_30')
    oos, oos_obj = run_split(samples, 'new_oos')

    # Production training uses every fully-labelled expiry available at the cutoff.
    all_y = meta[['target_return', 'target_high_return', 'target_low_return']].to_numpy(dtype=np.float32)
    prod_model, prod_norm = _fit_model(samples['global'], samples['stock'], all_y)
    latest = _latest_forecast(samples, prod_model, prod_norm)
    trade = paper_strategy(latest, samples['options'])

    wfo.to_csv(OUT / 'walk_forward_model_metrics.csv', index=False)
    split70.to_csv(OUT / 'split_70_30_model_metrics.csv', index=False)
    oos.to_csv(OUT / 'new_oos_model_metrics.csv', index=False)

    # Canonical prediction tables make leakage control and expiry-level inspection explicit.
    def save_predictions(obj, name):
        _, _, _, te, q, p = obj
        z = samples['meta'].loc[te, ['signal_date','expiry','target_date','spot','target_return']].reset_index(drop=True).copy()
        z['pred_close_q10'] = q[:,0]; z['pred_close_q50'] = q[:,1]; z['pred_close_q90'] = q[:,2]
        z['pred_high_q90'] = q[:,5]; z['pred_low_q10'] = q[:,6]
        z['regime_prob_bear'] = p[:,0]; z['regime_prob_range'] = p[:,1]; z['regime_prob_bull'] = p[:,2]
        z.to_csv(OUT / name, index=False)
    save_predictions(split_obj, 'split_70_30_predictions.csv')
    save_predictions(oos_obj, 'new_oos_predictions.csv')

    # WFO predictions aggregated only at canonical signal dates.
    wp = []
    exps = sorted(samples['meta']['expiry'].unique())
    for _, row in wfo[wfo['model'].eq('deep_transformer')].iterrows():
        pass
    # Write a concise research-status file; individual WFO metric rows remain the source of truth.
    news_manifest = json.loads((ROOT / 'data/cache/v62_news_manifest.json').read_text()) if (ROOT / 'data/cache/v62_news_manifest.json').exists() else {}
    flow_manifest = json.loads((ROOT / 'data/cache/v62_fii_dii_manifest.json').read_text()) if (ROOT / 'data/cache/v62_fii_dii_manifest.json').exists() else {}
    summary = {
        'version': 'V7',
        'sample_count': int(len(meta)),
        'feature_count_global': int(len(samples['global_columns'])),
        'constituent_symbols': int(len(samples['stock_symbols'])),
        'sequence_length': int(CFG['sequence_length']),
        'sample_stride': int(CFG.get('sample_stride',2)),
        'selected_production_model': 'deep_transformer',
        'history_end': str(hist_end.date()),
        'new_oos_start': CFG['new_oos_start'],
        'wfo': wfo.to_dict('records'),
        'split_70_30': split70.to_dict('records'),
        'new_oos': oos.to_dict('records'),
        'latest_forecast': latest,
        'latest_paper_trade': trade,
        'news_manifest': news_manifest,
        'fii_dii_manifest': flow_manifest,
        'research_gate': 'RESEARCH_ONLY_PENDING_STABLE_OOS',
    }
    json.dump(summary, open(OUT / 'summary.json', 'w'), indent=2)
    json.dump({'status':'V7_DEEP_MULTIMODAL_READY','history_end':str(hist_end.date()),'new_oos_start':CFG['new_oos_start'],'architecture':'stock-attention + multimodal temporal transformer','targets':['expiry_close_return','expiry_high_return','expiry_low_return','regime']}, open(OUT/'status.json','w'), indent=2)
    json.dump(latest, open(OUT / 'latest_forecast.json', 'w'), indent=2)
    pd.DataFrame([trade]).to_csv(OUT / 'latest_paper_trade.csv', index=False)
    print(json.dumps({'version':'V7','samples':len(meta),'global_features':len(samples['global_columns']),'constituents':len(samples['stock_symbols']),'latest_forecast':latest,'latest_paper_trade':trade,'new_oos':oos.to_dict('records')}, indent=2))


if __name__ == '__main__':
    main()
