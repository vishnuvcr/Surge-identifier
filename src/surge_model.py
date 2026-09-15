from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

FEATURES = [
    "ret1", "ret2", "ret3", "ret5", "ret10", "ret20",
    "gap", "range_pct", "body_pct", "close_pos", "atr_pct", "rsi14",
    "vol_z20", "turnover_z20", "relvol5", "relvol20", "dist_high20",
    "dist_high60", "dist_low20", "trend20", "trend60", "volatility20",
    "skew20", "market_ret1", "market_ret5", "market_breadth",
    "cross_ret_rank", "cross_vol_rank", "fut_oi_z20", "fut_oi_change_z20",
    "fut_volume_rel20", "pcr_dev20",
]


def _safe_div(a, b):
    return a / b.replace(0, np.nan)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"]).copy()
    for col, default in [("fut_oi", 0.0), ("fut_oi_change", 0.0), ("fut_volume", 0.0), ("pcr", 1.0)]:
        if col not in df.columns:
            df[col] = default
    g = df.groupby("symbol", group_keys=False)
    close = df["close"].astype(float)
    opn = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev = df["prev_close"].astype(float)
    rng = (high - low).clip(lower=0)

    for n in [1, 2, 3, 5, 10, 20]:
        df[f"ret{n}"] = g["close"].pct_change(n)
    df["gap"] = _safe_div(opn, prev) - 1
    df["range_pct"] = _safe_div(rng, prev)
    df["body_pct"] = _safe_div(close - opn, prev)
    df["close_pos"] = _safe_div(close - low, (high - low))

    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    df["atr_pct"] = _safe_div(tr.groupby(df["symbol"]).transform(lambda s: s.rolling(14, min_periods=8).mean()), close)

    delta = g["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.groupby(df["symbol"]).transform(lambda s: s.rolling(14, min_periods=8).mean())
    avg_down = down.groupby(df["symbol"]).transform(lambda s: s.rolling(14, min_periods=8).mean())
    rs = _safe_div(avg_up, avg_down)
    df["rsi14"] = 100 - (100 / (1 + rs))

    turnover = close * df["volume"].astype(float)
    df["turnover_cr"] = turnover / 1e7
    for col, out in [("volume", "vol_z20"), ("turnover_cr", "turnover_z20")]:
        mean = g[col].transform(lambda s: s.rolling(20, min_periods=10).mean())
        std = g[col].transform(lambda s: s.rolling(20, min_periods=10).std())
        df[out] = _safe_div(df[col] - mean, std)
    mean5 = g["volume"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    mean20 = g["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["relvol5"] = _safe_div(df["volume"], mean5)
    df["relvol20"] = _safe_div(df["volume"], mean20)

    hi20 = g["high"].transform(lambda s: s.rolling(20, min_periods=10).max())
    hi60 = g["high"].transform(lambda s: s.rolling(60, min_periods=20).max())
    lo20 = g["low"].transform(lambda s: s.rolling(20, min_periods=10).min())
    df["dist_high20"] = _safe_div(close, hi20) - 1
    df["dist_high60"] = _safe_div(close, hi60) - 1
    df["dist_low20"] = _safe_div(close, lo20) - 1
    df["trend20"] = _safe_div(close, g["close"].transform(lambda s: s.rolling(20, min_periods=10).mean())) - 1
    df["trend60"] = _safe_div(close, g["close"].transform(lambda s: s.rolling(60, min_periods=20).mean())) - 1
    df["volatility20"] = g["ret1"].transform(lambda s: s.rolling(20, min_periods=10).std())
    df["skew20"] = g["ret1"].transform(lambda s: s.rolling(20, min_periods=15).skew())

    for col, out in [("fut_oi", "fut_oi_z20"), ("fut_oi_change", "fut_oi_change_z20")]:
        mean = g[col].transform(lambda s: s.rolling(20, min_periods=5).mean())
        std = g[col].transform(lambda s: s.rolling(20, min_periods=5).std())
        df[out] = _safe_div(df[col] - mean, std)
    fvol_mean = g["fut_volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    df["fut_volume_rel20"] = _safe_div(df["fut_volume"], fvol_mean)
    pcr_mean = g["pcr"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    df["pcr_dev20"] = df["pcr"] - pcr_mean
    # Neutral values for stocks without an active derivatives contract.
    df["fut_oi_z20"] = df["fut_oi_z20"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["fut_oi_change_z20"] = df["fut_oi_change_z20"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["fut_volume_rel20"] = df["fut_volume_rel20"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    df["pcr_dev20"] = df["pcr_dev20"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    daily = df.groupby("date").agg(
        market_ret1=("ret1", "mean"),
        market_ret5=("ret5", "mean"),
        market_breadth=("ret1", lambda s: float((s > 0).mean())),
    )
    df = df.join(daily, on="date")
    df["cross_ret_rank"] = df.groupby("date")["ret1"].rank(pct=True)
    df["cross_vol_rank"] = df.groupby("date")["relvol20"].rank(pct=True)

    df["next_open"] = g["open"].shift(-1)
    df["next_high"] = g["high"].shift(-1)
    df["next_close"] = g["close"].shift(-1)
    df["next_intraday_max_return"] = _safe_div(df["next_high"], df["next_open"]) - 1
    df["target"] = (df["next_intraday_max_return"] >= 0.03).astype(float)
    return df.replace([np.inf, -np.inf], np.nan)


class SurgeNet(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 112), nn.LayerNorm(112), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(112, 56), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(56, 20), nn.GELU(), nn.Linear(20, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


@dataclass
class ModelBundle:
    model: SurgeNet
    scaler: StandardScaler
    threshold: float
    validation_precision: float
    validation_recall: float
    validation_specificity: float


def _metrics(y_true: np.ndarray, p: np.ndarray, threshold: float):
    pred = p >= threshold
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return precision, recall, specificity


def train_walk_forward(df: pd.DataFrame, validation_days: int = 60, seed: int = 42) -> ModelBundle:
    torch.manual_seed(seed)
    np.random.seed(seed)
    work = df.dropna(subset=FEATURES + ["target"]).copy()
    dates = np.array(sorted(work["date"].unique()))
    if len(dates) <= validation_days + 20:
        raise ValueError("Not enough trading history for walk-forward training")
    cut = dates[-validation_days]
    train = work[work["date"] < cut]
    valid = work[work["date"] >= cut]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train[FEATURES].astype(float))
    Xva = scaler.transform(valid[FEATURES].astype(float))
    ytr = train["target"].to_numpy(dtype=np.float32)
    yva = valid["target"].to_numpy(dtype=np.float32)

    pos = max(float(ytr.sum()), 1.0)
    neg = max(float((1 - ytr).sum()), 1.0)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32)
    model = SurgeNet(len(FEATURES))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=2048, shuffle=True)

    best_loss = math.inf
    patience = 0
    best_state = None
    for _ in range(35):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(torch.tensor(Xva, dtype=torch.float32)), torch.tensor(yva)).item()
        if vl < best_loss - 1e-4:
            best_loss = vl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 6:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(Xva, dtype=torch.float32))).numpy()

    choices = []
    for t in np.linspace(0.50, 0.90, 81):
        prec, rec, spec = _metrics(yva, p, float(t))
        n = int((p >= t).sum())
        if n >= 20:
            choices.append((prec + 0.25 * rec + 0.10 * spec, float(t), prec, rec, spec, n))
    if choices:
        _, threshold, precision, recall, specificity, _ = max(choices, key=lambda x: x[0])
    else:
        vals = []
        for t in np.linspace(0.50, 0.90, 81):
            prec, rec, spec = _metrics(yva, p, float(t))
            vals.append((prec, float(t), rec, spec))
        precision, threshold, recall, specificity = max(vals, key=lambda x: x[0])
    return ModelBundle(model, scaler, threshold, precision, recall, specificity)


def score_latest(bundle: ModelBundle, latest: pd.DataFrame) -> pd.DataFrame:
    work = latest.dropna(subset=FEATURES).copy()
    X = bundle.scaler.transform(work[FEATURES].astype(float))
    bundle.model.eval()
    with torch.no_grad():
        p = torch.sigmoid(bundle.model(torch.tensor(X, dtype=torch.float32))).numpy()
    work["surge_probability"] = p
    work["signal"] = p >= bundle.threshold
    work["expected_move_proxy"] = (work["atr_pct"] * 1.8).clip(0, 0.12)
    work["utility"] = p * (1 + work["expected_move_proxy"]) * np.log1p(work["turnover_cr"].clip(lower=0))
    work["risk_flag"] = np.select(
        [work["gap"] > 0.04, work["atr_pct"] > 0.10, work["turnover_cr"] < 5],
        ["large_gap", "very_high_volatility", "low_liquidity"], default="normal"
    )
    return work.sort_values(["signal", "utility"], ascending=[False, False])
