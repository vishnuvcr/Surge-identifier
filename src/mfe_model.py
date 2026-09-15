from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.surge_model import FEATURES

TARGETS = (0.025, 0.03, 0.04, 0.05)


class MFENet(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_features, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.18),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(64, 24), nn.GELU(),
        )
        self.mfe = nn.Linear(24, 1)
        self.hit = nn.Linear(24, len(TARGETS))

    def forward(self, x):
        h = self.body(x)
        return self.mfe(h).squeeze(-1), self.hit(h)


@dataclass
class MFEModelBundle:
    model: MFENet
    scaler: StandardScaler
    validation_mae: float
    validation_rank_ic: float
    validation_hit_rate: dict
    calibration: dict
    validation_brier: dict


def _rank_ic(y, p):
    if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0:
        return 0.0
    return float(pd.Series(y).corr(pd.Series(p), method="spearman"))


def _brier(actual, pred):
    return float(np.mean((np.asarray(pred) - np.asarray(actual)) ** 2))


def train_mfe_walk_forward(df: pd.DataFrame, validation_days: int = 60, seed: int = 42) -> MFEModelBundle:
    torch.manual_seed(seed + 71)
    np.random.seed(seed + 71)
    work = df.dropna(subset=FEATURES + ["next_intraday_max_return"]).copy()
    dates = np.array(sorted(work.date.unique()))
    if len(dates) <= validation_days + 20:
        raise ValueError("Not enough history for MFE walk-forward training")
    cut = dates[-validation_days]
    train = work[work.date < cut].copy()
    valid = work[work.date >= cut].copy()

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train[FEATURES].astype(float))
    Xva = scaler.transform(valid[FEATURES].astype(float))
    ytr = train.next_intraday_max_return.to_numpy(float)
    yva = valid.next_intraday_max_return.to_numpy(float)
    ytr_clip = np.clip(ytr, -0.20, 0.30).astype(np.float32)
    htr = np.column_stack([(ytr >= t).astype(np.float32) for t in TARGETS])

    model = MFENet(len(FEATURES))
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=3e-4)
    reg_loss = nn.HuberLoss(delta=0.03)
    pos = htr.sum(axis=0)
    neg = len(htr) - pos
    pos_weight = torch.tensor(np.maximum(neg / np.maximum(pos, 1), 1.0), dtype=torch.float32)
    cls_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr_clip), torch.tensor(htr))
    loader = DataLoader(ds, batch_size=2048, shuffle=True)

    best = math.inf
    patience = 0
    state = None
    for _ in range(45):
        model.train()
        for xb, yb, hb in loader:
            opt.zero_grad(set_to_none=True)
            pred_mfe, pred_hit = model(xb)
            loss = reg_loss(pred_mfe, yb) + 0.65 * cls_loss(pred_hit, hb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            xv = torch.tensor(Xva, dtype=torch.float32)
            pm, ph = model(xv)
            vl = reg_loss(pm, torch.tensor(np.clip(yva, -0.20, 0.30), dtype=torch.float32)).item()
            vl += 0.65 * cls_loss(ph, torch.tensor(np.column_stack([(yva >= t).astype(np.float32) for t in TARGETS]))).item()
        if vl < best - 1e-5:
            best = vl
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 8:
                break
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        pm, ph = model(torch.tensor(Xva, dtype=torch.float32))
        pred_mfe = pm.numpy()
        raw_probs = torch.sigmoid(ph).numpy()

    mae = float(np.mean(np.abs(pred_mfe - yva)))
    ric = _rank_ic(yva, pred_mfe)
    hit_rate = {}
    calibration = {}
    brier = {}
    for i, t in enumerate(TARGETS):
        actual = (yva >= t).astype(int)
        raw = raw_probs[:, i]
        if actual.sum() == 0 or actual.sum() == len(actual):
            cal = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            # Degenerate validation: preserve the empirical rate rather than fitting an unstable curve.
            cal.fit([0.0, 1.0], [float(actual.mean()), float(actual.mean())])
        else:
            cal = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            cal.fit(raw, actual)
        calibrated = np.asarray(cal.predict(raw), dtype=float)
        calibration[str(t)] = cal
        hit_rate[str(t)] = float((calibrated >= 0.5).eq(actual).mean() if isinstance(pd.Series(calibrated), pd.Series) else np.mean((calibrated >= 0.5) == actual))
        brier[str(t)] = {"raw": _brier(actual, raw), "calibrated": _brier(actual, calibrated), "base_rate": float(actual.mean())}

    return MFEModelBundle(model, scaler, mae, ric, hit_rate, calibration, brier)


def _score(bundle: MFEModelBundle, latest: pd.DataFrame, calibrated: bool) -> pd.DataFrame:
    work = latest.dropna(subset=FEATURES).copy()
    X = bundle.scaler.transform(work[FEATURES].astype(float))
    bundle.model.eval()
    with torch.no_grad():
        pm, ph = bundle.model(torch.tensor(X, dtype=torch.float32))
        mfe = pm.numpy()
        probs = torch.sigmoid(ph).numpy()
    work["predicted_mfe"] = np.clip(mfe, -0.20, 0.30)
    for i, t in enumerate(TARGETS):
        p = probs[:, i]
        if calibrated:
            p = np.asarray(bundle.calibration[str(t)].predict(p), dtype=float)
        work[f"p_hit_{str(t).rstrip('0').rstrip('.').replace('.', '_')}" if t == 0.025 else f"p_hit_{int(t*100)}"] = np.clip(p, 0.0, 1.0)
    return work


def score_mfe(bundle: MFEModelBundle, latest: pd.DataFrame) -> pd.DataFrame:
    return _score(bundle, latest, calibrated=False)


def score_mfe_calibrated(bundle: MFEModelBundle, latest: pd.DataFrame) -> pd.DataFrame:
    return _score(bundle, latest, calibrated=True)
