from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.surge_model import FEATURES


class ReturnNet(nn.Module):
    """Small regression MLP for next-session intraday max return."""

    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 112), nn.LayerNorm(112), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(112, 56), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(56, 20), nn.GELU(), nn.Linear(20, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


@dataclass
class ReturnModelBundle:
    model: ReturnNet
    scaler: StandardScaler
    validation_mae: float
    validation_rank_ic: float


def _rank_ic(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0:
        return 0.0
    return float(pd.Series(y).corr(pd.Series(p), method="spearman"))


def train_expected_return_walk_forward(
    df: pd.DataFrame, validation_days: int = 60, seed: int = 42
) -> ReturnModelBundle:
    """Train without leakage; each row's target is the next-session intraday max return."""
    torch.manual_seed(seed + 17)
    np.random.seed(seed + 17)
    target_col = "next_intraday_max_return"
    work = df.dropna(subset=FEATURES + [target_col]).copy()
    # Winsorise extreme training targets only; the raw target is retained for validation metrics.
    work["_target_train"] = work[target_col].clip(-0.20, 0.30)
    dates = np.array(sorted(work["date"].unique()))
    if len(dates) <= validation_days + 20:
        raise ValueError("Not enough trading history for expected-return walk-forward training")
    cut = dates[-validation_days]
    train = work[work["date"] < cut]
    valid = work[work["date"] >= cut]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train[FEATURES].astype(float))
    Xva = scaler.transform(valid[FEATURES].astype(float))
    ytr = train["_target_train"].to_numpy(dtype=np.float32)
    yva = valid[target_col].to_numpy(dtype=np.float32)

    model = ReturnNet(len(FEATURES))
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=3e-4)
    loss_fn = nn.HuberLoss(delta=0.03)
    ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=2048, shuffle=True)

    best_loss = math.inf
    patience = 0
    best_state = None
    for _ in range(40):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(
                model(torch.tensor(Xva, dtype=torch.float32)),
                torch.tensor(np.clip(yva, -0.20, 0.30), dtype=torch.float32),
            ).item()
        if vl < best_loss - 1e-5:
            best_loss = vl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 7:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xva, dtype=torch.float32)).numpy()
    mae = float(np.mean(np.abs(pred - yva)))
    ric = _rank_ic(yva, pred)
    return ReturnModelBundle(model, scaler, mae, ric)


def score_expected_return(bundle: ReturnModelBundle, latest: pd.DataFrame) -> pd.DataFrame:
    work = latest.dropna(subset=FEATURES).copy()
    X = bundle.scaler.transform(work[FEATURES].astype(float))
    bundle.model.eval()
    with torch.no_grad():
        pred = bundle.model(torch.tensor(X, dtype=torch.float32)).numpy()
    work["expected_intraday_return"] = np.clip(pred, -0.20, 0.30)
    # Rank/utility used only for portfolio construction; no future data enters this score.
    work["expected_return_utility"] = (
        work["expected_intraday_return"]
        / (0.01 + work["atr_pct"].abs())
        * np.log1p(work["turnover_cr"].clip(lower=0))
    )
    return work
