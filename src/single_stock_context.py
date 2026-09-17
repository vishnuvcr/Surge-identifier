from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_context_dir(data_dir: str) -> pd.DataFrame:
    """Load pre-downloaded, dated context files without forward filling from the future.

    Expected optional files under data_dir/context:
      index_daily.csv: date,nifty_ret1,nifty_ret5,nifty_gap,india_vix_ret1,sector_ret1,sector_ret5,breadth
      news_daily.csv: date,symbol,news_score,news_count
      corporate_actions.csv: ex_date,symbol,corporate_action_flag

    News rows must be based only on publications timestamped before the next session open.
    """
    root = Path(data_dir) / "context"
    if not root.exists():
        return pd.DataFrame(columns=["date", "news_score", "news_count", "corporate_action_flag"])

    idx = _read(root / "index_daily.csv")
    news = _read(root / "news_daily.csv")
    ca = _read(root / "corporate_actions.csv")
    if idx.empty and news.empty and ca.empty:
        return pd.DataFrame(columns=["date"])

    if not idx.empty:
        idx["date"] = pd.to_datetime(idx["date"])
        out = idx.copy()
    else:
        dates = pd.Series(dtype="datetime64[ns]")
        for z in [news, ca]:
            if not z.empty:
                col = "date" if "date" in z else "ex_date"
                dates = pd.concat([dates, pd.to_datetime(z[col])], ignore_index=True)
        out = pd.DataFrame({"date": pd.Series(dates).drop_duplicates().sort_values()})

    if not news.empty:
        news["date"] = pd.to_datetime(news["date"])
        if "symbol" in news.columns:
            # Runner selects a symbol later; aggregate only if no symbol dimension exists.
            news = news.groupby("date", as_index=False).agg(
                news_score=("news_score", "mean"), news_count=("news_count", "sum")
            )
        out = out.merge(news[["date", "news_score", "news_count"]], on="date", how="left")

    if not ca.empty:
        ca["date"] = pd.to_datetime(ca["ex_date"] if "ex_date" in ca else ca["date"])
        ca = ca.groupby("date", as_index=False).agg(corporate_action_flag=("corporate_action_flag", "max"))
        out = out.merge(ca, on="date", how="left")

    return out.sort_values("date").reset_index(drop=True)


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)
