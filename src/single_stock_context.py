from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_context_dir(data_dir: str, symbol: str | None = None) -> pd.DataFrame:
    """Load dated context. News/actions are filtered to the requested symbol."""
    root = Path(data_dir) / "context"
    if not root.exists():
        return pd.DataFrame(columns=["date"])
    idx, news, ca = [_read(root / f) for f in ("index_daily.csv", "news_daily.csv", "corporate_actions.csv")]
    if not idx.empty:
        idx["date"] = pd.to_datetime(idx["date"])
        out = idx.drop_duplicates("date").copy()
    else:
        dates = []
        if not news.empty and "date" in news: dates.extend(pd.to_datetime(news["date"]).tolist())
        if not ca.empty: dates.extend(pd.to_datetime(ca["ex_date"] if "ex_date" in ca else ca["date"]).tolist())
        out = pd.DataFrame({"date": pd.Series(dates).drop_duplicates().sort_values()})
    if not news.empty and "date" in news and "news_score" in news and "news_count" in news:
        news["date"] = pd.to_datetime(news["date"])
        if symbol is not None and "symbol" in news:
            news = news[news["symbol"].astype(str).str.upper().eq(symbol.upper())]
        news = news.groupby("date", as_index=False).agg(news_score=("news_score", "mean"), news_count=("news_count", "sum"))
        out = out.merge(news, on="date", how="left")
    if not ca.empty:
        ca["date"] = pd.to_datetime(ca["ex_date"] if "ex_date" in ca else ca["date"])
        if symbol is not None and "symbol" in ca:
            ca = ca[ca["symbol"].astype(str).str.upper().eq(symbol.upper())]
        if "corporate_action_flag" in ca:
            ca = ca.groupby("date", as_index=False).agg(corporate_action_flag=("corporate_action_flag", "max"))
            out = out.merge(ca, on="date", how="left")
    return out.sort_values("date").reset_index(drop=True)


def _read(path: Path) -> pd.DataFrame:
    if not path.exists(): return pd.DataFrame()
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
