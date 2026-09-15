from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.nse_data import download_day, load_prices
from src.nse_fo import _download as download_fo_day, load_fo

IST = timezone(timedelta(hours=5, minutes=30))


def _business_dates(start: date, end: date) -> list[date]:
    if start > end:
        return []
    return [
        start + timedelta(days=i)
        for i in range((end - start).days + 1)
        if (start + timedelta(days=i)).weekday() < 5
    ]


def _aggregate_fo(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=[
            "date", "symbol", "fut_oi", "fut_oi_change", "fut_volume",
            "call_oi", "put_oi", "pcr",
        ])
    raw = raw.copy()
    for col in ["instrument", "option_type", "symbol"]:
        raw[col] = raw[col].astype(str).str.upper()
    fut = raw[raw["instrument"].str.contains("FUT", na=False)].groupby(["date", "symbol"]).agg(
        fut_oi=("oi", "sum"),
        fut_oi_change=("oi_change", "sum"),
        fut_volume=("volume", "sum"),
    )
    calls = raw[raw["option_type"].isin(["CE", "CA", "CALL"])].groupby(["date", "symbol"])["oi"].sum().rename("call_oi")
    puts = raw[raw["option_type"].isin(["PE", "PA", "PUT"])].groupby(["date", "symbol"])["oi"].sum().rename("put_oi")
    out = fut.join(calls, how="outer").join(puts, how="outer").fillna(0).reset_index()
    out["pcr"] = out["put_oi"] / out["call_oi"].replace(0, pd.NA)
    return out


def refresh_cash(cache_dir: Path, lookback_days: int, force_rebuild: bool) -> pd.DataFrame:
    parquet = cache_dir / "prices.parquet"
    if force_rebuild or not parquet.exists():
        return load_prices(str(cache_dir), lookback_days)

    old = pd.read_parquet(parquet)
    if old.empty:
        return load_prices(str(cache_dir), lookback_days)

    old["date"] = pd.to_datetime(old["date"])
    max_date = old["date"].dt.date.max()
    today = datetime.now(IST).date()
    needed = _business_dates(max_date + timedelta(days=1), today)
    frames = [old]
    for dt in needed:
        x = download_day(dt, cache_dir / "raw")
        if x is not None and not x.empty:
            frames.append(x)
    data = pd.concat(frames, ignore_index=True)
    data["date"] = pd.to_datetime(data["date"])
    data = data.drop_duplicates(["date", "symbol"], keep="last").sort_values(["date", "symbol"])
    data.to_parquet(parquet, index=False)
    return data


def refresh_fo(cache_dir: Path, lookback_days: int, force_rebuild: bool) -> pd.DataFrame:
    parquet = cache_dir / "fo.parquet"
    if force_rebuild or not parquet.exists():
        return load_fo(str(cache_dir), lookback_days)

    old = pd.read_parquet(parquet)
    required = {"date", "symbol", "fut_oi", "fut_oi_change", "fut_volume", "call_oi", "put_oi", "pcr"}
    if old.empty or not required.issubset(old.columns):
        return load_fo(str(cache_dir), lookback_days)

    old["date"] = pd.to_datetime(old["date"])
    max_date = old["date"].dt.date.max()
    today = datetime.now(IST).date()
    needed = _business_dates(max_date + timedelta(days=1), today)
    raw_frames: list[pd.DataFrame] = []
    for dt in needed:
        x = download_fo_day(dt, cache_dir / "fo_raw")
        if x is not None and not x.empty:
            raw_frames.append(x)
    if raw_frames:
        new = _aggregate_fo(pd.concat(raw_frames, ignore_index=True))
        out = pd.concat([old, new], ignore_index=True)
        out["date"] = pd.to_datetime(out["date"])
        out = out.drop_duplicates(["date", "symbol"], keep="last").sort_values(["date", "symbol"])
        out.to_parquet(parquet, index=False)
        return out
    return old


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(cache_dir: Path, lookback_days: int, prices: pd.DataFrame, fo: pd.DataFrame) -> dict:
    prices["date"] = pd.to_datetime(prices["date"])
    fo["date"] = pd.to_datetime(fo["date"])
    manifest = {
        "schema_version": 1,
        "dataset_name": "Surge Identifier NSE master dataset",
        "lookback_days_requested": int(lookback_days),
        "generated_at_ist": datetime.now(IST).isoformat(),
        "cash": {
            "rows": int(len(prices)),
            "symbols": int(prices["symbol"].nunique()),
            "min_date": prices["date"].min().date().isoformat(),
            "max_date": prices["date"].max().date().isoformat(),
            "sha256": _sha256(cache_dir / "prices.parquet"),
        },
        "fno": {
            "rows": int(len(fo)),
            "symbols": int(fo["symbol"].nunique()),
            "min_date": fo["date"].min().date().isoformat() if not fo.empty else None,
            "max_date": fo["date"].max().date().isoformat() if not fo.empty else None,
            "sha256": _sha256(cache_dir / "fo.parquet"),
        },
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (cache_dir / "manifest.json").write_text(manifest_text, encoding="utf-8")
    return manifest


def validate(cache_dir: Path) -> None:
    prices = pd.read_parquet(cache_dir / "prices.parquet")
    fo = pd.read_parquet(cache_dir / "fo.parquet")
    required_cash = {"date", "symbol", "open", "high", "low", "close", "prev_close", "volume", "turnover"}
    required_fo = {"date", "symbol", "fut_oi", "fut_oi_change", "fut_volume", "call_oi", "put_oi", "pcr"}
    assert required_cash.issubset(prices.columns), f"Missing cash columns: {required_cash - set(prices.columns)}"
    assert required_fo.issubset(fo.columns), f"Missing F&O columns: {required_fo - set(fo.columns)}"
    assert not prices.empty and len(prices) > 1000, f"Cash dataset unexpectedly small: {len(prices):,} rows"
    assert not fo.empty and len(fo) > 1000, f"F&O dataset unexpectedly small: {len(fo):,} rows"
    assert prices["date"].notna().all(), "Cash dates contain NA"
    assert fo["date"].notna().all(), "F&O dates contain NA"
    assert prices[["symbol", "date"]].duplicated().sum() == 0, "Duplicate cash symbol/date rows"
    assert fo[["symbol", "date"]].duplicated().sum() == 0, "Duplicate F&O symbol/date rows"


def main() -> None:
    p = argparse.ArgumentParser(description="Build or incrementally refresh the persistent NSE master dataset.")
    p.add_argument("--lookback-days", type=int, default=1500)
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--force-rebuild", action="store_true")
    args = p.parse_args()

    if args.lookback_days < 300:
        raise SystemExit("lookback-days is unexpectedly small; refusing to reduce research history")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    prices = refresh_cash(cache_dir, args.lookback_days, args.force_rebuild)
    fo = refresh_fo(cache_dir, args.lookback_days, args.force_rebuild)
    validate(cache_dir)
    manifest = build_manifest(cache_dir, args.lookback_days, prices, fo)

    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    print("MASTER DATASET REFRESH PASSED", flush=True)


if __name__ == "__main__":
    main()
