from __future__ import annotations

import gzip
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

import pandas as pd


DATE_ALIASES = ["date", "Date", "DATE", "timestamp", "Timestamp", "TIMESTAMP", "TRADE_DATE", "trade_date"]
SYMBOL_ALIASES = ["symbol", "Symbol", "SYMBOL", "TckrSymb", "TICKER", "underlying", "Underlying", "UNDERLYING"]
EXPIRY_ALIASES = ["expiry", "Expiry", "EXPIRY", "expiry_date", "ExpiryDate", "EXPIRY_DT", "XpryDt"]
STRIKE_ALIASES = ["strike", "Strike", "STRIKE", "strike_price", "StrikePrice", "STRIKE_PR", "StrkPric"]
OPT_TYPE_ALIASES = ["option_type", "OptionType", "OPTION_TYPE", "optiontype", "OPTION_TYP", "OptnTp", "type", "Type", "TYPE"]
PRICE_ALIASES = ["close", "Close", "CLOSE", "ltp", "LTP", "last_price", "LastPrice", "last", "Last"]
OPEN_ALIASES = ["open", "Open", "OPEN"]
HIGH_ALIASES = ["high", "High", "HIGH"]
LOW_ALIASES = ["low", "Low", "LOW"]
VOLUME_ALIASES = ["volume", "Volume", "VOLUME", "vol", "Vol"]
OI_ALIASES = ["oi", "OI", "open_interest", "OpenInterest", "OPEN_INT"]
UNDERLYING_ALIASES = ["underlying_close", "UnderlyingClose", "UNDERLYING_CLOSE", "spot", "Spot", "SPOT", "index_close", "IndexClose", "NIFTY_CLOSE"]


def _find_col(columns: Iterable[str], aliases: list[str]) -> str | None:
    by_lower = {str(c).strip().lower(): c for c in columns}
    for alias in aliases:
        if alias.lower() in by_lower:
            return by_lower[alias.lower()]
    return None


def _candidate_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".csv", ".txt", ".parquet", ".gz"}:
            continue
        s = str(p).lower()
        if "banknifty" in s or "bank_nifty" in s:
            continue
        if p.name.startswith("."):
            continue
        files.append(p)
    return sorted(files, key=lambda x: str(x).lower())


def _read_csv_chunks(path: Path, chunksize: int = 200_000):
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt") as fh:
            yield from pd.read_csv(fh, chunksize=chunksize, low_memory=False)
    else:
        yield from pd.read_csv(path, chunksize=chunksize, low_memory=False)


def _iter_tables(path: Path, chunksize: int = 200_000):
    if path.suffix.lower() == ".parquet":
        table = pd.read_parquet(path)
        yield table
    else:
        yield from _read_csv_chunks(path, chunksize=chunksize)


def _normalize_option_chunk(df: pd.DataFrame) -> pd.DataFrame | None:
    if df.empty:
        return None

    date_col = _find_col(df.columns, DATE_ALIASES)
    symbol_col = _find_col(df.columns, SYMBOL_ALIASES)
    expiry_col = _find_col(df.columns, EXPIRY_ALIASES)
    strike_col = _find_col(df.columns, STRIKE_ALIASES)
    opt_col = _find_col(df.columns, OPT_TYPE_ALIASES)
    price_col = _find_col(df.columns, PRICE_ALIASES)

    if not all([date_col, expiry_col, strike_col, opt_col, price_col]):
        return None

    out = pd.DataFrame(index=df.index)
    out["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    out["expiry"] = pd.to_datetime(df[expiry_col], errors="coerce").dt.normalize()
    out["strike"] = pd.to_numeric(df[strike_col], errors="coerce")
    out["option_type"] = (
        df[opt_col].astype(str).str.strip().str.upper().replace({"CALL": "CE", "PUT": "PE", "C": "CE", "P": "PE"})
    )
    out["close"] = pd.to_numeric(df[price_col], errors="coerce")

    for name, aliases in [("open", OPEN_ALIASES), ("high", HIGH_ALIASES), ("low", LOW_ALIASES), ("volume", VOLUME_ALIASES), ("open_interest", OI_ALIASES), ("underlying_close", UNDERLYING_ALIASES)]:
        col = _find_col(df.columns, aliases)
        out[name] = pd.to_numeric(df[col], errors="coerce") if col else pd.NA

    if symbol_col:
        out["symbol"] = df[symbol_col].astype(str).str.strip().str.upper()
        mask_nifty = out["symbol"].str.contains(r"(^|[^A-Z])NIFTY([^A-Z]|$)|NIFTY", regex=True, na=False)
        out = out.loc[mask_nifty]
    else:
        out["symbol"] = "NIFTY"

    out = out.loc[out["option_type"].isin(["CE", "PE"])]
    out = out.dropna(subset=["date", "expiry", "strike", "close"])
    out = out.loc[out["close"] > 0]
    if out.empty:
        return None

    # Some datasets provide an underlying/index close alongside each option row.
    # Keep it here; other sources can be merged later by date.
    return out[["date", "expiry", "strike", "option_type", "open", "high", "low", "close", "volume", "open_interest", "underlying_close", "symbol"]]


def _load_spot_proxy(root: Path, chunksize: int = 200_000) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in _candidate_files(root):
        s = str(path).lower()
        if not any(token in s for token in ("spot", "index", "future", "futures", "nifty50")):
            continue
        for chunk in _iter_tables(path, chunksize=chunksize):
            if chunk.empty:
                continue
            date_col = _find_col(chunk.columns, DATE_ALIASES)
            price_col = _find_col(chunk.columns, PRICE_ALIASES)
            symbol_col = _find_col(chunk.columns, SYMBOL_ALIASES)
            if not date_col or not price_col:
                continue
            x = pd.DataFrame({
                "date": pd.to_datetime(chunk[date_col], errors="coerce").dt.normalize(),
                "underlying_close": pd.to_numeric(chunk[price_col], errors="coerce"),
            })
            if symbol_col:
                sym = chunk[symbol_col].astype(str).str.upper()
                x = x.loc[sym.str.contains("NIFTY", na=False)]
            x = x.dropna(subset=["date", "underlying_close"])
            if not x.empty:
                parts.append(x.groupby("date", as_index=False)["underlying_close"].last())
    if not parts:
        return pd.DataFrame(columns=["date", "underlying_close"])
    proxy = pd.concat(parts, ignore_index=True)
    return proxy.drop_duplicates("date", keep="last").sort_values("date")


def load_from_kaggle_roots(roots: list[str]) -> pd.DataFrame:
    """Load NIFTY option data using bounded-memory, disk-backed Parquet parts."""
    work_root = Path(tempfile.mkdtemp(prefix="kaggle-nifty-ingest-"))
    option_part_dir = work_root / "options"
    option_part_dir.mkdir(parents=True, exist_ok=True)
    part_index = 0
    option_files = 0

    try:
        for root_str in roots:
            root = Path(root_str)
            if not root.exists():
                continue
            for path in _candidate_files(root):
                wrote_for_file = False
                try:
                    for chunk_no, chunk in enumerate(_iter_tables(path), start=1):
                        normalized = _normalize_option_chunk(chunk)
                        if normalized is None or normalized.empty:
                            continue
                        part_path = option_part_dir / f"part-{part_index:06d}.parquet"
                        normalized.to_parquet(part_path, index=False)
                        part_index += 1
                        wrote_for_file = True
                        if part_index % 10 == 0:
                            print(f"Kaggle ingest progress: parts={part_index}, source={path.name}, chunk={chunk_no}", flush=True)
                except (pd.errors.EmptyDataError, UnicodeDecodeError, OSError, ValueError) as exc:
                    print(f"Skipping unreadable file {path}: {exc}", flush=True)
                option_files += int(wrote_for_file)

        part_paths = sorted(option_part_dir.glob("part-*.parquet"))
        if not part_paths:
            raise RuntimeError("No usable NIFTY option rows found in Kaggle datasets")

        frames: list[pd.DataFrame] = []
        for i, part in enumerate(part_paths, start=1):
            frames.append(pd.read_parquet(part))
            if i % 25 == 0:
                print(f"Kaggle combine progress: loaded_parts={i}/{len(part_paths)}", flush=True)

        options = pd.concat(frames, ignore_index=True)
        del frames

        options = options.sort_values(["date", "expiry", "strike", "option_type"])
        options = options.drop_duplicates(["date", "expiry", "strike", "option_type"], keep="last")

        # Merge any explicitly supplied underlying value first.
        explicit = options[["date", "underlying_close"]].dropna()
        if not explicit.empty:
            explicit = explicit.groupby("date", as_index=False)["underlying_close"].last()
            options = options.drop(columns=["underlying_close"]).merge(explicit, on="date", how="left")
        else:
            options["underlying_close"] = pd.NA

        # Add spot/futures proxy only when it fills missing dates.
        missing = options["underlying_close"].isna().mean()
        if missing > 0:
            proxies = []
            for root_str in roots:
                proxy = _load_spot_proxy(Path(root_str))
                if not proxy.empty:
                    proxies.append(proxy)
            if proxies:
                proxy = pd.concat(proxies, ignore_index=True).drop_duplicates("date", keep="last")
                options = options.drop(columns=["underlying_close"]).merge(proxy, on="date", how="left")

        # Compact dtypes before returning to the main research process.
        options["option_type"] = options["option_type"].astype("category")
        for col in ["strike", "open", "high", "low", "close", "volume", "open_interest", "underlying_close"]:
            options[col] = pd.to_numeric(options[col], errors="coerce", downcast="float")

        options = options.sort_values(["date", "expiry", "strike", "option_type"]).reset_index(drop=True)
        print(f"Kaggle ingest complete: parts={len(part_paths)}, source_files={option_files}, rows={len(options)}, dates={options['date'].nunique()}", flush=True)
        return options
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
