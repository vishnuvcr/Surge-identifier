from __future__ import annotations

import gzip
import tempfile
from pathlib import Path
from typing import Iterable

import pandas as pd

DATE_ALIASES = ["date", "Date", "DATE", "timestamp", "Timestamp", "TIMESTAMP", "TRADE_DATE", "trade_date", "DateTime", "datetime"]
SYMBOL_ALIASES = ["symbol", "Symbol", "SYMBOL", "TckrSymb", "TICKER", "underlying", "Underlying", "UNDERLYING", "Ticker", "ticker"]
EXPIRY_ALIASES = ["expiry", "Expiry", "EXPIRY", "expiry_date", "ExpiryDate", "EXPIRY_DT", "XpryDt", "Expiry Date", "expirydate"]
STRIKE_ALIASES = ["strike", "Strike", "STRIKE", "strike_price", "StrikePrice", "STRIKE_PR", "StrkPric", "Strike Price"]
OPT_TYPE_ALIASES = ["option_type", "OptionType", "OPTION_TYPE", "optiontype", "OPTION_TYP", "OptnTp", "type", "Type", "TYPE", "Option Type"]
PRICE_ALIASES = ["close", "Close", "CLOSE", "ltp", "LTP", "last_price", "LastPrice", "last", "Last", "price", "Price"]
OPEN_ALIASES = ["open", "Open", "OPEN", "Open Price"]
HIGH_ALIASES = ["high", "High", "HIGH", "High Price"]
LOW_ALIASES = ["low", "Low", "LOW", "Low Price"]
VOLUME_ALIASES = ["volume", "Volume", "VOLUME", "vol", "Vol", "Total Volume"]
OI_ALIASES = ["oi", "OI", "open_interest", "OpenInterest", "OPEN_INT", "Open Interest"]
UNDERLYING_ALIASES = ["underlying_close", "UnderlyingClose", "UNDERLYING_CLOSE", "spot", "Spot", "SPOT", "index_close", "IndexClose", "NIFTY_CLOSE", "underlying_price", "Underlying Price"]


def _find_col(columns: Iterable[str], aliases: list[str]) -> str | None:
    by_lower = {str(c).strip().lower(): c for c in columns}
    for alias in aliases:
        if alias.lower() in by_lower:
            return by_lower[alias.lower()]
    return None


def _candidate_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".csv", ".txt", ".parquet", ".gz"}:
            continue
        name = str(p).lower()
        if p.name.startswith(".") or "banknifty" in name or "bank_nifty" in name:
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
        yield pd.read_parquet(path)
    else:
        yield from _read_csv_chunks(path, chunksize=chunksize)


def _normalize_option_chunk(df: pd.DataFrame) -> pd.DataFrame | None:
    if df.empty:
        return None
    date_col = _find_col(df.columns, DATE_ALIASES)
    expiry_col = _find_col(df.columns, EXPIRY_ALIASES)
    strike_col = _find_col(df.columns, STRIKE_ALIASES)
    opt_col = _find_col(df.columns, OPT_TYPE_ALIASES)
    price_col = _find_col(df.columns, PRICE_ALIASES)
    symbol_col = _find_col(df.columns, SYMBOL_ALIASES)
    if not all([date_col, expiry_col, strike_col, opt_col, price_col]):
        return None

    out = pd.DataFrame(index=df.index)
    out["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    out["expiry"] = pd.to_datetime(df[expiry_col], errors="coerce").dt.normalize()
    out["strike"] = pd.to_numeric(df[strike_col], errors="coerce")
    out["option_type"] = (
        df[opt_col].astype(str).str.strip().str.upper()
        .replace({"CALL": "CE", "PUT": "PE", "C": "CE", "P": "PE"})
    )
    out["close"] = pd.to_numeric(df[price_col], errors="coerce")
    for name, aliases in [
        ("open", OPEN_ALIASES),
        ("high", HIGH_ALIASES),
        ("low", LOW_ALIASES),
        ("volume", VOLUME_ALIASES),
        ("open_interest", OI_ALIASES),
        ("underlying_close", UNDERLYING_ALIASES),
    ]:
        col = _find_col(df.columns, aliases)
        out[name] = pd.to_numeric(df[col], errors="coerce") if col else pd.NA

    out["symbol"] = df[symbol_col].astype(str).str.strip().str.upper() if symbol_col else "NIFTY"
    # Some option-chain datasets omit an explicit index symbol. Their contract schema
    # is unambiguous enough for this research when the file itself is known to be NIFTY.
    if symbol_col is None:
        out["symbol"] = "NIFTY"
    mask_nifty = out["symbol"].str.contains("NIFTY", regex=False, na=False)
    out = out.loc[mask_nifty & out["option_type"].isin(["CE", "PE"])]
    out = out.dropna(subset=["date", "expiry", "strike", "close"])
    out = out.loc[out["close"] > 0]
    if out.empty:
        return None
    return out[["date", "expiry", "strike", "option_type", "open", "high", "low", "close", "volume", "open_interest", "underlying_close", "symbol"]]


def _load_spot_proxy(root: Path, chunksize: int = 200_000) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in _candidate_files(root):
        s = str(path).lower()
        if not any(token in s for token in ("spot", "index", "future", "futures", "nifty50", "underlying", "merged")):
            continue
        try:
            for chunk in _iter_tables(path, chunksize=chunksize):
                date_col = _find_col(chunk.columns, DATE_ALIASES)
                price_col = _find_col(chunk.columns, PRICE_ALIASES)
                if not date_col or not price_col:
                    continue
                x = pd.DataFrame({
                    "date": pd.to_datetime(chunk[date_col], errors="coerce").dt.normalize(),
                    "underlying_close": pd.to_numeric(chunk[price_col], errors="coerce"),
                })
                sym_col = _find_col(chunk.columns, SYMBOL_ALIASES)
                if sym_col:
                    sym = chunk[sym_col].astype(str).str.upper()
                    x = x.loc[sym.str.contains("NIFTY", na=False)]
                x = x.dropna(subset=["date", "underlying_close"])
                if not x.empty:
                    parts.append(x.groupby("date", as_index=False)["underlying_close"].last())
        except Exception:
            continue
    if not parts:
        return pd.DataFrame(columns=["date", "underlying_close"])
    return (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def load_from_kaggle_roots(roots: list[str]) -> pd.DataFrame:
    """Build normalized NIFTY option data without retaining raw chunks in memory."""
    work_root = Path(tempfile.mkdtemp(prefix="kaggle-nifty-ingest-"))
    part_dir = work_root / "options"
    part_dir.mkdir(parents=True, exist_ok=True)
    part_count = 0
    try:
        for root_str in roots:
            root = Path(root_str)
            if not root.exists():
                continue
            for path in _candidate_files(root):
                wrote = False
                try:
                    for chunk_no, chunk in enumerate(_iter_tables(path), start=1):
                        normalized = _normalize_option_chunk(chunk)
                        if normalized is None or normalized.empty:
                            continue
                        normalized.to_parquet(part_dir / f"part-{part_count:06d}.parquet", index=False)
                        part_count += 1
                        wrote = True
                        if part_count % 10 == 0:
                            print(f"Kaggle ingest progress: parts={part_count}, source={path.name}, chunk={chunk_no}", flush=True)
                except Exception as exc:
                    print(f"Skipping {path}: {type(exc).__name__}: {exc}", flush=True)
                # Do not stop just because a non-option file was encountered.
                _ = wrote

        parts = sorted(part_dir.glob("part-*.parquet"))
        if not parts:
            raise RuntimeError("No usable NIFTY option rows found in Kaggle datasets")

        # Reduce each part independently, then concatenate only the compact result.
        for i, part in enumerate(parts, start=1):
            x = pd.read_parquet(part)
            x = x.sort_values(["date", "expiry", "strike", "option_type"])
            x = x.drop_duplicates(["date", "expiry", "strike", "option_type"], keep="last")
            x.to_parquet(part_dir / f"reduced-{i:06d}.parquet", index=False)
            del x
            if i % 25 == 0:
                print(f"Kaggle reduce progress: {i}/{len(parts)}", flush=True)

        reduced = sorted(part_dir.glob("reduced-*.parquet"))
        frames = [pd.read_parquet(p) for p in reduced]
        options = pd.concat(frames, ignore_index=True)
        del frames
        options = options.sort_values(["date", "expiry", "strike", "option_type"])
        options = options.drop_duplicates(["date", "expiry", "strike", "option_type"], keep="last")

        # Prefer any underlying value supplied by the source. If absent, build a daily
        # proxy from NIFTY spot/index/future files rather than silently leaving it null.
        supplied = options[["date", "underlying_close"]].dropna()
        if not supplied.empty:
            supplied = supplied.groupby("date", as_index=False)["underlying_close"].last()
            options = options.drop(columns=["underlying_close"]).merge(supplied, on="date", how="left", validate="many_to_one")

        if options["underlying_close"].isna().any():
            proxies = []
            for root_str in roots:
                p = _load_spot_proxy(Path(root_str))
                if not p.empty:
                    proxies.append(p)
            if proxies:
                proxy = pd.concat(proxies, ignore_index=True).drop_duplicates("date", keep="last")
                options = options.drop(columns=["underlying_close"]).merge(proxy, on="date", how="left", validate="many_to_one")

        options["option_type"] = options["option_type"].astype("category")
        for col in ["strike", "open", "high", "low", "close", "volume", "open_interest", "underlying_close"]:
            options[col] = pd.to_numeric(options[col], errors="coerce", downcast="float")
        return options.sort_values(["date", "expiry", "strike", "option_type"]).reset_index(drop=True)
    finally:
        import shutil
        shutil.rmtree(work_root, ignore_errors=True)
