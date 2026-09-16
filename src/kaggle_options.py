from __future__ import annotations

import gzip
import tempfile
from pathlib import Path
from typing import Iterable

import pandas as pd

DATE_ALIASES = ["date", "Date", "DATE", "timestamp", "Timestamp", "TIMESTAMP", "TRADE_DATE", "trade_date", "DateTime", "datetime", "TradDt"]
SYMBOL_ALIASES = ["symbol", "Symbol", "SYMBOL", "TckrSymb", "TICKER", "underlying", "Underlying", "UNDERLYING", "Ticker", "ticker"]
EXPIRY_ALIASES = ["expiry", "Expiry", "EXPIRY", "expiry_date", "ExpiryDate", "EXPIRY_DT", "XpryDt", "Expiry Date", "expirydate"]
STRIKE_ALIASES = ["strike", "Strike", "STRIKE", "strike_price", "StrikePrice", "STRIKE_PR", "StrkPric", "Strike Price"]
OPT_TYPE_ALIASES = ["option_type", "OptionType", "OPTION_TYPE", "optiontype", "OPTION_TYP", "OptnTp", "OPTTYPE", "type", "Type", "TYPE", "Option Type"]
INSTRUMENT_ALIASES = ["instrument", "Instrument", "INSTRUMENT", "instrument_type", "InstrumentType", "INSTRUMENT_TYPE", "FinInstrmTp", "FinInstrmTpNm", "security_type", "SecurityType"]
PRICE_ALIASES = ["close", "Close", "CLOSE", "ltp", "LTP", "last_price", "LastPrice", "last", "Last", "price", "Price", "ClsPric", "CLSPRIC"]
OPEN_ALIASES = ["open", "Open", "OPEN", "Open Price", "OpnPric", "OPENPRICE"]
HIGH_ALIASES = ["high", "High", "HIGH", "High Price", "HghPric", "HIGHPRICE"]
LOW_ALIASES = ["low", "Low", "LOW", "Low Price", "LwPric", "LOWPRICE"]
VOLUME_ALIASES = ["volume", "Volume", "VOLUME", "vol", "Vol", "Total Volume", "TtlTradgVol", "TOTTRDQTY"]
OI_ALIASES = ["oi", "OI", "open_interest", "OpenInterest", "OPEN_INT", "Open Interest", "OpnIntrst", "OPENINT"]
UNDERLYING_ALIASES = [
    "underlying_close", "UnderlyingClose", "UNDERLYING_CLOSE", "spot", "Spot", "SPOT",
    "spot_price", "SpotPrice", "SPOT_PRICE", "index_close", "IndexClose", "INDEX_CLOSE",
    "index_price", "IndexPrice", "INDEX_PRICE", "NIFTY_CLOSE", "nifty_close", "nifty_spot",
    "NIFTY_SPOT", "underlying_price", "Underlying Price", "underlying value", "Underlying Value",
    "UNDERLYING_VALUE", "UnderlyingValue", "index value", "Index Value", "IndexValue",
    "Nifty Index", "NIFTY Index", "NIFTY_INDEX", "Nifty Spot", "NIFTY Spot",
]


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


def _extract_underlying_proxy_chunk(df: pd.DataFrame) -> pd.DataFrame | None:
    """Extract spot/index/futures values from any raw Kaggle chunk without re-reading it."""
    if df.empty:
        return None
    date_col = _find_col(df.columns, DATE_ALIASES)
    price_col = _find_col(df.columns, PRICE_ALIASES)
    if not date_col or not price_col:
        return None

    x = pd.DataFrame(index=df.index)
    x["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    x["underlying_close"] = pd.to_numeric(df[price_col], errors="coerce")

    symbol_col = _find_col(df.columns, SYMBOL_ALIASES)
    if symbol_col:
        sym = df[symbol_col].astype(str).str.strip().str.upper()
        x = x.loc[sym.str.contains("NIFTY", regex=False, na=False)]
    else:
        x["symbol"] = "NIFTY"

    if x.empty:
        return None

    priority = pd.Series(3, index=x.index, dtype="int64")

    # A direct underlying/index-price column is the strongest signal.
    direct_col = _find_col(df.columns, UNDERLYING_ALIASES)
    if direct_col:
        direct = pd.to_numeric(df.loc[x.index, direct_col], errors="coerce")
        direct_mask = direct.notna() & (direct > 0)
        x.loc[direct_mask, "underlying_close"] = direct.loc[direct_mask]
        priority.loc[direct_mask] = 0

    opt_col = _find_col(df.columns, OPT_TYPE_ALIASES)
    opt_values = None
    if opt_col:
        opt_values = df.loc[x.index, opt_col].astype(str).str.strip().str.upper()
        option_mask = opt_values.isin(["CE", "PE", "CALL", "PUT", "C", "P"])
    else:
        option_mask = pd.Series(False, index=x.index)

    instrument_col = _find_col(df.columns, INSTRUMENT_ALIASES)
    if instrument_col:
        inst = df.loc[x.index, instrument_col].astype(str).str.strip().str.upper()
        spot_mask = inst.str.contains("SPOT|INDEX|IDX|EQUITY", regex=True, na=False)
        fut_mask = inst.str.contains("FUT", regex=True, na=False)
        priority.loc[spot_mask] = priority.loc[spot_mask].clip(upper=1)
        priority.loc[fut_mask & (priority > 1)] = 2
        keep = spot_mask | fut_mask | (priority == 0)
    else:
        # Rows with CE/PE are option rows. A row without an option type is
        # eligible to be a spot/index/futures record.
        keep = (~option_mask) | (priority == 0)

    x = x.loc[keep & x["date"].notna() & x["underlying_close"].notna() & (x["underlying_close"] > 0)].copy()
    if x.empty:
        return None
    x["priority"] = priority.loc[x.index].astype("int8")

    expiry_col = _find_col(df.columns, EXPIRY_ALIASES)
    if expiry_col:
        expiry = pd.to_datetime(df.loc[x.index, expiry_col], errors="coerce").dt.normalize()
        x["days_to_expiry"] = (expiry - x["date"]).dt.days
    else:
        x["days_to_expiry"] = pd.NA
    return x[["date", "underlying_close", "priority", "days_to_expiry"]]


def _collapse_proxy(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=["date", "underlying_close"])
    proxy = pd.concat(parts, ignore_index=True)
    proxy["days_to_expiry"] = pd.to_numeric(proxy["days_to_expiry"], errors="coerce")
    # Prefer direct underlying/spot/index values, then the nearest non-negative
    # futures expiry, then the remaining candidates.
    proxy["future_distance"] = proxy["days_to_expiry"].where(proxy["days_to_expiry"].ge(0), 99999)
    proxy = proxy.sort_values(["date", "priority", "future_distance"])
    return proxy.drop_duplicates("date", keep="first")[["date", "underlying_close"]].sort_values("date").reset_index(drop=True)


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
    out["option_type"] = df[opt_col].astype(str).str.strip().str.upper().replace({"CALL": "CE", "PUT": "PE", "C": "CE", "P": "PE"})
    out["close"] = pd.to_numeric(df[price_col], errors="coerce")
    for name, aliases in [("open", OPEN_ALIASES), ("high", HIGH_ALIASES), ("low", LOW_ALIASES), ("volume", VOLUME_ALIASES), ("open_interest", OI_ALIASES), ("underlying_close", UNDERLYING_ALIASES)]:
        col = _find_col(df.columns, aliases)
        out[name] = pd.to_numeric(df[col], errors="coerce") if col else pd.NA
    out["symbol"] = df[symbol_col].astype(str).str.strip().str.upper() if symbol_col else "NIFTY"
    if symbol_col is None:
        out["symbol"] = "NIFTY"
    out = out.loc[out["symbol"].str.contains("NIFTY", regex=False, na=False) & out["option_type"].isin(["CE", "PE"])]
    out = out.dropna(subset=["date", "expiry", "strike", "close"])
    out = out.loc[out["close"] > 0]
    if out.empty:
        return None
    return out[["date", "expiry", "strike", "option_type", "open", "high", "low", "close", "volume", "open_interest", "underlying_close", "symbol"]]


def _load_underlying_proxy(root: Path, chunksize: int = 200_000) -> pd.DataFrame:
    """Fallback scan for NIFTY spot/index/futures values in downloaded files."""
    parts: list[pd.DataFrame] = []
    for path in _candidate_files(root):
        try:
            for chunk in _iter_tables(path, chunksize=chunksize):
                proxy = _extract_underlying_proxy_chunk(chunk)
                if proxy is not None and not proxy.empty:
                    parts.append(proxy)
        except Exception as exc:
            print(f"Underlying proxy skipped: {path.name}: {type(exc).__name__}: {exc}", flush=True)
    return _collapse_proxy(parts)


def load_from_kaggle_roots(roots: list[str]) -> pd.DataFrame:
    """Build normalized NIFTY option data without retaining raw chunks in memory."""
    work_root = Path(tempfile.mkdtemp(prefix="kaggle-nifty-ingest-"))
    part_dir = work_root / "options"
    proxy_dir = work_root / "proxy"
    part_dir.mkdir(parents=True, exist_ok=True)
    proxy_dir.mkdir(parents=True, exist_ok=True)
    try:
        part_count = 0
        proxy_part_count = 0
        for root_str in roots:
            root = Path(root_str)
            if not root.exists():
                continue
            for path in _candidate_files(root):
                try:
                    for chunk_no, chunk in enumerate(_iter_tables(path), start=1):
                        # Extract the underlying in the same pass as option
                        # normalization. This avoids a second 1+ GB scan and
                        # works even when spot/futures are in final_merged_output.csv.
                        proxy_chunk = _extract_underlying_proxy_chunk(chunk)
                        if proxy_chunk is not None and not proxy_chunk.empty:
                            proxy_chunk.to_parquet(proxy_dir / f"part-{proxy_part_count:06d}.parquet", index=False)
                            proxy_part_count += 1

                        normalized = _normalize_option_chunk(chunk)
                        if normalized is None or normalized.empty:
                            continue
                        normalized.to_parquet(part_dir / f"part-{part_count:06d}.parquet", index=False)
                        part_count += 1
                        if part_count % 10 == 0:
                            print(f"Kaggle ingest progress: parts={part_count}, source={path.name}, chunk={chunk_no}", flush=True)
                except Exception as exc:
                    print(f"Skipping {path}: {type(exc).__name__}: {exc}", flush=True)

        parts = sorted(part_dir.glob("part-*.parquet"))
        if not parts:
            raise RuntimeError("No usable NIFTY option rows found in Kaggle datasets")

        for i, part in enumerate(parts, start=1):
            x = pd.read_parquet(part)
            x = x.sort_values(["date", "expiry", "strike", "option_type"])
            x = x.drop_duplicates(["date", "expiry", "strike", "option_type"], keep="last")
            x.to_parquet(part_dir / f"reduced-{i:06d}.parquet", index=False)
            del x
            if i % 25 == 0:
                print(f"Kaggle reduce progress: {i}/{len(parts)}", flush=True)

        reduced = sorted(part_dir.glob("reduced-*.parquet"))
        options = pd.concat([pd.read_parquet(p) for p in reduced], ignore_index=True)
        options = options.sort_values(["date", "expiry", "strike", "option_type"])
        options = options.drop_duplicates(["date", "expiry", "strike", "option_type"], keep="last")

        proxy_parts = [pd.read_parquet(p) for p in sorted(proxy_dir.glob("part-*.parquet"))]
        proxy = _collapse_proxy(proxy_parts)
        if proxy.empty and roots:
            # Last-resort scan of the downloaded roots for a dedicated spot/
            # futures/index file using the same schema-flexible extractor.
            for root_str in roots:
                proxy = _load_underlying_proxy(Path(root_str))
                if not proxy.empty:
                    break

        # First use any underlying value embedded in the option rows, then
        # fill missing dates from the extracted spot/futures proxy.
        supplied = options[["date", "underlying_close"]].dropna().groupby("date", as_index=False)["underlying_close"].last()
        options = options.drop(columns=["underlying_close"]).merge(
            supplied.rename(columns={"underlying_close": "supplied_underlying"}),
            on="date", how="left", validate="many_to_one",
        )
        if not proxy.empty:
            options = options.merge(proxy.rename(columns={"underlying_close": "proxy_underlying"}), on="date", how="left", validate="many_to_one")
            options["underlying_close"] = options["supplied_underlying"].fillna(options["proxy_underlying"])
            options = options.drop(columns=["supplied_underlying", "proxy_underlying"])
        else:
            options["underlying_close"] = options["supplied_underlying"]
            options = options.drop(columns=["supplied_underlying"])

        options["option_type"] = options["option_type"].astype("category")
        for col in ["strike", "open", "high", "low", "close", "volume", "open_interest", "underlying_close"]:
            options[col] = pd.to_numeric(options[col], errors="coerce", downcast="float")
        return options.sort_values(["date", "expiry", "strike", "option_type"]).reset_index(drop=True)
    finally:
        import shutil
        shutil.rmtree(work_root, ignore_errors=True)
