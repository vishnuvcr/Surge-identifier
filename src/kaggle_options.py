from __future__ import annotations

import gzip
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

import pandas as pd

DATE_ALIASES = [
    "date", "Date", "DATE", "timestamp", "Timestamp", "TIMESTAMP",
    "TRADE_DATE", "trade_date", "DateTime", "datetime", "TradDt"
]
SYMBOL_ALIASES = [
    "symbol", "Symbol", "SYMBOL", "TckrSymb", "TICKER", "Ticker", "ticker"
]
EXPIRY_ALIASES = [
    "expiry", "Expiry", "EXPIRY", "expiry_date", "ExpiryDate", "EXPIRY_DT",
    "XpryDt", "Expiry Date", "expirydate"
]
STRIKE_ALIASES = [
    "strike", "Strike", "STRIKE", "strike_price", "StrikePrice", "STRIKE_PR",
    "StrkPric", "Strike Price"
]
OPT_TYPE_ALIASES = [
    "option_type", "OptionType", "OPTION_TYPE", "optiontype", "OPTION_TYP",
    "OptnTp", "OPTTYPE", "type", "Type", "TYPE", "Option Type"
]
INSTRUMENT_ALIASES = [
    "instrument", "Instrument", "INSTRUMENT", "instrument_type", "InstrumentType",
    "INSTRUMENT_TYPE", "FinInstrmTp", "FinInstrmTpNm", "security_type", "SecurityType"
]
PRICE_ALIASES = [
    "close", "Close", "CLOSE", "ltp", "LTP", "last_price", "LastPrice", "last",
    "Last", "price", "Price", "ClsPric", "CLSPRIC", "settlement", "SttlmPric"
]
OPEN_ALIASES = ["open", "Open", "OPEN", "Open Price", "OpnPric", "OPENPRICE"]
HIGH_ALIASES = ["high", "High", "HIGH", "High Price", "HghPric", "HIGHPRICE"]
LOW_ALIASES = ["low", "Low", "LOW", "Low Price", "LwPric", "LOWPRICE"]
VOLUME_ALIASES = ["volume", "Volume", "VOLUME", "vol", "Vol", "Total Volume", "TtlTradgVol", "TOTTRDQTY"]
OI_ALIASES = ["oi", "OI", "open_interest", "OpenInterest", "OPEN_INT", "Open Interest", "OpnIntrst", "OPENINT"]
UNDERLYING_ALIASES = [
    "underlying_close", "UnderlyingClose", "UNDERLYING_CLOSE", "underlying_price",
    "Underlying Price", "underlying spot", "UnderlyingSpot", "UNDERLYING_SPOT",
    "spot", "Spot", "SPOT", "spot_price", "SpotPrice", "SPOT_PRICE",
    "index_close", "IndexClose", "INDEX_CLOSE", "index_price", "IndexPrice",
    "INDEX_PRICE", "NIFTY_CLOSE", "nifty_close", "nifty_spot", "NIFTY_SPOT",
    "underlying value", "Underlying Value", "UNDERLYING_VALUE", "UnderlyingValue",
    "index value", "Index Value", "IndexValue", "Nifty Index", "NIFTY Index",
    "NIFTY_INDEX", "Nifty Spot", "NIFTY Spot", "NIFTY50_CLOSE"
]


def _find_col(columns: Iterable[str], aliases: list[str]) -> str | None:
    by_lower = {str(c).strip().lower(): c for c in columns}
    for alias in aliases:
        if alias.lower() in by_lower:
            return by_lower[alias.lower()]
    return None


def _candidate_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".csv", ".txt", ".parquet", ".gz"}:
            continue
        name = str(p).lower()
        if p.name.startswith(".") or "banknifty" in name or "bank_nifty" in name:
            continue
        out.append(p)
    return sorted(out, key=lambda p: str(p).lower())


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
        yield from _read_csv_chunks(path, chunksize)


def _normalise_symbol(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper().str.replace(r"\s+", "", regex=True)


def _extract_underlying_proxy_chunk(df: pd.DataFrame, filename: str = "") -> pd.DataFrame | None:
    """Extract one NIFTY underlying value per raw record where possible."""
    if df.empty:
        return None
    date_col = _find_col(df.columns, DATE_ALIASES)
    if not date_col:
        return None

    price_col = _find_col(df.columns, PRICE_ALIASES)
    direct_col = _find_col(df.columns, UNDERLYING_ALIASES)
    symbol_col = _find_col(df.columns, SYMBOL_ALIASES)
    instrument_col = _find_col(df.columns, INSTRUMENT_ALIASES)
    opt_col = _find_col(df.columns, OPT_TYPE_ALIASES)
    strike_col = _find_col(df.columns, STRIKE_ALIASES)

    # Direct underlying/spot fields are valid even when the row itself is an option.
    if direct_col:
        px = pd.to_numeric(df[direct_col], errors="coerce")
        direct = pd.DataFrame({
            "date": pd.to_datetime(df[date_col], errors="coerce").dt.normalize(),
            "underlying_close": px,
        }, index=df.index)
        if symbol_col:
            direct = direct.loc[_normalise_symbol(df[symbol_col]).str.contains("NIFTY", na=False)]
        elif "nifty" not in filename.lower().replace("_", ""):
            # Without a symbol, only trust a file whose name identifies NIFTY.
            direct = direct.iloc[0:0]
        direct = direct.dropna(subset=["date", "underlying_close"])
        direct = direct.loc[direct["underlying_close"] > 0]
        if not direct.empty:
            direct["priority"] = 0
            direct["days_to_expiry"] = pd.NA
            return direct[["date", "underlying_close", "priority", "days_to_expiry"]]

    if price_col is None:
        return None

    raw_symbol = _normalise_symbol(df[symbol_col]) if symbol_col else None
    if raw_symbol is not None:
        nifty_mask = raw_symbol.str.contains("NIFTY", na=False)
    else:
        nifty_mask = pd.Series("nifty" in filename.lower().replace("_", ""), index=df.index)

    px = pd.to_numeric(df[price_col], errors="coerce")
    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce").dt.normalize(),
        "underlying_close": px,
    }, index=df.index)
    out = out.loc[nifty_mask]
    if out.empty:
        return None

    option_mask = pd.Series(False, index=df.index)
    if opt_col:
        option_mask = df[opt_col].astype(str).str.strip().str.upper().isin(["CE", "PE", "CALL", "PUT", "C", "P"])
    has_strike = strike_col is not None

    if instrument_col:
        inst = df[instrument_col].astype(str).str.upper()
        spot_mask = inst.str.contains("SPOT|INDEX|IDX|EQUITY", regex=True, na=False)
        fut_mask = inst.str.contains("FUT", regex=True, na=False)
        keep = spot_mask | fut_mask
    else:
        # If this looks like an option contract, do not use its option premium as spot.
        keep = ~option_mask if has_strike else pd.Series(True, index=df.index)

    out = out.loc[keep].dropna(subset=["date", "underlying_close"])
    out = out.loc[out["underlying_close"] > 0]
    if out.empty:
        return None

    priority = pd.Series(3, index=df.index, dtype="int8")
    if instrument_col:
        inst = df[instrument_col].astype(str).str.upper()
        priority.loc[inst.str.contains("SPOT|INDEX|IDX|EQUITY", regex=True, na=False)] = 1
        priority.loc[inst.str.contains("FUT", regex=True, na=False)] = 2

    expiry_col = _find_col(df.columns, EXPIRY_ALIASES)
    if expiry_col:
        expiry = pd.to_datetime(df[expiry_col], errors="coerce").dt.normalize()
        days = (expiry - out["date"]).dt.days
    else:
        days = pd.Series(pd.NA, index=df.index)

    out["priority"] = priority.loc[out.index]
    out["days_to_expiry"] = days.loc[out.index]
    return out[["date", "underlying_close", "priority", "days_to_expiry"]]


def _collapse_proxy(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=["date", "underlying_close"])
    proxy = pd.concat(parts, ignore_index=True)
    proxy["days_to_expiry"] = pd.to_numeric(proxy["days_to_expiry"], errors="coerce")
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
    for name, aliases in [
        ("open", OPEN_ALIASES), ("high", HIGH_ALIASES), ("low", LOW_ALIASES),
        ("volume", VOLUME_ALIASES), ("open_interest", OI_ALIASES),
        ("underlying_close", UNDERLYING_ALIASES)
    ]:
        col = _find_col(df.columns, aliases)
        out[name] = pd.to_numeric(df[col], errors="coerce") if col else pd.NA

    out["symbol"] = _normalise_symbol(df[symbol_col]) if symbol_col else "NIFTY"
    out = out.loc[out["symbol"].str.contains("NIFTY", regex=False, na=False)]
    out = out.loc[out["option_type"].isin(["CE", "PE"])]
    out = out.dropna(subset=["date", "expiry", "strike", "close"])
    out = out.loc[out["close"] > 0]
    if out.empty:
        return None
    return out[["date", "expiry", "strike", "option_type", "open", "high", "low", "close", "volume", "open_interest", "underlying_close", "symbol"]]


def _load_underlying_proxy(root: Path, chunksize: int = 200_000) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in _candidate_files(root):
        try:
            for chunk in _iter_tables(path, chunksize):
                p = _extract_underlying_proxy_chunk(chunk, path.name)
                if p is not None and not p.empty:
                    parts.append(p)
        except Exception as exc:
            print(f"Underlying proxy skipped: {path.name}: {type(exc).__name__}: {exc}", flush=True)
    return _collapse_proxy(parts)


def load_from_kaggle_roots(roots: list[str]) -> pd.DataFrame:
    work_root = Path(tempfile.mkdtemp(prefix="kaggle-nifty-ingest-"))
    option_dir = work_root / "options"
    proxy_dir = work_root / "proxy"
    option_dir.mkdir(parents=True, exist_ok=True)
    proxy_dir.mkdir(parents=True, exist_ok=True)
    try:
        option_parts = 0
        proxy_parts = 0
        for root_str in roots:
            root = Path(root_str)
            if not root.exists():
                continue
            for path in _candidate_files(root):
                try:
                    for chunk_no, chunk in enumerate(_iter_tables(path), start=1):
                        p = _extract_underlying_proxy_chunk(chunk, path.name)
                        if p is not None and not p.empty:
                            p.to_parquet(proxy_dir / f"part-{proxy_parts:06d}.parquet", index=False)
                            proxy_parts += 1
                        normalized = _normalize_option_chunk(chunk)
                        if normalized is None or normalized.empty:
                            continue
                        normalized.to_parquet(option_dir / f"part-{option_parts:06d}.parquet", index=False)
                        option_parts += 1
                        if option_parts % 10 == 0:
                            print(f"Kaggle ingest progress: parts={option_parts}, source={path.name}, chunk={chunk_no}", flush=True)
                except Exception as exc:
                    print(f"Skipping {path}: {type(exc).__name__}: {exc}", flush=True)

        parts = sorted(option_dir.glob("part-*.parquet"))
        if not parts:
            raise RuntimeError("No usable NIFTY option rows found in Kaggle datasets")

        for i, part in enumerate(parts, start=1):
            x = pd.read_parquet(part).sort_values(["date", "expiry", "strike", "option_type"])
            x = x.drop_duplicates(["date", "expiry", "strike", "option_type"], keep="last")
            x.to_parquet(option_dir / f"reduced-{i:06d}.parquet", index=False)
            if i % 25 == 0:
                print(f"Kaggle reduce progress: {i}/{len(parts)}", flush=True)

        reduced = sorted(option_dir.glob("reduced-*.parquet"))
        options = pd.concat([pd.read_parquet(p) for p in reduced], ignore_index=True)
        options = options.sort_values(["date", "expiry", "strike", "option_type"])
        options = options.drop_duplicates(["date", "expiry", "strike", "option_type"], keep="last")

        proxy_parts_list = [pd.read_parquet(p) for p in sorted(proxy_dir.glob("part-*.parquet"))]
        proxy = _collapse_proxy(proxy_parts_list)
        if proxy.empty:
            for root_str in roots:
                proxy = _load_underlying_proxy(Path(root_str))
                if not proxy.empty:
                    break

        supplied = options[["date", "underlying_close"]].dropna().groupby("date", as_index=False)["underlying_close"].last()
        if not supplied.empty:
            supplied = supplied.rename(columns={"underlying_close": "supplied_underlying"})
            options = options.drop(columns=["underlying_close"]).merge(supplied, on="date", how="left", validate="many_to_one")
        else:
            options = options.drop(columns=["underlying_close"])
            options["supplied_underlying"] = pd.NA

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
        shutil.rmtree(work_root, ignore_errors=True)
