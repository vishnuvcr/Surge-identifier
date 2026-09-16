from __future__ import annotations

import gzip
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

DATE_ALIASES = ["date", "Date", "DATE", "timestamp", "Timestamp", "TIMESTAMP", "TRADE_DATE", "trade_date", "DateTime", "datetime", "TradDt"]
SYMBOL_ALIASES = ["symbol", "Symbol", "SYMBOL", "TckrSymb", "TICKER", "underlying", "Underlying", "UNDERLYING", "Ticker", "ticker"]
EXPIRY_ALIASES = ["expiry", "Expiry", "EXPIRY", "expiry_date", "ExpiryDate", "EXPIRY_DT", "XpryDt", "Expiry Date", "expirydate"]
STRIKE_ALIASES = ["strike", "Strike", "STRIKE", "strike_price", "StrikePrice", "STRIKE_PR", "StrkPric", "Strike Price"]
OPT_TYPE_ALIASES = ["option_type", "OptionType", "OPTION_TYPE", "optiontype", "OPTION_TYP", "OptnTp", "OPTTYPE", "type", "Type", "TYPE", "Option Type"]
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
    lookup = {str(c).strip().lower(): c for c in columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    return None


def _candidate_files(root: Path) -> list[Path]:
    out = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".csv", ".txt", ".parquet", ".gz"}:
            name = str(p).lower()
            if not p.name.startswith(".") and "banknifty" not in name and "bank_nifty" not in name:
                out.append(p)
    return sorted(out, key=lambda x: str(x).lower())


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
    out = out.loc[out["symbol"].str.contains("NIFTY", regex=False, na=False) & out["option_type"].isin(["CE", "PE"])]
    out = out.dropna(subset=["date", "expiry", "strike", "close"])
    out = out.loc[out["close"] > 0]
    if out.empty:
        return None
    return out[["date", "expiry", "strike", "option_type", "open", "high", "low", "close", "volume", "open_interest", "underlying_close", "symbol"]]


def _embedded_underlying(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "underlying_close"])
    date_col = _find_col(df.columns, DATE_ALIASES)
    direct_col = _find_col(df.columns, UNDERLYING_ALIASES)
    if not date_col or not direct_col:
        return pd.DataFrame(columns=["date", "underlying_close"])
    x = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce").dt.normalize(),
        "underlying_close": pd.to_numeric(df[direct_col], errors="coerce"),
    })
    x = x.dropna(subset=["date", "underlying_close"])
    x = x.loc[x["underlying_close"] > 0]
    return x.groupby("date", as_index=False)["underlying_close"].last()


def _fetch_yahoo_nifty(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Independent daily NIFTY 50 close series used when option data has no spot field."""
    if pd.isna(start) or pd.isna(end):
        return pd.DataFrame(columns=["date", "underlying_close"])
    period1 = int((start - pd.Timedelta(days=2)).timestamp())
    period2 = int((end + pd.Timedelta(days=3)).timestamp())
    url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI"
    params = {"period1": period1, "period2": period2, "interval": "1d", "events": "history", "includeAdjustedClose": "true"}
    try:
        r = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        r.raise_for_status()
        payload = r.json()
        result = payload["chart"]["result"][0]
        ts = result.get("timestamp", [])
        closes = result["indicators"]["quote"][0].get("close", [])
        if not ts or not closes:
            return pd.DataFrame(columns=["date", "underlying_close"])
        x = pd.DataFrame({"date": pd.to_datetime(ts, unit="s", utc=True).tz_convert("Asia/Kolkata").tz_localize(None).normalize(), "underlying_close": pd.to_numeric(closes, errors="coerce")})
        x = x.dropna(subset=["date", "underlying_close"])
        x = x.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
        print(f"Yahoo NIFTY proxy: rows={len(x)}, dates={x['date'].min()} to {x['date'].max()}", flush=True)
        return x
    except Exception as exc:
        print(f"Yahoo NIFTY proxy failed: {type(exc).__name__}: {exc}", flush=True)
        return pd.DataFrame(columns=["date", "underlying_close"])


def load_from_kaggle_roots(roots: list[str]) -> pd.DataFrame:
    work_root = Path(tempfile.mkdtemp(prefix="kaggle-nifty-ingest-"))
    part_dir = work_root / "options"
    part_dir.mkdir(parents=True, exist_ok=True)
    embedded_proxy: list[pd.DataFrame] = []
    try:
        part_count = 0
        for root_str in roots:
            root = Path(root_str)
            if not root.exists():
                continue
            for path in _candidate_files(root):
                try:
                    for chunk_no, chunk in enumerate(_iter_tables(path), start=1):
                        proxy = _embedded_underlying(chunk)
                        if not proxy.empty:
                            embedded_proxy.append(proxy)
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
            x = pd.read_parquet(part).sort_values(["date", "expiry", "strike", "option_type"])
            x = x.drop_duplicates(["date", "expiry", "strike", "option_type"], keep="last")
            x.to_parquet(part_dir / f"reduced-{i:06d}.parquet", index=False)
            if i % 25 == 0:
                print(f"Kaggle reduce progress: {i}/{len(parts)}", flush=True)

        reduced = sorted(part_dir.glob("reduced-*.parquet"))
        options = pd.concat([pd.read_parquet(p) for p in reduced], ignore_index=True)
        options = options.sort_values(["date", "expiry", "strike", "option_type"])
        options = options.drop_duplicates(["date", "expiry", "strike", "option_type"], keep="last")

        embedded = pd.concat(embedded_proxy, ignore_index=True) if embedded_proxy else pd.DataFrame(columns=["date", "underlying_close"])
        if not embedded.empty:
            embedded = embedded.drop_duplicates("date", keep="last")

        yahoo = _fetch_yahoo_nifty(options["date"].min(), options["date"].max())
        if not yahoo.empty:
            yahoo = yahoo.rename(columns={"underlying_close": "yahoo_underlying"})
        else:
            yahoo = pd.DataFrame(columns=["date", "yahoo_underlying"])

        supplied = options[["date", "underlying_close"]].dropna().groupby("date", as_index=False)["underlying_close"].last()
        supplied = supplied.rename(columns={"underlying_close": "supplied_underlying"})
        embedded = embedded.rename(columns={"underlying_close": "embedded_underlying"})

        options = options.drop(columns=["underlying_close"]).merge(supplied, on="date", how="left", validate="many_to_one")
        options = options.merge(embedded, on="date", how="left", validate="many_to_one")
        options = options.merge(yahoo, on="date", how="left", validate="many_to_one")
        options["underlying_close"] = options["supplied_underlying"].fillna(options["embedded_underlying"]).fillna(options["yahoo_underlying"])
        options = options.drop(columns=["supplied_underlying", "embedded_underlying", "yahoo_underlying"])

        options["option_type"] = options["option_type"].astype("category")
        for col in ["strike", "open", "high", "low", "close", "volume", "open_interest", "underlying_close"]:
            options[col] = pd.to_numeric(options[col], errors="coerce", downcast="float")
        return options.sort_values(["date", "expiry", "strike", "option_type"]).reset_index(drop=True)
    finally:
        import shutil
        shutil.rmtree(work_root, ignore_errors=True)
