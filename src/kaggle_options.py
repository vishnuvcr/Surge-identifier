from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DATE_ALIASES = ["date", "trade_date", "tradedate", "trad_dt", "timestamp", "datetime", "time"]
SYMBOL_ALIASES = ["symbol", "ticker", "underlying", "underlying_symbol", "instrument_name"]
EXPIRY_ALIASES = ["expiry", "expiry_date", "expirydate", "exp_date", "expiration", "expiration_date"]
STRIKE_ALIASES = ["strike", "strike_price", "strikeprice", "strike_pr"]
TYPE_ALIASES = ["option_type", "optiontype", "opt_type", "opttype", "type", "cp", "call_put", "option_typ", "ce_pe"]
PRICE_ALIASES = ["close", "close_price", "closeprice", "clspic", "ltp", "last", "last_price", "price", "settle", "settlement"]
OPEN_ALIASES = ["open", "open_price", "openprice"]
HIGH_ALIASES = ["high", "high_price", "highprice"]
LOW_ALIASES = ["low", "low_price", "lowprice"]
VOLUME_ALIASES = ["volume", "contracts", "total_volume", "total_traded_volume", "traded_volume", "tottrdqty", "qty"]
OI_ALIASES = ["oi", "open_interest", "openinterest", "open_int"]
UNDERLYING_ALIASES = ["underlying_close", "underlying_price", "underlying_value", "spot", "spot_price", "spotprice", "nifty_spot", "nifty_close"]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _pick(df: pd.DataFrame, aliases: list[str]):
    by_norm = {_norm(c): c for c in df.columns}
    for alias in aliases:
        c = by_norm.get(_norm(alias))
        if c is not None:
            return df[c]
    return None


def _date_from_path(path: Path):
    m = re.search(r"(?<!\d)(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)(?!\d)", str(path))
    if not m:
        return None
    try:
        return pd.Timestamp(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
    except Exception:
        return None


def _read_table(path: Path, chunksize: int = 250_000):
    suf = path.suffix.lower()
    if suf == ".parquet":
        return [pd.read_parquet(path)]
    if suf in {".csv", ".txt"}:
        return pd.read_csv(path, low_memory=False, chunksize=chunksize)
    if suf == ".gz" and path.name.lower().endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip", low_memory=False, chunksize=chunksize)
    return []


def _candidate_files(roots: list[Path]):
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in {".csv", ".txt", ".parquet", ".gz"}:
                continue
            s = str(p).lower()
            if "banknifty" in s or "bank_nifty" in s or "nifty" not in s:
                continue
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                yield p


def _normalize_option_chunk(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df.columns = [str(c).strip() for c in df.columns]
    d = _pick(df, DATE_ALIASES)
    if d is None:
        inferred = _date_from_path(path)
        if inferred is None:
            return pd.DataFrame()
        d = pd.Series(inferred, index=df.index)
    else:
        raw_d = d
        d = pd.to_datetime(raw_d, errors="coerce")
        if d.isna().mean() > 0.5:
            d = pd.to_datetime(raw_d.astype("string"), format="%Y%m%d", errors="coerce")
    d = pd.to_datetime(d, errors="coerce").dt.normalize()

    symbol = _pick(df, SYMBOL_ALIASES)
    symbol = symbol.astype(str).str.strip().str.upper() if symbol is not None else pd.Series("NIFTY", index=df.index)
    expiry = _pick(df, EXPIRY_ALIASES)
    strike = _pick(df, STRIKE_ALIASES)
    opt_type = _pick(df, TYPE_ALIASES)
    price = _pick(df, PRICE_ALIASES)
    if expiry is None or strike is None or opt_type is None or price is None:
        return pd.DataFrame()

    et = pd.to_datetime(expiry, errors="coerce")
    if et.isna().mean() > 0.5:
        et = pd.to_datetime(expiry.astype("string"), format="%Y%m%d", errors="coerce")
    et = et.dt.normalize()
    ot = opt_type.astype(str).str.strip().str.upper().replace({"CALL": "CE", "PUT": "PE", "C": "CE", "P": "PE"})
    k = pd.to_numeric(strike, errors="coerce")
    close = pd.to_numeric(price, errors="coerce")
    valid = d.notna() & et.notna() & k.notna() & close.notna() & ot.isin(["CE", "PE"]) & symbol.eq("NIFTY")
    if not valid.any():
        return pd.DataFrame()

    def numeric(aliases, default):
        x = _pick(df, aliases)
        return pd.to_numeric(x, errors="coerce") if x is not None else pd.Series(default, index=df.index, dtype=float)

    out = pd.DataFrame({
        "date": d[valid], "symbol": symbol[valid], "instrument": "OPTIDX", "option_type": ot[valid],
        "expiry": et[valid], "strike": k[valid], "open": numeric(OPEN_ALIASES, float("nan"))[valid],
        "high": numeric(HIGH_ALIASES, float("nan"))[valid], "low": numeric(LOW_ALIASES, float("nan"))[valid],
        "close": close[valid], "settlement": close[valid], "volume": numeric(VOLUME_ALIASES, 0)[valid].fillna(0),
        "oi": numeric(OI_ALIASES, 0)[valid].fillna(0),
    })
    for col in ["open", "high", "low"]:
        out[col] = out[col].fillna(out["close"])
    u = _pick(df, UNDERLYING_ALIASES)
    if u is not None:
        out["underlying_close"] = pd.to_numeric(u, errors="coerce")[valid].values
    return out


def load_from_kaggle_roots(roots: list[str | Path]) -> pd.DataFrame:
    roots = [Path(x) for x in roots]
    option_parts: list[pd.DataFrame] = []
    underlying_parts: list[pd.DataFrame] = []
    inspected = 0

    for path in _candidate_files(roots):
        inspected += 1
        try:
            chunks = _read_table(path)
            for chunk_no, df in enumerate(chunks, 1):
                part = _normalize_option_chunk(df, path)
                if not part.empty:
                    option_parts.append(part)
                if chunk_no % 10 == 0:
                    print(f"Kaggle ingest progress: file={path.name}, chunk={chunk_no}, option_parts={len(option_parts)}", flush=True)
        except Exception as exc:
            print(f"Skipping {path}: {exc}", flush=True)

        # Also detect separate spot/futures tables when they are explicitly named.
        try:
            if any(token in str(path).lower() for token in ("spot", "future", "futures")):
                for df in _read_table(path):
                    price = _pick(df, PRICE_ALIASES)
                    strike = _pick(df, STRIKE_ALIASES)
                    d = _pick(df, DATE_ALIASES)
                    if price is None or strike is not None or d is None:
                        continue
                    symbol = _pick(df, SYMBOL_ALIASES)
                    symbol = symbol.astype(str).str.upper().str.strip() if symbol is not None else pd.Series("NIFTY", index=df.index)
                    dd = pd.to_datetime(d, errors="coerce").dt.normalize()
                    u = pd.DataFrame({"date": dd, "close": pd.to_numeric(price, errors="coerce"), "symbol": symbol})
                    u = u.dropna(subset=["date", "close"])
                    if not u.empty:
                        u["priority"] = 2 if "future" in str(path).lower() else 1
                        underlying_parts.append(u)
        except Exception:
            pass

    if not option_parts:
        return pd.DataFrame()

    opt = pd.concat(option_parts, ignore_index=True)
    opt["date"] = pd.to_datetime(opt["date"], errors="coerce").dt.normalize()
    opt["expiry"] = pd.to_datetime(opt["expiry"], errors="coerce").dt.normalize()
    opt = opt[opt["symbol"].eq("NIFTY")].copy()
    if "underlying_close" not in opt.columns:
        opt["underlying_close"] = float("nan")
    else:
        opt["underlying_close"] = pd.to_numeric(opt["underlying_close"], errors="coerce")

    if underlying_parts:
        under = pd.concat(underlying_parts, ignore_index=True)
        under = under[under.symbol.eq("NIFTY")].sort_values(["date", "priority"]).drop_duplicates("date", keep="first")
        under = under[["date", "close"]].rename(columns={"close": "underlying_close_proxy"})
        opt = opt.merge(under, on="date", how="left", validate="many_to_one")
        opt["underlying_close"] = opt["underlying_close"].fillna(opt["underlying_close_proxy"])
        opt = opt.drop(columns=["underlying_close_proxy"])

    opt = opt.dropna(subset=["date", "expiry", "strike", "close"])
    opt = opt.drop_duplicates(subset=["date", "expiry", "strike", "option_type"], keep="last")
    opt = opt.sort_values(["date", "expiry", "strike", "option_type"]).reset_index(drop=True)
    print(f"Kaggle option tables inspected={inspected}, rows={len(opt):,}, dates={opt.date.min()} to {opt.date.max()}", flush=True)
    print(f"Unique expiries={opt.expiry.nunique():,}, contracts={opt[['expiry','strike','option_type']].drop_duplicates().shape[0]:,}", flush=True)
    return opt
