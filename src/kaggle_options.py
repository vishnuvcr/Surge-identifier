from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd


DATE_ALIASES = ["date", "trade_date", "tradedate", "trad_dt", "timestamp", "datetime", "time"]
SYMBOL_ALIASES = ["symbol", "ticker", "underlying", "underlying_symbol", "instrument_name"]
EXPIRY_ALIASES = ["expiry", "expiry_date", "expirydate", "exp_date", "expiration", "expiration_date"]
STRIKE_ALIASES = ["strike", "strike_price", "strikeprice", "strike_pr", "strikeprice"]
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


def _read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf in {".csv", ".txt"}:
        return pd.read_csv(path, low_memory=False)
    if suf == ".gz" and path.name.lower().endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip", low_memory=False)
    return pd.DataFrame()


def _candidate_files(roots: list[Path]):
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".csv", ".txt", ".parquet", ".gz"}:
                continue
            s = str(p).lower()
            # Ignore BankNifty-only paths and unrelated documentation files.
            if "banknifty" in s or "bank_nifty" in s:
                continue
            if "nifty" not in s:
                continue
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                yield p


def load_from_kaggle_roots(roots: list[str | Path]) -> pd.DataFrame:
    roots = [Path(x) for x in roots]
    option_frames: list[pd.DataFrame] = []
    underlying_frames: list[pd.DataFrame] = []
    inspected = 0

    for path in _candidate_files(roots):
        inspected += 1
        try:
            df = _read_table(path)
        except Exception:
            continue
        if df.empty:
            continue
        df.columns = [str(c).strip() for c in df.columns]

        d = _pick(df, DATE_ALIASES)
        if d is None:
            inferred = _date_from_path(path)
            if inferred is None:
                continue
            d = pd.Series(inferred, index=df.index)
        else:
            d = pd.to_datetime(d, errors="coerce", dayfirst=False)
            # Some datasets encode dates as YYYYMMDD integers.
            if d.isna().mean() > 0.5:
                d = pd.to_datetime(d.astype("string"), format="%Y%m%d", errors="coerce")
        d = pd.to_datetime(d, errors="coerce").dt.normalize()

        symbol = _pick(df, SYMBOL_ALIASES)
        symbol = symbol.astype(str).str.strip().str.upper() if symbol is not None else pd.Series("NIFTY", index=df.index)

        expiry = _pick(df, EXPIRY_ALIASES)
        strike = _pick(df, STRIKE_ALIASES)
        opt_type = _pick(df, TYPE_ALIASES)
        price = _pick(df, PRICE_ALIASES)
        close = pd.to_numeric(price, errors="coerce") if price is not None else pd.Series(float("nan"), index=df.index)

        # Option-chain table detection.
        if expiry is not None and strike is not None and opt_type is not None and price is not None:
            et = pd.to_datetime(expiry, errors="coerce", dayfirst=False).dt.normalize()
            if et.isna().mean() > 0.5:
                et = pd.to_datetime(expiry.astype("string"), format="%Y%m%d", errors="coerce")
            ot = opt_type.astype(str).str.strip().str.upper().replace({"CALL": "CE", "PUT": "PE", "C": "CE", "P": "PE"})
            k = pd.to_numeric(strike, errors="coerce")
            valid = d.notna() & et.notna() & k.notna() & close.notna() & ot.isin(["CE", "PE"])
            if valid.any():
                out = pd.DataFrame({
                    "date": d[valid],
                    "symbol": symbol[valid],
                    "instrument": "OPTIDX",
                    "option_type": ot[valid],
                    "expiry": et[valid],
                    "strike": k[valid],
                    "open": pd.to_numeric(_pick(df, OPEN_ALIASES), errors="coerce")[valid] if _pick(df, OPEN_ALIASES) is not None else close[valid],
                    "high": pd.to_numeric(_pick(df, HIGH_ALIASES), errors="coerce")[valid] if _pick(df, HIGH_ALIASES) is not None else close[valid],
                    "low": pd.to_numeric(_pick(df, LOW_ALIASES), errors="coerce")[valid] if _pick(df, LOW_ALIASES) is not None else close[valid],
                    "close": close[valid],
                    "settlement": close[valid],
                    "volume": pd.to_numeric(_pick(df, VOLUME_ALIASES), errors="coerce")[valid].fillna(0) if _pick(df, VOLUME_ALIASES) is not None else 0.0,
                    "oi": pd.to_numeric(_pick(df, OI_ALIASES), errors="coerce")[valid].fillna(0) if _pick(df, OI_ALIASES) is not None else 0.0,
                })
                u = _pick(df, UNDERLYING_ALIASES)
                if u is not None:
                    out["underlying_close"] = pd.to_numeric(u, errors="coerce")[valid].values
                option_frames.append(out)
                continue

        # Spot/futures table detection for an underlying proxy.
        if price is not None and strike is None:
            path_text = str(path).lower()
            if any(token in path_text for token in ("spot", "future", "futures")):
                u = pd.DataFrame({"date": d, "close": close, "symbol": symbol})
                if "future" in path_text:
                    u["priority"] = 2
                else:
                    u["priority"] = 1
                u = u.dropna(subset=["date", "close"])
                if not u.empty:
                    underlying_frames.append(u)

    if not option_frames:
        return pd.DataFrame()

    opt = pd.concat(option_frames, ignore_index=True)
    opt["date"] = pd.to_datetime(opt["date"], errors="coerce").dt.normalize()
    opt["expiry"] = pd.to_datetime(opt["expiry"], errors="coerce").dt.normalize()
    opt = opt[opt["symbol"].eq("NIFTY")].copy()

    if "underlying_close" not in opt.columns:
        opt["underlying_close"] = float("nan")
    else:
        opt["underlying_close"] = pd.to_numeric(opt["underlying_close"], errors="coerce")

    if underlying_frames:
        under = pd.concat(underlying_frames, ignore_index=True)
        under["symbol"] = under["symbol"].astype(str).str.upper().str.strip()
        under = under[under.symbol.eq("NIFTY")].copy()
        # Spot gets priority over futures. Deduplicate to one value per day.
        under = under.sort_values(["date", "priority"]).drop_duplicates("date", keep="first")
        under = under[["date", "close"]].rename(columns={"close": "underlying_close_proxy"})
        opt = opt.merge(under, on="date", how="left", validate="many_to_one")
        opt["underlying_close"] = opt["underlying_close"].fillna(opt["underlying_close_proxy"])
        opt = opt.drop(columns=["underlying_close_proxy"])

    opt = opt.dropna(subset=["date", "expiry", "strike", "close"])
    opt = opt.drop_duplicates(subset=["date", "expiry", "strike", "option_type"], keep="last")
    opt = opt.sort_values(["date", "expiry", "strike", "option_type"]).reset_index(drop=True)
    print(f"Kaggle option tables inspected={inspected}, rows={len(opt):,}, dates={opt.date.min()} to {opt.date.max()}")
    print(f"Unique expiries={opt.expiry.nunique():,}, contracts={opt[['expiry','strike','option_type']].drop_duplicates().shape[0]:,}")
    return opt
