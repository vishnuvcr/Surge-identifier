from __future__ import annotations

import io
import os
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

IST = timezone(timedelta(hours=5, minutes=30))
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/134 Safari/537.36"


def _pick(raw: pd.DataFrame, *names):
    for n in names:
        if n in raw.columns:
            return raw[n]
    return pd.Series(index=raw.index, dtype=float)


def _parse(content: bytes, dt: date | None = None) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        with zf.open(zf.namelist()[0]) as f:
            raw = pd.read_csv(f)
    raw.columns = [str(c).strip() for c in raw.columns]
    if dt is not None:
        parsed_date = pd.Series(pd.Timestamp(dt).normalize(), index=raw.index)
    else:
        trade_date = _pick(raw, "TradDt", "TIMESTAMP", "Trade_Date")
        parsed_date = pd.to_datetime(trade_date, errors="coerce").dt.normalize()
    return pd.DataFrame({
        "date": parsed_date,
        "symbol": _pick(raw, "TckrSymb", "SYMBOL").astype(str).str.strip().str.upper(),
        "instrument": _pick(raw, "FinInstrmTp", "INSTRUMENT", "INSTRUMENT_TYPE").astype(str).str.strip().str.upper(),
        "option_type": _pick(raw, "OptnTp", "OPTTYPE", "OPTION_TYP").astype(str).str.strip().str.upper(),
        "expiry": pd.to_datetime(_pick(raw, "XpryDt", "EXPIRYDT", "EXPIRY_DT"), errors="coerce").dt.normalize(),
        "strike": pd.to_numeric(_pick(raw, "StrkPric", "STRIKEPRICE", "STRIKE_PR"), errors="coerce"),
        "open": pd.to_numeric(_pick(raw, "OpnPric", "OPENPRICE", "OPEN"), errors="coerce"),
        "high": pd.to_numeric(_pick(raw, "HghPric", "HIGHPRICE", "HIGH"), errors="coerce"),
        "low": pd.to_numeric(_pick(raw, "LwPric", "LOWPRICE", "LOW"), errors="coerce"),
        "close": pd.to_numeric(_pick(raw, "ClsPric", "CLSPRIC", "CLOSEPRICE", "CLOSE"), errors="coerce"),
        "settlement": pd.to_numeric(_pick(raw, "SttlmPric", "SETTLEPRICE", "SETTLE_PR"), errors="coerce"),
        "volume": pd.to_numeric(_pick(raw, "TtlTradgVol", "CONTRACTS", "TOTTRDQTY"), errors="coerce").fillna(0),
        "oi": pd.to_numeric(_pick(raw, "OpnIntrst", "OPENINT", "OPEN_INT"), errors="coerce").fillna(0),
    })


def _download(dt: date, raw_dir: Path):
    raw_dir.mkdir(parents=True, exist_ok=True)
    fp = raw_dir / f"{dt:%Y%m%d}.zip"
    try:
        if fp.exists():
            return _parse(fp.read_bytes(), dt)
        for host in ("https://archives.nseindia.com", "https://nsearchives.nseindia.com"):
            url = f"{host}/content/fo/BhavCopy_NSE_FO_0_0_0_{dt:%Y%m%d}_F_0000.csv.zip"
            r = requests.get(url, headers={"User-Agent": UA}, timeout=40)
            if r.ok and r.content[:2] == b"PK":
                fp.write_bytes(r.content)
                return _parse(r.content, dt)
    except Exception:
        pass
    time.sleep(0.4)
    return None


def _parse_external_archives(archive_root: Path, start: date, end: date) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    files = list(archive_root.glob("data/2025/**/*.zip")) + list(archive_root.glob("data/2026/**/*.zip"))
    for fp in sorted(files):
        m = re.search(r"(\d{8})", fp.name)
        if not m:
            continue
        try:
            dt = datetime.strptime(m.group(1), "%Y%m%d").date()
        except ValueError:
            continue
        if not (start <= dt <= end):
            continue
        try:
            frames.append(_parse(fp.read_bytes(), dt))
        except Exception:
            continue
    return frames


def _finalize(raw: pd.DataFrame, root: Path, start: date, end: date) -> pd.DataFrame:
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    raw["expiry"] = pd.to_datetime(raw["expiry"], errors="coerce").dt.normalize()
    raw["symbol"] = raw["symbol"].astype(str).str.strip().str.upper()
    raw["instrument"] = raw["instrument"].astype(str).str.strip().str.upper()
    raw["option_type"] = raw["option_type"].astype(str).str.strip().str.upper().replace({"CALL": "CE", "PUT": "PE"})
    raw = raw.dropna(subset=["date", "expiry", "strike", "close"])
    raw = raw[(raw.date.dt.date >= start) & (raw.date.dt.date <= end)]

    nifty = raw[raw.symbol == "NIFTY"].copy()
    if "underlying_close" not in nifty.columns:
        nifty["underlying_close"] = float("nan")
    nifty["underlying_close"] = pd.to_numeric(nifty["underlying_close"], errors="coerce")

    # NSE archive rows need a futures-derived underlying proxy; Kaggle option-chain
    # rows can already carry a spot/underlying price, which is retained.
    fut = nifty[nifty.instrument.str.contains("FUT", na=False) & nifty.close.notna()].copy()
    if not fut.empty:
        fut["days_to_expiry"] = (fut.expiry - fut.date).dt.days
        fut = fut[fut.days_to_expiry >= 0].sort_values(["date", "days_to_expiry"])
        fut_daily = fut.groupby("date", as_index=False).first()[["date", "close"]].rename(columns={"close": "futures_underlying"})
        nifty = nifty.merge(fut_daily, on="date", how="left", validate="many_to_one")
        nifty["underlying_close"] = nifty["underlying_close"].fillna(nifty["futures_underlying"])
        nifty = nifty.drop(columns=["futures_underlying"])

    opt = nifty[nifty.instrument.str.contains("OPT", na=False)].copy()
    opt = opt[opt.option_type.isin(["CE", "PE"])]
    opt = opt.drop_duplicates(subset=["date", "expiry", "strike", "option_type"], keep="last")
    opt = opt.sort_values(["date", "expiry", "strike", "option_type"])
    root.mkdir(parents=True, exist_ok=True)
    opt.to_parquet(root / "nifty_options.parquet", index=False)
    return opt


def load_nifty_options(cache_dir: str, lookback_days: int = 900) -> pd.DataFrame:
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    parquet = root / "nifty_options.parquet"
    end = datetime.now(IST).date()
    start = end - timedelta(days=lookback_days)

    kaggle_roots = [Path(x) for x in os.environ.get("KAGGLE_DATASET_ROOTS", "").split(os.pathsep) if x]
    if kaggle_roots:
        try:
            from src.kaggle_options import load_from_kaggle_roots

            k = load_from_kaggle_roots(kaggle_roots)
            if not k.empty:
                k = k[(pd.to_datetime(k.date).dt.date >= start) & (pd.to_datetime(k.date).dt.date <= end)].copy()
                if not k.empty:
                    return _finalize(k, root, start, end)
        except Exception as exc:
            print(f"Kaggle loader failed; falling back to NSE archives/cache: {exc}")

    frames: list[pd.DataFrame] = []
    if parquet.exists():
        old = pd.read_parquet(parquet)
        if not old.empty:
            old["date"] = pd.to_datetime(old["date"], errors="coerce").dt.normalize()
            old["expiry"] = pd.to_datetime(old["expiry"], errors="coerce").dt.normalize()
            old = old[(old.date.dt.date >= start) & (old.date.dt.date <= end)]
            if not old.empty:
                frames.append(old)

    external = Path("/tmp/nse-fno-data-bank")
    if external.exists():
        frames.extend(_parse_external_archives(external, start, end))

    if not frames:
        raw_dir = root / "fo_raw"
        dates = [end - timedelta(days=i) for i in range(lookback_days, -1, -1) if (end - timedelta(days=i)).weekday() < 5]
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(_download, d, raw_dir) for d in dates]
            for fut in as_completed(futures):
                x = fut.result()
                if x is not None:
                    frames.append(x)

    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    return _finalize(raw, root, start, end)
