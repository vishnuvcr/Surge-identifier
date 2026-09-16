from __future__ import annotations

import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

IST = timezone(timedelta(hours=5, minutes=30))
BASE = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{ymd}_F_0000.csv.zip"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/134 Safari/537.36"


def _pick(raw: pd.DataFrame, *names):
    for n in names:
        if n in raw.columns:
            return raw[n]
    return pd.Series(index=raw.index, dtype=float)


def _parse(content: bytes, dt: date) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        with zf.open(zf.namelist()[0]) as f:
            raw = pd.read_csv(f)
    raw.columns = [str(c).strip() for c in raw.columns]
    out = pd.DataFrame({
        "date": pd.Timestamp(dt),
        "symbol": _pick(raw, "TckrSymb", "SYMBOL").astype(str).str.strip().str.upper(),
        "instrument": _pick(raw, "FinInstrmTp", "INSTRUMENT").astype(str).str.strip().str.upper(),
        "option_type": _pick(raw, "OptnTp", "OPTTYPE").astype(str).str.strip().str.upper(),
        "expiry": pd.to_datetime(_pick(raw, "XpryDt", "EXPIRYDT"), errors="coerce"),
        "strike": pd.to_numeric(_pick(raw, "StrkPric", "STRIKEPRICE"), errors="coerce"),
        "open": pd.to_numeric(_pick(raw, "OpnPric", "OPENPRICE"), errors="coerce"),
        "high": pd.to_numeric(_pick(raw, "HghPric", "HIGHPRICE"), errors="coerce"),
        "low": pd.to_numeric(_pick(raw, "LwPric", "LOWPRICE"), errors="coerce"),
        "close": pd.to_numeric(_pick(raw, "ClsPric", "CLSPRIC", "CLOSEPRICE"), errors="coerce"),
        "settlement": pd.to_numeric(_pick(raw, "SttlmPric", "SETTLEPRICE"), errors="coerce"),
        "volume": pd.to_numeric(_pick(raw, "TtlTradgVol", "CONTRACTS"), errors="coerce").fillna(0),
        "oi": pd.to_numeric(_pick(raw, "OpnIntrst", "OPENINT"), errors="coerce").fillna(0),
    })
    return out


def _download(dt: date, raw_dir: Path):
    raw_dir.mkdir(parents=True, exist_ok=True)
    fp = raw_dir / f"{dt:%Y%m%d}.zip"
    try:
        if fp.exists():
            return _parse(fp.read_bytes(), dt)
        r = requests.get(BASE.format(ymd=dt.strftime("%Y%m%d")), headers={"User-Agent": UA}, timeout=40)
        if r.ok and r.content[:2] == b"PK":
            fp.write_bytes(r.content)
            return _parse(r.content, dt)
    except Exception:
        pass
    time.sleep(0.4)
    return None


def load_nifty_options(cache_dir: str, lookback_days: int = 900) -> pd.DataFrame:
    root = Path(cache_dir)
    raw_dir = root / "fo_raw"
    parquet = root / "nifty_options.parquet"
    end = datetime.now(IST).date()
    dates = [end - timedelta(days=i) for i in range(lookback_days, -1, -1) if (end - timedelta(days=i)).weekday() < 5]
    frames = []
    existing = set()
    if parquet.exists():
        old = pd.read_parquet(parquet)
        if not old.empty:
            frames.append(old)
            existing = set(pd.to_datetime(old["date"]).dt.date.unique())
    needed = [d for d in dates if d not in existing]
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(_download, d, raw_dir) for d in needed]
        for fut in as_completed(futures):
            x = fut.result()
            if x is not None:
                frames.append(x)
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[(raw["symbol"] == "NIFTY") & raw["instrument"].str.contains("OPT", na=False)]
    raw = raw[raw["option_type"].isin(["CE", "PE", "CALL", "PUT"])]
    raw["option_type"] = raw["option_type"].replace({"CALL": "CE", "PUT": "PE"})
    raw = raw.drop_duplicates(subset=["date", "expiry", "strike", "option_type"], keep="last")
    raw = raw.sort_values(["date", "expiry", "strike", "option_type"])
    raw.to_parquet(parquet, index=False)
    return raw
