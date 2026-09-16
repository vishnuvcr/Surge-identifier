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
    return pd.DataFrame({
        "date": pd.Timestamp(dt).normalize(),
        "symbol": _pick(raw, "TckrSymb", "SYMBOL").astype(str).str.strip().str.upper(),
        "instrument": _pick(raw, "FinInstrmTp", "INSTRUMENT").astype(str).str.strip().str.upper(),
        "option_type": _pick(raw, "OptnTp", "OPTTYPE").astype(str).str.strip().str.upper(),
        "expiry": pd.to_datetime(_pick(raw, "XpryDt", "EXPIRYDT"), errors="coerce").dt.normalize(),
        "strike": pd.to_numeric(_pick(raw, "StrkPric", "STRIKEPRICE"), errors="coerce"),
        "open": pd.to_numeric(_pick(raw, "OpnPric", "OPENPRICE"), errors="coerce"),
        "high": pd.to_numeric(_pick(raw, "HghPric", "HIGHPRICE"), errors="coerce"),
        "low": pd.to_numeric(_pick(raw, "LwPric", "LOWPRICE"), errors="coerce"),
        "close": pd.to_numeric(_pick(raw, "ClsPric", "CLSPRIC", "CLOSEPRICE"), errors="coerce"),
        "settlement": pd.to_numeric(_pick(raw, "SttlmPric", "SETTLEPRICE"), errors="coerce"),
        "volume": pd.to_numeric(_pick(raw, "TtlTradgVol", "CONTRACTS"), errors="coerce").fillna(0),
        "oi": pd.to_numeric(_pick(raw, "OpnIntrst", "OPENINT"), errors="coerce").fillna(0),
    })


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

    # Normalize date/expiry after concatenating cached and newly downloaded frames.
    # Older parquet files can preserve object/string dtypes, which otherwise break
    # the merge against the freshly parsed datetime64[ns] futures table.
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    raw["expiry"] = pd.to_datetime(raw["expiry"], errors="coerce").dt.normalize()
    raw = raw.dropna(subset=["date", "symbol"])

    # Keep both NIFTY options and the nearest available NIFTY futures close as an underlying proxy.
    nifty = raw[raw["symbol"] == "NIFTY"].copy()
    fut = nifty[nifty["instrument"].str.contains("FUT", na=False) & nifty["close"].notna()].copy()
    if not fut.empty:
        fut["days_to_expiry"] = (fut["expiry"] - fut["date"]).dt.days
        fut = fut[fut["days_to_expiry"] >= 0]
        fut = fut.sort_values(["date", "days_to_expiry"])
        fut_daily = fut.groupby("date", as_index=False).first()[["date", "close"]].rename(columns={"close": "underlying_close"})
        fut_daily["date"] = pd.to_datetime(fut_daily["date"], errors="coerce").dt.normalize()
    else:
        fut_daily = pd.DataFrame({"date": pd.Series([], dtype="datetime64[ns]"), "underlying_close": pd.Series([], dtype=float)})

    opt = nifty[nifty["instrument"].str.contains("OPT", na=False)].copy()
    opt = opt[opt["option_type"].isin(["CE", "PE", "CALL", "PUT"])]
    opt["option_type"] = opt["option_type"].replace({"CALL": "CE", "PUT": "PE"})
    opt = opt.merge(fut_daily, on="date", how="left", validate="many_to_one")
    opt = opt.drop_duplicates(subset=["date", "expiry", "strike", "option_type"], keep="last")
    opt = opt.sort_values(["date", "expiry", "strike", "option_type"])
    opt.to_parquet(parquet, index=False)
    return opt
