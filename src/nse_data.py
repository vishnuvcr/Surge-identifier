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
BASE = "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{ymd}_F_0000.csv.zip"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/134 Safari/537.36"


def trading_dates(end: date, lookback_days: int) -> list[date]:
    return [end - timedelta(days=i) for i in range(lookback_days, -1, -1) if (end - timedelta(days=i)).weekday() < 5]


def _read_zip(content: bytes, dt: date) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        with zf.open(zf.namelist()[0]) as f:
            raw = pd.read_csv(f)
    raw.columns = [str(c).strip() for c in raw.columns]
    aliases = {
        "symbol": ["TckrSymb", "SYMBOL"],
        "series": ["SctySrs", "SERIES"],
        "open": ["OpnPric", "OPEN"],
        "high": ["HghPric", "HIGH"],
        "low": ["LwPric", "LOW"],
        "close": ["ClsPric", "CLOSE"],
        "prev_close": ["PrvsClsgPric", "PREV_CLOSE", "PREVCLOSE"],
        "volume": ["TtlTradgVol", "TOTTRDQTY", "TOTTRADGVOL"],
        "turnover": ["TtlTrfVal", "TOTTRDVAL"],
    }
    out = pd.DataFrame(index=raw.index)
    for dest, names in aliases.items():
        found = next((n for n in names if n in raw.columns), None)
        if found is not None:
            out[dest] = raw[found]
    required = ["symbol", "open", "high", "low", "close", "prev_close", "volume"]
    if any(c not in out for c in required):
        raise ValueError(f"Unexpected NSE columns on {dt}: {raw.columns.tolist()}")
    out["series"] = out.get("series", "EQ")
    out["date"] = pd.Timestamp(dt)
    for c in ["open", "high", "low", "close", "prev_close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["turnover"] = pd.to_numeric(out.get("turnover", out["close"] * out["volume"]), errors="coerce")
    out = out[(out["series"].astype(str).str.upper() == "EQ") & (out["volume"] > 0)].copy()
    out["symbol"] = out["symbol"].astype(str).str.strip()
    return out[["date", "symbol", "series", "open", "high", "low", "close", "prev_close", "volume", "turnover"]]


def download_day(dt: date, raw_dir: Path, attempts: int = 3) -> pd.DataFrame | None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    fp = raw_dir / f"{dt:%Y%m%d}.zip"
    if fp.exists():
        try:
            return _read_zip(fp.read_bytes(), dt)
        except Exception:
            fp.unlink(missing_ok=True)
    url = BASE.format(ymd=dt.strftime("%Y%m%d"))
    for attempt in range(attempts):
        try:
            r = requests.get(url, headers={"User-Agent": UA, "Referer": "https://www.nseindia.com/"}, timeout=30)
            if r.ok and r.content[:2] == b"PK":
                fp.write_bytes(r.content)
                return _read_zip(r.content, dt)
        except Exception:
            pass
        time.sleep(0.8 * (attempt + 1))
    return None


def load_prices(cache_dir: str, lookback_days: int = 420) -> pd.DataFrame:
    root = Path(cache_dir)
    raw_dir = root / "raw"
    parquet = root / "prices.parquet"
    end = datetime.now(IST).date()
    dates = trading_dates(end, lookback_days)

    frames: list[pd.DataFrame] = []
    if parquet.exists():
        old = pd.read_parquet(parquet)
        if not old.empty:
            frames.append(old)
            existing = set(pd.to_datetime(old["date"]).dt.date.unique())
        else:
            existing = set()
    else:
        existing = set()

    needed = [d for d in dates if d not in existing]
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(download_day, d, raw_dir): d for d in needed}
        for fut in as_completed(futures):
            x = fut.result()
            if x is not None and not x.empty:
                frames.append(x)

    if not frames:
        raise RuntimeError("No NSE bhavcopy data could be loaded")
    data = pd.concat(frames, ignore_index=True)
    data["date"] = pd.to_datetime(data["date"])
    data = data.drop_duplicates(["date", "symbol"], keep="last").sort_values(["date", "symbol"])
    data.to_parquet(parquet, index=False)
    return data
