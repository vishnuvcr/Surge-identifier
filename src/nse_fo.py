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


def _parse(content: bytes, dt: date) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        with zf.open(zf.namelist()[0]) as f:
            raw = pd.read_csv(f)
    raw.columns = [str(c).strip() for c in raw.columns]

    def pick(*names):
        for n in names:
            if n in raw.columns:
                return raw[n]
        return pd.Series(index=raw.index, dtype=float)

    out = pd.DataFrame({
        "symbol": pick("TckrSymb", "SYMBOL").astype(str).str.strip(),
        "instrument": pick("FinInstrmTp", "INSTRUMENT").astype(str).str.upper(),
        "option_type": pick("OptnTp", "OPTTYPE").astype(str).str.upper(),
        "oi": pd.to_numeric(pick("OpnIntrst", "OPENINT"), errors="coerce").fillna(0),
        "oi_change": pd.to_numeric(pick("ChngInOpnIntrst", "CHNGINOPENINT"), errors="coerce").fillna(0),
        "volume": pd.to_numeric(pick("TtlTradgVol", "CONTRACTS"), errors="coerce").fillna(0),
    })
    out["date"] = pd.Timestamp(dt)
    return out


def _download(dt: date, raw_dir: Path) -> pd.DataFrame | None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    fp = raw_dir / f"{dt:%Y%m%d}.zip"
    try:
        if fp.exists():
            return _parse(fp.read_bytes(), dt)
        r = requests.get(BASE.format(ymd=dt.strftime("%Y%m%d")), headers={"User-Agent": UA}, timeout=30)
        if r.ok and r.content[:2] == b"PK":
            fp.write_bytes(r.content)
            return _parse(r.content, dt)
    except Exception:
        pass
    time.sleep(0.4)
    return None


def load_fo(cache_dir: str, lookback_days: int = 420) -> pd.DataFrame:
    root = Path(cache_dir)
    raw_dir = root / "fo_raw"
    parquet = root / "fo.parquet"
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
        futures = {ex.submit(_download, d, raw_dir): d for d in needed}
        for fut in as_completed(futures):
            x = fut.result()
            if x is not None and not x.empty:
                frames.append(x)
    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "fut_oi", "fut_oi_change", "fut_volume", "call_oi", "put_oi", "pcr"])

    raw = pd.concat(frames, ignore_index=True)
    fut = raw[raw["instrument"].str.contains("FUT", na=False)].groupby(["date", "symbol"]).agg(
        fut_oi=("oi", "sum"), fut_oi_change=("oi_change", "sum"), fut_volume=("volume", "sum")
    )
    calls = raw[raw["option_type"].isin(["CE", "CA", "CALL"])].groupby(["date", "symbol"])["oi"].sum().rename("call_oi")
    puts = raw[raw["option_type"].isin(["PE", "PA", "PUT"])].groupby(["date", "symbol"])["oi"].sum().rename("put_oi")
    out = fut.join(calls, how="outer").join(puts, how="outer").fillna(0).reset_index()
    out["pcr"] = out["put_oi"] / out["call_oi"].replace(0, pd.NA)
    out.to_parquet(parquet, index=False)
    return out
