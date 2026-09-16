from __future__ import annotations

import io
import os
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "Chrome/134 Safari/537.36"
)
HEADERS = {
    "User-Agent": UA,
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
}


def trading_days(start: date, end: date) -> list[date]:
    return [x.date() for x in pd.date_range(start, end, freq="B")]


def _download_bytes(dt: date, retries: int = 4) -> bytes | None:
    dd = dt.strftime("%d")
    mon = dt.strftime("%b").upper()
    yyyy = dt.strftime("%Y")
    if dt >= date(2024, 7, 8):
        urls = [
            f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{yyyy}{dt:%m}{dd}_F_0000.csv.zip",
            f"https://archives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{yyyy}{dt:%m}{dd}_F_0000.csv.zip",
        ]
    else:
        urls = [
            f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{yyyy}/{mon}/fo{dd}{mon}{yyyy}bhav.csv.zip",
            f"https://archives.nseindia.com/content/historical/DERIVATIVES/{yyyy}/{mon}/fo{dd}{mon}{yyyy}bhav.csv.zip",
        ]
    for url in urls:
        for attempt in range(retries):
            try:
                r = requests.get(url, headers=HEADERS, timeout=45)
                if r.ok and r.content[:2] == b"PK":
                    return r.content
                if r.status_code == 404:
                    break
            except Exception:
                pass
            time.sleep(min(2.5 * (attempt + 1), 10))
    return None


def _pick(raw: pd.DataFrame, *names: str) -> pd.Series | None:
    for name in names:
        if name in raw.columns:
            return raw[name]
    return None


def _num(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(np.nan, index=index, dtype="float64")
    return pd.to_numeric(series, errors="coerce")


def _extract_nifty_options(content: bytes, dt: date) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        csvs = [n for n in zf.namelist() if n.lower().endswith((".csv", ".txt"))]
        if not csvs:
            return pd.DataFrame()
        with zf.open(csvs[0]) as fh:
            raw = pd.read_csv(fh, low_memory=False)

    raw.columns = [str(c).strip() for c in raw.columns]

    # Legacy NSE FO bhavcopy (through 05-Jul-2024).
    if {"INSTRUMENT", "SYMBOL", "EXPIRY_DT", "STRIKE_PR", "OPTION_TYP", "CLOSE"}.issubset(raw.columns):
        symbol = raw["SYMBOL"].astype(str).str.upper()
        option_type = raw["OPTION_TYP"].astype(str).str.upper()
        expiry = pd.to_datetime(raw["EXPIRY_DT"], errors="coerce").dt.normalize()
        strike = _num(raw["STRIKE_PR"], raw.index)
        close = _num(raw["CLOSE"], raw.index)
        open_ = _num(_pick(raw, "OPEN"), raw.index)
        high = _num(_pick(raw, "HIGH"), raw.index)
        low = _num(_pick(raw, "LOW"), raw.index)
        settlement = _num(_pick(raw, "SETTLE_PR"), raw.index)
        volume = _num(_pick(raw, "CONTRACTS"), raw.index).fillna(0)
        oi = _num(_pick(raw, "OPEN_INT"), raw.index).fillna(0)
        xmask = symbol.eq("NIFTY") & option_type.isin(["CE", "PE"])
    # NSE UDiFF F&O common bhavcopy (from 08-Jul-2024).
    else:
        required = {"TckrSymb", "XpryDt", "StrkPric", "OptnTp", "ClsPric"}
        if not required.issubset(raw.columns):
            return pd.DataFrame()
        symbol = raw["TckrSymb"].astype(str).str.upper().str.strip()
        option_type = raw["OptnTp"].astype(str).str.upper().str.strip()
        expiry = pd.to_datetime(raw["XpryDt"], errors="coerce").dt.normalize()
        strike = _num(raw["StrkPric"], raw.index)
        close = _num(raw["ClsPric"], raw.index)
        open_ = _num(_pick(raw, "OpnPric"), raw.index)
        high = _num(_pick(raw, "HghPric"), raw.index)
        low = _num(_pick(raw, "LwPric"), raw.index)
        settlement = _num(_pick(raw, "SttlmPric"), raw.index)
        volume = _num(_pick(raw, "TtlTradgVol"), raw.index).fillna(0)
        oi = _num(_pick(raw, "OpnIntrst"), raw.index).fillna(0)
        # Do not rely on FinInstrmTp here: NSE's UDiFF instrument-type labels have varied
        # across exchange/format revisions. NIFTY + CE/PE is the stable identifier.
        xmask = symbol.eq("NIFTY") & option_type.isin(["CE", "PE"])

    xidx = raw.index[xmask]
    if len(xidx) == 0:
        return pd.DataFrame()

    out = pd.DataFrame({
        "date": pd.Timestamp(dt),
        "expiry": expiry.loc[xidx].values,
        "strike": strike.loc[xidx].values,
        "option_type": option_type.loc[xidx].values,
        "open": open_.loc[xidx].values,
        "high": high.loc[xidx].values,
        "low": low.loc[xidx].values,
        "close": close.loc[xidx].values,
        "settlement": settlement.loc[xidx].values,
        "volume": volume.loc[xidx].values,
        "open_interest": oi.loc[xidx].values,
    })
    out = out.dropna(subset=["expiry", "strike", "close"])
    out = out.loc[out["expiry"] >= out["date"]]
    return out.drop_duplicates(["date", "expiry", "strike", "option_type"], keep="last")


def main() -> None:
    start = date.fromisoformat(os.environ.get("NSE_HISTORY_START", "2015-01-01"))
    end = date.fromisoformat(os.environ.get("NSE_HISTORY_END", str(datetime.now().date())))
    max_workers = int(os.environ.get("NSE_HISTORY_WORKERS", "5"))
    out_dir = Path(os.environ.get("NSE_HISTORY_OUT", "data/cache/nse_nifty_parts"))
    out_dir.mkdir(parents=True, exist_ok=True)

    days = trading_days(start, end)
    print(f"NSE history window: {start} -> {end} ({len(days):,} business days)", flush=True)
    failures: list[str] = []

    def worker(dt: date):
        fp = out_dir / f"{dt:%Y%m%d}.parquet"
        if fp.exists() and fp.stat().st_size > 0:
            return dt, fp, None
        content = _download_bytes(dt)
        if content is None:
            return dt, None, "download_failed_or_holiday"
        try:
            frame = _extract_nifty_options(content, dt)
            if frame.empty:
                return dt, None, "no_nifty_options"
            frame.to_parquet(fp, index=False)
            return dt, fp, None
        except Exception as exc:
            return dt, None, f"parse:{type(exc).__name__}:{exc}"

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(worker, dt) for dt in days]
        for i, fut in enumerate(as_completed(futures), start=1):
            dt, fp, err = fut.result()
            if err and err != "download_failed_or_holiday":
                failures.append(f"{dt}:{err}")
            if i % 100 == 0 or i == len(days):
                ready = len(list(out_dir.glob("*.parquet")))
                print(f"Progress: {i:,}/{len(days):,} | nonempty parts={ready:,} | parse failures={len(failures):,}", flush=True)

    files = sorted(out_dir.glob("*.parquet"))
    if not files:
        raise RuntimeError("No NIFTY option files were downloaded from NSE archives")

    merged = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"]).dt.normalize()
    merged["expiry"] = pd.to_datetime(merged["expiry"]).dt.normalize()
    merged = merged.sort_values(["date", "expiry", "strike", "option_type"])
    merged = merged.drop_duplicates(["date", "expiry", "strike", "option_type"], keep="last").reset_index(drop=True)

    out = Path("data/cache/nifty_options_long.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    report = {
        "start": str(merged.date.min().date()),
        "end": str(merged.date.max().date()),
        "rows": int(len(merged)),
        "trading_days": int(merged.date.nunique()),
        "expiries": int(merged.expiry.nunique()),
        "contracts": int(merged[["expiry", "strike", "option_type"]].drop_duplicates().shape[0]),
        "files": int(len(files)),
        "parse_failure_count": int(len(failures)),
        "parse_failures": failures[:100],
    }
    Path("data/cache/nifty_options_long_manifest.json").write_text(pd.Series(report).to_json(indent=2))
    shutil.rmtree(out_dir, ignore_errors=True)
    print("Final long-history dataset:", report, flush=True)


if __name__ == "__main__":
    main()
