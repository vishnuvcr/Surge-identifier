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
    urls = [
        f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{yyyy}/{mon}/fo{dd}{mon}{yyyy}bhav.csv.zip",
        f"https://archives.nseindia.com/content/historical/DERIVATIVES/{yyyy}/{mon}/fo{dd}{mon}{yyyy}bhav.csv.zip",
        f"https://archives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{yyyy}{dt:%m}{dd}_F_0000.csv.zip",
        f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{yyyy}{dt:%m}{dd}_F_0000.csv.zip",
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


def _extract_nifty_options(content: bytes, dt: date) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        csvs = [n for n in zf.namelist() if n.lower().endswith((".csv", ".txt"))]
        if not csvs:
            return pd.DataFrame()
        with zf.open(csvs[0]) as fh:
            raw = pd.read_csv(fh, low_memory=False)
    raw.columns = [str(c).strip() for c in raw.columns]
    required = {"INSTRUMENT", "SYMBOL", "EXPIRY_DT", "STRIKE_PR", "OPTION_TYP", "CLOSE"}
    if not required.issubset(raw.columns):
        return pd.DataFrame()
    x = raw.loc[
        (raw["INSTRUMENT"].astype(str).str.upper() == "OPTIDX")
        & (raw["SYMBOL"].astype(str).str.upper() == "NIFTY")
        & (raw["OPTION_TYP"].astype(str).str.upper().isin(["CE", "PE"]))
    ].copy()
    if x.empty:
        return pd.DataFrame()

    def num(col):
        return pd.to_numeric(x[col], errors="coerce") if col in x.columns else pd.Series(np.nan, index=x.index)

    out = pd.DataFrame({
        "date": pd.Timestamp(dt),
        "expiry": pd.to_datetime(x["EXPIRY_DT"], errors="coerce").dt.normalize(),
        "strike": num("STRIKE_PR"),
        "option_type": x["OPTION_TYP"].astype(str).str.upper(),
        "open": num("OPEN"),
        "high": num("HIGH"),
        "low": num("LOW"),
        "close": num("CLOSE"),
        "settlement": num("SETTLE_PR"),
        "volume": num("CONTRACTS").fillna(0),
        "open_interest": num("OPEN_INT").fillna(0),
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
