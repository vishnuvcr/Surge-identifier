from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / 'v5/context_config.yaml').read_text())
OUT = ROOT / CFG['context_path']
START = pd.Timestamp(CFG['start']).tz_localize('UTC')
END = (pd.Timestamp(CFG['end']) + pd.Timedelta(days=1)).tz_localize('UTC')


def yahoo_history(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    p1 = int(start.timestamp())
    p2 = int(end.timestamp())
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(symbol, safe="")}'
    params = {'period1': p1, 'period2': p2, 'interval': '1d', 'events': 'history', 'includeAdjustedClose': 'true'}
    r = requests.get(url, params=params, timeout=45, headers={'User-Agent': 'Mozilla/5.0'})
    r.raise_for_status()
    js = r.json()['chart']['result'][0]
    q = js['indicators']['quote'][0]
    ts = pd.to_datetime(js['timestamp'], unit='s', utc=True).tz_convert('Asia/Kolkata').normalize().tz_localize(None)
    df = pd.DataFrame({'date': ts})
    for k in ['open', 'high', 'low', 'close', 'volume']:
        v = q.get(k, [np.nan] * len(df))
        df[k] = v
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['date', 'close'])
    return df[['date', 'open', 'high', 'low', 'close', 'volume']]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, symbol in CFG['yahoo_symbols'].items():
        last_err = None
        for attempt in range(3):
            try:
                d = yahoo_history(symbol, START, END)
                d['series'] = name
                frames.append(d)
                print(f'{name}: {len(d)} rows {d.date.min().date()}..{d.date.max().date()}')
                break
            except Exception as exc:  # pragma: no cover - network retry
                last_err = exc
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f'failed to download {name}: {last_err}')
    out = pd.concat(frames, ignore_index=True).sort_values(['date', 'series'])
    out.to_parquet(OUT, index=False)
    print(f'wrote {OUT} rows={len(out)} series={out.series.nunique()}')


if __name__ == '__main__':
    main()
