from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
import pandas as pd
import requests

BASES = [
    "https://raw.githubusercontent.com/ganeshbiyer/Nse_Historical_Data/main/{symbol}.parquet",
    "https://raw.githubusercontent.com/ganeshbiyer/Nse_Historical_Data_2026/main/{symbol}.parquet",
]


def normalise(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    cols = {str(c).strip().lower().replace(' ', '_'): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None
    ts_col = pick('timestamp', 'datetime', 'date', 'time')
    if ts_col is None:
        raise ValueError(f'{symbol}: no timestamp-like column in {df.columns.tolist()}')
    out = pd.DataFrame()
    out['timestamp'] = pd.to_datetime(df[ts_col], errors='coerce')
    for target, aliases in {
        'open': ('open', 'opn'),
        'high': ('high', 'hgh'),
        'low': ('low', 'l'),
        'close': ('close', 'cls'),
        'volume': ('volume', 'vol', 'traded_volume'),
    }.items():
        c = pick(*aliases)
        out[target] = pd.to_numeric(df[c], errors='coerce') if c else 0.0
    out['symbol'] = symbol.upper()
    out = out.dropna(subset=['timestamp','open','high','low','close']).sort_values('timestamp')
    out = out[(out.timestamp.dt.time >= pd.Timestamp('09:15').time()) & (out.timestamp.dt.time <= pd.Timestamp('15:30').time())]
    return out[['timestamp','symbol','open','high','low','close','volume']]


def fetch_symbol(symbol: str, out_dir: Path) -> Path:
    chunks = []
    for base in BASES:
        url = base.format(symbol=symbol)
        r = requests.get(url, timeout=90)
        if r.ok and len(r.content) > 1000:
            df = pd.read_parquet(BytesIO(r.content))
            chunks.append(normalise(df, symbol))
    if not chunks:
        raise RuntimeError(f'Could not download {symbol} from configured public archives')
    x = pd.concat(chunks, ignore_index=True).drop_duplicates(['symbol','timestamp']).sort_values('timestamp')
    # Convert 1-minute source bars to 5-minute execution bars.
    x = x.set_index('timestamp')
    agg = x.groupby('symbol').resample('5min', origin='start_day', offset='15min').agg(
        open=('open','first'), high=('high','max'), low=('low','min'), close=('close','last'), volume=('volume','sum')
    ).dropna(subset=['open','high','low','close']).reset_index()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'{symbol}.parquet'
    agg.to_parquet(path, index=False)
    print(symbol, len(agg), agg.timestamp.min(), agg.timestamp.max())
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', default='RELIANCE,HDFCBANK,ICICIBANK,INFY,TCS')
    ap.add_argument('--output-dir', default='data/cpr_v1/intraday')
    a = ap.parse_args()
    out = Path(a.output_dir)
    for symbol in [s.strip().upper() for s in a.symbols.split(',') if s.strip()]:
        fetch_symbol(symbol, out)

if __name__ == '__main__':
    main()
