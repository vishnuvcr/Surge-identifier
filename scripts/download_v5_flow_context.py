from __future__ import annotations

from pathlib import Path
import requests
import pandas as pd

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / 'data/cache/v5_flow_context.parquet'
    url = 'https://fii-diidata.mrchartist.com/api/history-full'
    r = requests.get(url, timeout=60, headers={'User-Agent': 'V5-research/1.0'})
    r.raise_for_status()
    payload = r.json()
    rows = payload.get('data', payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise RuntimeError('No FII/DII history returned')
    df = pd.DataFrame(rows)
    if 'd' in df.columns: df['date'] = pd.to_datetime(df['d'])
    elif 'date' in df.columns: df['date'] = pd.to_datetime(df['date'])
    else: raise RuntimeError(f'No date field in flow payload: {df.columns.tolist()}')
    ren = {'fn':'fii_net','dn':'dii_net','pcr':'pcr','sentiment_score':'sentiment_score',
           'fii_idx_fut_long':'fii_idx_fut_long','fii_idx_fut_short':'fii_idx_fut_short'}
    keep = ['date'] + [c for c in ren if c in df.columns]
    df = df[keep].rename(columns=ren).sort_values('date').drop_duplicates('date', keep='last')
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f'wrote {out} rows={len(df)} range={df.date.min().date()}..{df.date.max().date()}')

if __name__ == '__main__': main()
