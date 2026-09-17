from __future__ import annotations

import concurrent.futures as cf
import io
import json
import os
from pathlib import Path

import pandas as pd
import requests
import yaml
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / 'v6.2/config.yaml').read_text())
CACHE = ROOT / 'data/cache'
CACHE.mkdir(parents=True, exist_ok=True)

UA = {'User-Agent': 'V6.2-multimodal-research/1.0'}


def download_weights() -> pd.DataFrame:
    out = CACHE / 'v62_constituent_weights.csv'
    if not out.exists():
        r = requests.get(CFG['constituents']['weights_url'], headers=UA, timeout=120)
        r.raise_for_status()
        out.write_bytes(r.content)
    w = pd.read_csv(out)
    w.columns = [str(c).strip().upper() for c in w.columns]
    if 'DATE' not in w.columns:
        raise RuntimeError('Constituent weights file has no DATE column')
    w['DATE'] = pd.to_datetime(w['DATE'], errors='coerce').dt.normalize()
    w = w.dropna(subset=['DATE']).sort_values('DATE').drop_duplicates('DATE', keep='last')
    return w


def download_sectors() -> pd.DataFrame:
    out = CACHE / 'v62_constituent_sectors.csv'
    if not out.exists():
        r = requests.get(CFG['constituents']['sectors_url'], headers=UA, timeout=120)
        r.raise_for_status()
        out.write_bytes(r.content)
    s = pd.read_csv(out)
    s.columns = [str(c).strip().upper() for c in s.columns]
    return s


def _symbols(weights: pd.DataFrame) -> list[str]:
    ignore = {'DATE'}
    cols = [c for c in weights.columns if c not in ignore]
    # Use symbols that have at least one non-zero historical observation.
    return [c for c in cols if pd.to_numeric(weights[c], errors='coerce').fillna(0).abs().sum() > 0]


def _fetch_price(symbol: str) -> tuple[str, str | None]:
    safe = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in symbol)
    target = CACHE / 'v62_constituent_parts' / f'{safe}.parquet'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return symbol, None
    try:
        path = hf_hub_download(
            repo_id=CFG['constituents']['prices_repo'],
            filename=f'stocks/{symbol}.parquet',
            repo_type='dataset',
        )
        df = pd.read_parquet(path)
        df.columns = [str(c).strip().lower() for c in df.columns]
        required = {'date', 'close'}
        if not required.issubset(df.columns):
            return symbol, 'missing_date_close'
        keep = ['date', 'close']
        if 'volume' in df.columns:
            keep.append('volume')
        x = df[keep].copy()
        x['date'] = pd.to_datetime(x['date'], errors='coerce').dt.normalize()
        x['close'] = pd.to_numeric(x['close'], errors='coerce')
        x['symbol'] = symbol
        x = x.dropna(subset=['date', 'close']).drop_duplicates(['date'], keep='last')
        x.to_parquet(target, index=False)
        return symbol, None
    except Exception as exc:  # best effort; coverage is reported explicitly
        return symbol, f'{type(exc).__name__}:{exc}'


def download_prices(weights: pd.DataFrame) -> dict:
    symbols = _symbols(weights)
    failures: list[dict] = []
    workers = int(os.environ.get('V62_PRICE_WORKERS', '8'))
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch_price, s) for s in symbols]
        for fut in cf.as_completed(futures):
            sym, err = fut.result()
            if err:
                failures.append({'symbol': sym, 'error': err})
    parts = sorted((CACHE / 'v62_constituent_parts').glob('*.parquet'))
    if not parts:
        raise RuntimeError('No constituent price files downloaded')
    frames = [pd.read_parquet(p) for p in parts]
    all_prices = pd.concat(frames, ignore_index=True)
    all_prices['date'] = pd.to_datetime(all_prices['date']).dt.normalize()
    all_prices = all_prices.sort_values(['symbol', 'date']).drop_duplicates(['symbol', 'date'], keep='last')
    out = CACHE / 'v62_constituent_prices.parquet'
    all_prices.to_parquet(out, index=False)
    coverage = all_prices.groupby('symbol')['date'].agg(['min', 'max', 'count']).reset_index()
    coverage.to_csv(CACHE / 'v62_constituent_coverage.csv', index=False)
    manifest = {
        'requested_symbols': len(symbols),
        'downloaded_symbols': int(all_prices.symbol.nunique()),
        'rows': int(len(all_prices)),
        'failures': failures[:200],
    }
    (CACHE / 'v62_constituent_manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def download_flows() -> dict:
    out = CACHE / 'v62_fii_dii_raw.xlsx'
    url = CFG['flows']['fallback_url']
    if not out.exists():
        r = requests.get(url, headers=UA, timeout=120)
        r.raise_for_status()
        out.write_bytes(r.content)
    parsed_rows = []
    xl = pd.ExcelFile(out)
    for sheet in xl.sheet_names:
        try:
            raw = pd.read_excel(out, sheet_name=sheet)
        except Exception:
            continue
        if raw.empty:
            continue
        raw.columns = [str(c).strip().lower() for c in raw.columns]
        date_col = next((c for c in raw.columns if 'date' in c), None)
        if not date_col:
            continue
        raw[date_col] = pd.to_datetime(raw[date_col], errors='coerce', dayfirst=True).dt.normalize()
        for side, needles in {
            'fii_net': ['fii net', 'fii/fpi net', 'fii net value', 'fpi net'],
            'dii_net': ['dii net', 'dii net value'],
        }.items():
            col = next((c for c in raw.columns if any(n in c for n in needles)), None)
            if col:
                z = raw[[date_col, col]].rename(columns={date_col: 'date', col: side}).copy()
                z[side] = pd.to_numeric(z[side].astype(str).str.replace(',', '', regex=False), errors='coerce')
                parsed_rows.append(z)
    if parsed_rows:
        out_df = parsed_rows[0]
        for z in parsed_rows[1:]:
            out_df = out_df.merge(z, on='date', how='outer')
        out_df = out_df.groupby('date', as_index=False).last().sort_values('date')
    else:
        out_df = pd.DataFrame(columns=['date', 'fii_net', 'dii_net'])
    out_df.to_parquet(CACHE / 'v62_fii_dii.parquet', index=False)
    report = {
        'rows': int(len(out_df)),
        'start': str(out_df.date.min().date()) if len(out_df) else None,
        'end': str(out_df.date.max().date()) if len(out_df) else None,
        'sheets': xl.sheet_names,
    }
    (CACHE / 'v62_fii_dii_manifest.json').write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    w = download_weights()
    s = download_sectors()
    manifest = download_prices(w)
    flows = download_flows()
    print(json.dumps({
        'weights_rows': len(w),
        'weights_start': str(w.DATE.min().date()),
        'weights_end': str(w.DATE.max().date()),
        'sector_rows': len(s),
        'constituent_prices': manifest,
        'fii_dii': flows,
    }, indent=2))


if __name__ == '__main__':
    main()
