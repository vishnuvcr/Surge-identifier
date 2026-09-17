from __future__ import annotations

import json
import re
from pathlib import Path

import kagglehub
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data/cache'
DATASET = 'pravinpari/fii-and-dii-investments-in-indian-stock-market'


def norm(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', str(name).strip().lower()).strip('_')


def read_candidate(path: Path) -> list[pd.DataFrame]:
    out: list[pd.DataFrame] = []
    if path.suffix.lower() == '.csv':
        try:
            out.append(pd.read_csv(path))
        except Exception:
            pass
    elif path.suffix.lower() in {'.xlsx', '.xls'}:
        try:
            xl = pd.ExcelFile(path)
            for sheet in xl.sheet_names:
                try:
                    out.append(pd.read_excel(path, sheet_name=sheet))
                except Exception:
                    continue
        except Exception:
            pass
    elif path.suffix.lower() == '.parquet':
        try:
            out.append(pd.read_parquet(path))
        except Exception:
            pass
    return out


def find_date_column(cols: list[str]) -> str | None:
    preferred = ['date', 'trade_date', 'trading_date', 'timestamp', 'time']
    for key in preferred:
        for c in cols:
            if c == key or c.startswith(key + '_'):
                return c
    return None


def find_col(cols: list[str], must_include: tuple[str, ...], exclude: tuple[str, ...] = ()) -> str | None:
    ranked = []
    for c in cols:
        if not any(x in c for x in must_include):
            continue
        if any(x in c for x in exclude):
            continue
        score = 0
        if c.endswith('_net') or c == 'net':
            score += 20
        if 'net' in c:
            score += 10
        if 'cash' in c or 'equity' in c:
            score += 5
        ranked.append((score, c))
    return sorted(ranked, reverse=True)[0][1] if ranked else None


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(',', '', regex=False)
        .str.replace('%', '', regex=False)
        .str.replace('₹', '', regex=False)
        .str.strip(),
        errors='coerce',
    )


def extract_frame(df: pd.DataFrame, source: str) -> tuple[pd.DataFrame, dict] | None:
    if df.empty:
        return None
    x = df.copy()
    x.columns = [norm(c) for c in x.columns]
    cols = list(x.columns)
    date_col = find_date_column(cols)
    if date_col is None:
        return None

    x['date'] = pd.to_datetime(x[date_col], errors='coerce', dayfirst=True).dt.normalize()

    fii_net = find_col(cols, ('fii', 'fpi'), exclude=('future', 'futures', 'long', 'short'))
    dii_net = find_col(cols, ('dii',), exclude=('future', 'futures', 'long', 'short'))

    def buy_col(kind: str) -> str | None:
        return next((c for c in cols if kind in c and 'buy' in c), None)

    def sell_col(kind: str) -> str | None:
        return next((c for c in cols if kind in c and 'sell' in c), None)

    fii_buy, fii_sell = buy_col('fii'), sell_col('fii')
    dii_buy, dii_sell = buy_col('dii'), sell_col('dii')

    if fii_net is None and fii_buy and fii_sell:
        x['fii_net'] = numeric(x[fii_buy]) - numeric(x[fii_sell])
        fii_source = f'{fii_buy} - {fii_sell}'
    elif fii_net is not None:
        x['fii_net'] = numeric(x[fii_net])
        fii_source = fii_net
    else:
        return None

    if dii_net is None and dii_buy and dii_sell:
        x['dii_net'] = numeric(x[dii_buy]) - numeric(x[dii_sell])
        dii_source = f'{dii_buy} - {dii_sell}'
    elif dii_net is not None:
        x['dii_net'] = numeric(x[dii_net])
        dii_source = dii_net
    else:
        return None

    out = x[['date', 'fii_net', 'dii_net']].copy()
    out = out.dropna(subset=['date'])
    out = out.groupby('date', as_index=False)[['fii_net', 'dii_net']].last()
    out = out.sort_values('date').drop_duplicates('date', keep='last')
    if out.empty:
        return None

    meta = {
        'source_file': source,
        'date_column': date_col,
        'fii_source': fii_source,
        'dii_source': dii_source,
        'rows': int(len(out)),
        'start': str(out['date'].min().date()),
        'end': str(out['date'].max().date()),
    }
    return out, meta


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    root = Path(kagglehub.dataset_download(DATASET))
    candidates = sorted(
        p for p in root.rglob('*')
        if p.is_file() and p.suffix.lower() in {'.csv', '.xlsx', '.xls', '.parquet'}
    )
    if not candidates:
        raise RuntimeError(f'No tabular files found in Kaggle dataset: {root}')

    matches: list[tuple[pd.DataFrame, dict]] = []
    for path in candidates:
        for frame in read_candidate(path):
            parsed = extract_frame(frame, str(path.relative_to(root)))
            if parsed is not None:
                matches.append(parsed)

    if not matches:
        raise RuntimeError('Could not identify a table containing both FII/FPI and DII daily net flow fields')

    # Prefer the longest, latest-ending valid table.
    matches.sort(key=lambda t: (len(t[0]), t[0]['date'].max()), reverse=True)
    data, meta = matches[0]
    data.to_parquet(CACHE / 'v62_fii_dii.parquet', index=False)

    report = {
        'dataset': DATASET,
        'dataset_path': str(root),
        'selected': meta,
        'candidate_tables': [m for _, m in matches],
        'rows': int(len(data)),
        'start': str(data.date.min().date()),
        'end': str(data.date.max().date()),
        'fii_missing': int(data.fii_net.isna().sum()),
        'dii_missing': int(data.dii_net.isna().sum()),
    }
    (CACHE / 'v62_fii_dii_manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
