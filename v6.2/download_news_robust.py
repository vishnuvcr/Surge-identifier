from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/cache/v5_news_daily.csv'
MANIFEST = ROOT / 'data/cache/v62_news_manifest.json'
OUT.parent.mkdir(parents=True, exist_ok=True)

# Prefer the six-year research dataset. The old pinned URL returned HTTP 404
# in Actions even though the dataset card still advertises the file. Try the
# current revision and the known snapshot, then use a documented Indian-news
# fallback rather than silently dropping the news modality.
CANDIDATES = [
    ('mveen3/Six_Year_Indian_Stock_Market_Dataset-News_and_Ticker', 'dataset/news_sentiment.csv', None),
    ('mveen3/Six_Year_Indian_Stock_Market_Dataset-News_and_Ticker', 'dataset/news_sentiment.csv', '394913e6b62d643b6449538cf6d70f7f36bd0678'),
    ('mveen3/Six_Year_Indian_Stock_Market_Dataset-News_and_Ticker', 'dataset/processed_news_dataset.csv', None),
    ('dixitdharmansh07/indic-finance', 'indic-finance.csv', None),
]

last_error = None
selected = None
for repo, filename, revision in CANDIDATES:
    try:
        path = hf_hub_download(repo_id=repo, filename=filename, repo_type='dataset', revision=revision)
        df = pd.read_csv(path)
        df.columns = [str(c).strip().lower() for c in df.columns]
        if 'date' not in df.columns and 'parsed_date' in df.columns:
            df['date'] = df['parsed_date']
        if 'date' not in df.columns:
            raise ValueError(f'no date column in {repo}/{filename}')
        # The V6.2 engine consumes aggregate positive/negative/count fields.
        # Keep files that already contain those fields; otherwise use the
        # Indic-Finance fallback, which has them natively.
        signal_cols = [c for c in df.columns if 'pos' in c or 'neg' in c or 'count' in c]
        if not signal_cols:
            raise ValueError(f'no usable sentiment/count columns in {repo}/{filename}')
        df.to_csv(OUT, index=False)
        selected = {'repo': repo, 'filename': filename, 'revision': revision or 'main', 'rows': int(len(df)), 'min_date': str(pd.to_datetime(df['date'], errors='coerce').min().date()), 'max_date': str(pd.to_datetime(df['date'], errors='coerce').max().date()), 'columns': list(df.columns)}
        break
    except Exception as e:
        last_error = repr(e)

if selected is None:
    raise RuntimeError(f'No usable historical news source found. Last error: {last_error}')

MANIFEST.write_text(json.dumps(selected, indent=2))
print(json.dumps(selected, indent=2))
