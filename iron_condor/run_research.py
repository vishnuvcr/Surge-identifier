from __future__ import annotations

import argparse
import pandas as pd

from backtest import summarize


def load_contracts(path: str) -> pd.DataFrame:
    """Load normalized option history.

    Required columns for the full engine include timestamp/trade_date, expiry,
    strike, option_type, close or mid, and preferably bid/ask, volume, OI,
    IV and delta. This loader intentionally validates rather than fabricates.
    """
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    required = {"expiry", "strike", "option_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not ({"mid", "close", "ltp"} & set(df.columns)):
        raise ValueError("Need observed option premium column: mid, close or ltp")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="normalized option history")
    ap.add_argument("--output", default="reports/iron_condor_weekly.csv")
    args = ap.parse_args()
    df = load_contracts(args.data)
    print(f"Loaded {len(df):,} option observations")
    print("Next step: connect normalized observations to the weekly lifecycle simulator.")
    print("Do not interpret this validation loader as a completed backtest.")


if __name__ == "__main__":
    main()
