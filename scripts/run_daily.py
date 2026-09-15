from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# When executed as `python scripts/run_daily.py`, Python puts `scripts/` on
# sys.path rather than the repository root. Add the project root explicitly so
# the `src` package is importable in GitHub Actions and local runs alike.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.nse_data import IST, load_prices
from src.nse_fo import load_fo
from src.surge_model import build_features, score_latest, train_walk_forward


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    lookback = int(cfg["lookback_days"])
    data = load_prices(str(ROOT / "data" / "cache"), lookback)
    data["date"] = pd.to_datetime(data["date"])

    # Keep the broad NSE EQ universe, but suppress structurally illiquid names.
    med_turn = data.groupby("symbol")["turnover"].median() / 1e7
    eligible = med_turn[med_turn >= float(cfg["min_avg_turnover_cr"])].index
    data = data[data["symbol"].isin(eligible)].copy()

    # Merge stock-level futures/options positioning. The model has neutral fallbacks
    # when a historical F&O file is temporarily unavailable.
    fo = load_fo(str(ROOT / "data" / "cache"), lookback)
    if not fo.empty:
        data = data.merge(fo, how="left", on=["date", "symbol"])

    feats = build_features(data)
    keep_n = int(cfg["max_stocks"])
    ranked = feats.assign(_turn=feats["turnover"]).sort_values(["date", "_turn"], ascending=[True, False])
    train = ranked.groupby("date", group_keys=False).head(keep_n).drop(columns=["_turn"])

    bundle = train_walk_forward(train, int(cfg["validation_days"]), int(cfg["seed"]))
    latest_date = feats["date"].max()
    latest = feats[feats["date"] == latest_date].copy()
    scores = score_latest(bundle, latest)
    candidates = scores[
        (scores["signal"])
        & (scores["surge_probability"] >= float(cfg["min_candidate_probability"]))
        & (scores["risk_flag"] != "low_liquidity")
    ].head(int(cfg["max_candidates"]))

    out_cols = [
        "symbol", "close", "surge_probability", "expected_move_proxy", "atr_pct",
        "relvol20", "turnover_cr", "risk_flag", "gap", "pcr_dev20",
        "fut_oi_change_z20", "fut_volume_rel20",
    ]
    candidates = candidates[out_cols].copy()
    candidates.columns = [
        "symbol", "last_close", "surge_probability", "expected_move_proxy", "atr_pct",
        "relative_volume", "turnover_cr", "risk_flag", "gap", "pcr_deviation",
        "futures_oi_change_z", "futures_volume_rel20",
    ]
    records = []
    for row in candidates.to_dict("records"):
        for k, v in list(row.items()):
            if hasattr(v, "item"):
                row[k] = v.item()
            if pd.isna(row[k]):
                row[k] = None
        records.append(row)

    stamp = datetime.now(IST)
    report = {
        "asof_date": str(latest_date.date()),
        "generated_at_ist": stamp.isoformat(),
        "event_definition": "Next session intraday high >= +3% from next session open",
        "universe": "NSE CM EQ series with sufficient median turnover",
        "data_layers": ["NSE CM bhavcopy", "NSE F&O bhavcopy / positioning context"],
        "candidates": records,
        "model": {
            "type": "PyTorch MLP with LayerNorm/GELU/Dropout",
            "features": 32,
            "threshold": bundle.threshold,
            "validation_precision": bundle.validation_precision,
            "validation_recall_sensitivity": bundle.validation_recall,
            "validation_specificity": bundle.validation_specificity,
            "validation_window_days": int(cfg["validation_days"]),
        },
        "research_status": "signal-ranking research only; no guarantee of daily/monthly profitability",
    }

    reports = ROOT / "reports"
    site = ROOT / "site"
    (reports / "daily").mkdir(parents=True, exist_ok=True)
    site.mkdir(parents=True, exist_ok=True)
    date_key = str(latest_date.date())
    (reports / "daily" / f"{date_key}.json").write_text(json.dumps(report, indent=2))
    (site / "latest.json").write_text(json.dumps(report, indent=2))

    if candidates.empty:
        table_html = '<tr><td colspan="9">NO QUALIFIED TRADE — confidence/liquidity gate not satisfied.</td></tr>'
    else:
        rows = []
        for r in records:
            rows.append(
                "<tr>"
                f"<td>{r['symbol']}</td><td>{r['last_close']:.2f}</td>"
                f"<td>{r['surge_probability']:.1%}</td><td>{r['expected_move_proxy']:.1%}</td>"
                f"<td>{r['atr_pct']:.1%}</td><td>{r['relative_volume']:.2f}x</td>"
                f"<td>{r['pcr_deviation'] if r['pcr_deviation'] is not None else 'n/a'}</td>"
                f"<td>{r['futures_oi_change_z'] if r['futures_oi_change_z'] is not None else 'n/a'}</td>"
                f"<td>{r['risk_flag']}</td></tr>"
            )
        table_html = "".join(rows)

    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>NSE Surge Identifier</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1250px;margin:40px auto;padding:0 18px;background:#0b1020;color:#e9eef8}}.card{{background:#151c30;border:1px solid #293554;border-radius:14px;padding:20px;margin:14px 0}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:9px;border-bottom:1px solid #2b3550;text-align:left;white-space:nowrap}}.muted{{color:#9eabc4}}.big{{font-size:28px;font-weight:700}}a{{color:#9fc5ff}}</style></head>
<body><h1>NSE Surge Identifier</h1>
<div class='card'><div class='big'>{date_key}</div><div class='muted'>Generated {stamp.strftime('%Y-%m-%d %H:%M IST')} · Event: next session high ≥ +3% from open</div></div>
<div class='card'><h2>Next-session candidates</h2><table><thead><tr><th>Symbol</th><th>Last close</th><th>Surge probability</th><th>Move proxy</th><th>ATR%</th><th>Rel. volume</th><th>PCR dev.</th><th>Fut OI z</th><th>Risk</th></tr></thead><tbody>{table_html}</tbody></table></div>
<div class='card'><h2>Out-of-sample validation</h2><p>Precision: <b>{bundle.validation_precision:.1%}</b> · Sensitivity/Recall: <b>{bundle.validation_recall:.1%}</b> · Specificity: <b>{bundle.validation_specificity:.1%}</b> · Threshold: <b>{bundle.threshold:.2f}</b></p><p class='muted'>These are historical validation statistics, not guarantees. The system can intentionally publish no trade.</p></div>
<div class='card'><h2>Method</h2><p>Official NSE CM + F&O archives → leakage-safe rolling features → deep-learning classifier → walk-forward validation → liquidity/risk gates → 1–5 candidate ranking. <a href='latest.json'>latest.json</a></p></div></body></html>"""
    (site / "index.html").write_text(html)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
