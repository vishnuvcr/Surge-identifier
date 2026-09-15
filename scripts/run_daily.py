from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import yaml

from src.nse_data import load_prices, IST
from src.surge_model import build_features, train_walk_forward, score_latest

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    data = load_prices(str(ROOT / "data" / "cache"), int(cfg["lookback_days"]))
    data["date"] = pd.to_datetime(data["date"])

    # Liquidity universe: remove instruments whose typical turnover is too small, then
    # retain a broad cross-section so the model can discover unusual winners.
    med_turn = data.groupby("symbol")["turnover"].median() / 1e7
    eligible = med_turn[med_turn >= float(cfg["min_avg_turnover_cr"])].index
    data = data[data["symbol"].isin(eligible)].copy()

    feats = build_features(data)
    # Cap training rows per day by turnover, but score the full eligible universe.
    keep_n = int(cfg["max_stocks"])
    ranked = feats.assign(_turn=feats["turnover"]).sort_values(["date", "_turn"], ascending=[True, False])
    train = ranked.groupby("date", group_keys=False).head(keep_n).drop(columns=["_turn"])

    bundle = train_walk_forward(train, int(cfg["validation_days"]), int(cfg["seed"]))
    latest_date = feats["date"].max()
    latest = feats[feats["date"] == latest_date].copy()
    scores = score_latest(bundle, latest)
    candidates = scores[
        (scores["signal"]) &
        (scores["surge_probability"] >= float(cfg["min_candidate_probability"])) &
        (scores["risk_flag"] != "low_liquidity")
    ].head(int(cfg["max_candidates"]))

    out_cols = ["symbol", "close", "surge_probability", "expected_move_proxy", "atr_pct", "relvol20", "turnover_cr", "risk_flag", "gap"]
    candidates = candidates[out_cols].copy()
    candidates.columns = [
        "symbol", "last_close", "surge_probability", "expected_move_proxy", "atr_pct", "relative_volume", "turnover_cr", "risk_flag", "gap"
    ]
    records = []
    for row in candidates.to_dict("records"):
        for k, v in list(row.items()):
            if hasattr(v, "item"):
                row[k] = v.item()
        records.append(row)

    stamp = datetime.now(IST)
    report = {
        "asof_date": str(latest_date.date()),
        "generated_at_ist": stamp.isoformat(),
        "event_definition": "Next session intraday high >= +3% from next session open",
        "universe": "NSE CM EQ series with sufficient median turnover",
        "candidates": records,
        "model": {
            "type": "PyTorch MLP with LayerNorm/GELU/Dropout",
            "features": 28,
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
        table_html = '<tr><td colspan="7">NO QUALIFIED TRADE — model confidence/liquidity gate not satisfied.</td></tr>'
    else:
        rows = []
        for r in records:
            rows.append(
                "<tr>"
                f"<td>{r['symbol']}</td><td>{r['last_close']:.2f}</td>"
                f"<td>{r['surge_probability']:.1%}</td><td>{r['expected_move_proxy']:.1%}</td>"
                f"<td>{r['atr_pct']:.1%}</td><td>{r['relative_volume']:.2f}x</td>"
                f"<td>{r['risk_flag']}</td></tr>"
            )
        table_html = "".join(rows)

    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>NSE Surge Identifier</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 18px;background:#0b1020;color:#e9eef8}}.card{{background:#151c30;border:1px solid #293554;border-radius:14px;padding:20px;margin:14px 0}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #2b3550;text-align:left}}.muted{{color:#9eabc4}}.big{{font-size:28px;font-weight:700}}</style></head>
<body><h1>NSE Surge Identifier</h1>
<div class='card'><div class='big'>{date_key}</div><div class='muted'>Generated {stamp.strftime('%Y-%m-%d %H:%M IST')} · Event: next session high ≥ +3% from open</div></div>
<div class='card'><h2>Next-session candidates</h2><table><thead><tr><th>Symbol</th><th>Last close</th><th>Surge probability</th><th>Move proxy</th><th>ATR%</th><th>Rel. volume</th><th>Risk</th></tr></thead><tbody>{table_html}</tbody></table></div>
<div class='card'><h2>Out-of-sample validation</h2><p>Precision: <b>{bundle.validation_precision:.1%}</b> · Sensitivity/Recall: <b>{bundle.validation_recall:.1%}</b> · Specificity: <b>{bundle.validation_specificity:.1%}</b> · Threshold: <b>{bundle.threshold:.2f}</b></p><p class='muted'>These are historical validation statistics, not guarantees. The system is deliberately allowed to issue no trade when the model does not meet its gates.</p></div>
<div class='card'><h2>Method</h2><p>Official NSE CM bhavcopy archive → leakage-safe rolling features → PyTorch MLP → walk-forward validation → probability/liquidity/risk filtering. See <a href='latest.json'>latest.json</a> for machine-readable output.</p></div></body></html>"""
    (site / "index.html").write_text(html)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
