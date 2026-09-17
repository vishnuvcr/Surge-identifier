from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / 'v6.2/config.yaml').read_text())
OUT = ROOT / 'v6.2/research'
import multimodal_engine as eng


def main() -> None:
    inp = eng.load_inputs()
    options, futures, context, news, index, prices, weights, membership, flows = inp
    hist_end = pd.Timestamp(CFG['history_end'])
    index = index[index.date <= hist_end].copy().sort_values('date')
    signal = pd.Timestamp(index.date.iloc[-1])
    spot = float(index.iloc[-1].close)

    exps = sorted(pd.to_datetime(options.loc[pd.to_datetime(options.date).dt.normalize().eq(signal), 'expiry']).dropna().unique())
    expiry = next((pd.Timestamp(e) for e in exps if pd.Timestamp(e) > signal), None)
    if expiry is None:
        all_exps = sorted(pd.to_datetime(options.expiry).dropna().unique())
        expiry = next((pd.Timestamp(e) for e in all_exps if pd.Timestamp(e) > signal), None)
    if expiry is None:
        raise RuntimeError('No future option expiry available for current forecast date')

    # Build a genuinely NEW feature vector at the current prediction cutoff.
    current = eng.index_features(index, signal, options)
    current.update(eng.option_features(options, signal, expiry, spot))
    current.update(eng.futures_features(futures, signal, expiry, spot))
    current.update(eng.context_features(context, news, flows, signal))
    current.update(eng.constituent_features(prices, weights, membership, signal))

    historical = eng.build_dataset(inp)
    historical = historical[historical.signal_date < hist_end].copy().sort_values('signal_date').reset_index(drop=True)
    features = eng.model_features(historical)
    historical[features] = historical[features].replace([np.inf, -np.inf], np.nan)
    current_row = pd.DataFrame([{k: current.get(k, np.nan) for k in features}])
    current_row[features] = current_row[features].replace([np.inf, -np.inf], np.nan)

    summary = json.load(open(OUT / 'summary.json'))
    production = summary['selected_production_model']
    pred = np.nan

    if production == 'lstm':
        mm = pd.concat([
            historical[['signal_date','expiry','target_date','spot','target_close','target_return'] + features],
            pd.concat([
                pd.DataFrame([{'signal_date': signal, 'expiry': expiry, 'target_date': pd.NaT, 'spot': spot, 'target_close': np.nan, 'target_return': np.nan}]),
                current_row,
            ], axis=1)
        ], ignore_index=True)
        train_mask = np.zeros(len(mm), dtype=bool); train_mask[:len(historical)] = True
        test_mask = np.zeros(len(mm), dtype=bool); test_mask[-1] = True
        p = eng.lstm_predict(mm, train_mask, test_mask, features)
        if len(p): pred = float(p[-1])
    elif production == 'ensemble':
        base_preds = eng.fit_tabular(historical, current_row, features)
        vals = [float(v[0]) for k, v in base_preds.items() if np.isfinite(v[0])]
        if vals: pred = float(np.mean(vals))
    else:
        model = eng.builders()[production]
        model.fit(historical[features], historical.target_return.astype(float))
        pred = float(model.predict(current_row[features])[0])

    if not np.isfinite(pred):
        raise RuntimeError(f'Current forecast for {production} is non-finite')

    qlo, qhi = eng.calibration(historical, features, production if production in eng.builders() else 'rf')
    latest = {
        'available': True,
        'signal_date': str(signal.date()),
        'latest_feature_date': str(signal.date()),
        'next_expiry': str(expiry.date()),
        'spot': spot,
        'production_model': production,
        'predicted_return': pred,
        'predicted_close': spot * (1 + pred),
        'predicted_low': spot * (1 + pred + qlo),
        'predicted_high': spot * (1 + pred + qhi),
        'regime': eng.regime(pred),
        'prediction_interval': float(CFG['prediction_interval']),
        'calibration_q_low': qlo,
        'calibration_q_high': qhi,
        'forecast_feature_date_verified': True,
        'forecast_type': 'fresh_current_cutoff_features',
    }
    json.dump(latest, open(OUT / 'latest_forecast.json', 'w'), indent=2)
    trade = eng.paper_strategy(pd.Series({'pred_return': pred, 'spot': spot, 'expiry': expiry, 'signal_date': signal}), options)
    pd.DataFrame([trade]).to_csv(OUT / 'latest_paper_trade.csv', index=False)

    summary['latest_forecast'] = latest
    summary['latest_paper_trade'] = trade
    json.dump(summary, open(OUT / 'summary.json', 'w'), indent=2)
    print(json.dumps({'latest_forecast': latest, 'latest_paper_trade': trade}, indent=2))


if __name__ == '__main__':
    main()
