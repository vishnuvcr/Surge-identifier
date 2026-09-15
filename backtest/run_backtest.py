from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.nse_data import load_prices
from src.nse_fo import load_fo
from src.surge_model import build_features, score_latest, train_walk_forward


def fees(buy, sell, c):
    turnover = buy + sell
    exchange = turnover * c['exchange_turnover_rate']
    brokerage = 2 * c['brokerage_per_order']
    stt = sell * c['stt_sell_rate']
    sebi = turnover * c['sebi_turnover_rate']
    stamp = buy * c['stamp_buy_rate']
    gst = c['gst_rate'] * (brokerage + exchange)
    total = brokerage + exchange + stt + sebi + stamp + gst
    return {'brokerage': brokerage, 'exchange': exchange, 'stt': stt, 'sebi': sebi,
            'stamp': stamp, 'gst': gst, 'total': total}


def qty_for_allocation(allocation, raw_open, c):
    entry = raw_open * (1 + c['slippage_per_side'])
    # Conservative cash fit: buy value + fixed brokerage + buy-side variable charges.
    v = c['exchange_turnover_rate'] + c['sebi_turnover_rate'] + c['stamp_buy_rate']
    q = int(max((allocation - c['brokerage_per_order']) / (entry * (1 + v)), 0))
    return q, entry


def execute(row, allocation, c, mode, target):
    raw_open, raw_high, raw_close = map(float, [row.open, row.high, row.close])
    q, entry = qty_for_allocation(allocation, raw_open, c)
    if q <= 0:
        return {'qty': 0, 'net_pnl': 0.0, 'gross_pnl': 0.0, 'fees': 0.0, 'return_pct': 0.0, 'exit_reason': 'no_qty'}
    buy = q * entry
    tp = entry * (1 + target)
    hit = raw_high >= tp
    if mode == 'tp3_close' and hit:
        raw_exit, reason = tp, 'take_profit'
    else:
        raw_exit, reason = raw_close, 'close'
    exit_price = raw_exit * (1 - c['slippage_per_side'])
    sell = q * exit_price
    gross = sell - buy
    f = fees(buy, sell, c)
    net = gross - f['total']
    return {'qty': q, 'entry_price': entry, 'exit_price': exit_price, 'raw_open': raw_open,
            'raw_high': raw_high, 'raw_close': raw_close, 'buy_value': buy, 'sell_value': sell,
            'gross_pnl': gross, 'net_pnl': net, 'fees': f['total'], **{f'fee_{k}': v for k, v in f.items() if k != 'total'},
            'return_pct': net / max(allocation, 1), 'exit_reason': reason}


def build_predictions(feat, prices, cfg):
    days = sorted(pd.to_datetime(feat.date).dt.normalize().unique())
    min_hist = cfg['min_train_days'] + cfg['validation_days']
    eval_days = days[min_hist:]
    if cfg.get('backtest_start'):
        eval_days = [d for d in eval_days if d >= pd.Timestamp(cfg['backtest_start'])]
    if cfg.get('backtest_end'):
        eval_days = [d for d in eval_days if d <= pd.Timestamp(cfg['backtest_end'])]
    predictions = []
    last_period = None
    bundle = None

    for day in eval_days:
        period = pd.Timestamp(day).to_period('M')
        if period != last_period:
            prior = feat[feat.date < pd.Timestamp(day)].copy()
            prior_days = sorted(pd.to_datetime(prior.date).unique())
            prior = prior[prior.date.isin(prior_days[-cfg['training_lookback_sessions']:])]
            bundle = train_walk_forward(prior, validation_days=cfg['validation_days'], seed=cfg['seed'])
            last_period = period

        d = feat[feat.date == day].copy()
        hist = prices[prices.date < day]
        hdays = sorted(hist.date.unique())[-cfg['liquidity_lookback_sessions']:]
        med = hist[hist.date.isin(hdays)].groupby('symbol').turnover.median() / 1e7
        eligible = med[med >= cfg['min_avg_turnover_cr']].index
        d = d[d.symbol.isin(eligible)]
        if d.empty:
            continue
        s = score_latest(bundle, d)
        s = s[(s.signal) & (s.risk_flag != 'low_liquidity')]
        s = s.sort_values(['utility', 'surge_probability'], ascending=False).head(cfg['max_trades_per_day'])
        for _, r in s.iterrows():
            predictions.append({
                'signal_date': day, 'symbol': r.symbol,
                'surge_probability': float(r.surge_probability), 'utility': float(r.utility),
                'atr_pct': float(r.atr_pct), 'relvol20': float(r.relvol20),
                'turnover_cr': float(r.turnover_cr), 'threshold': float(bundle.threshold),
                'validation_precision': float(bundle.validation_precision),
                'validation_recall': float(bundle.validation_recall),
                'validation_specificity': float(bundle.validation_specificity),
            })
    return pd.DataFrame(predictions)


def run(cfg, mode):
    prices = load_prices('data/cache', cfg['lookback_days'])
    prices.date = pd.to_datetime(prices.date).dt.normalize()
    fo = load_fo('data/cache', cfg['lookback_days'])
    if not fo.empty:
        fo.date = pd.to_datetime(fo.date).dt.normalize()
        data = prices.merge(fo, on=['date', 'symbol'], how='left')
    else:
        data = prices
    feat = build_features(data)
    feat.date = pd.to_datetime(feat.date).dt.normalize()
    pred = build_predictions(feat, prices, cfg)
    if pred.empty:
        raise RuntimeError('No qualified historical signals')

    trading_days = sorted(prices.date.unique())
    next_day = {trading_days[i]: trading_days[i+1] for i in range(len(trading_days)-1)}
    pred['entry_date'] = pred.signal_date.map(next_day)
    pred = pred.dropna(subset=['entry_date'])

    by_key = prices.set_index(['date', 'symbol']).sort_index()
    capital = float(cfg['initial_capital'])
    trades, daily = [], []
    for entry_date, sigs in pred.groupby('entry_date', sort=True):
        sigs = sigs.drop_duplicates('symbol').head(cfg['max_trades_per_day'])
        start = capital
        allocation = start / len(sigs)
        for _, sig in sigs.iterrows():
            key = (entry_date, sig.symbol)
            if key not in by_key.index:
                continue
            t = execute(by_key.loc[key], allocation, cfg['costs'], mode, cfg['take_profit_pct'])
            t.update({'signal_date': sig.signal_date, 'entry_date': entry_date, 'symbol': sig.symbol,
                      'surge_probability': sig.surge_probability, 'utility': sig.utility,
                      'atr_pct': sig.atr_pct, 'relvol20': sig.relvol20, 'turnover_cr': sig.turnover_cr,
                      'model_threshold': sig.threshold, 'validation_precision': sig.validation_precision,
                      'validation_recall': sig.validation_recall, 'validation_specificity': sig.validation_specificity})
            trades.append(t)
            capital += t['net_pnl']
        daily.append({'date': entry_date, 'starting_equity': start, 'ending_equity': capital,
                      'daily_pnl': capital - start, 'daily_return_pct': capital / start - 1,
                      'n_trades': len(sigs)})

    trades = pd.DataFrame(trades)
    daily = pd.DataFrame(daily)
    daily['peak_equity'] = daily.ending_equity.cummax()
    daily['drawdown_pct'] = daily.ending_equity / daily.peak_equity - 1
    initial = cfg['initial_capital']
    summary = {
        'mode': mode, 'initial_capital': initial, 'final_equity': float(capital),
        'net_pnl': float(trades.net_pnl.sum()), 'net_return_pct': float(capital / initial - 1),
        'gross_pnl': float(trades.gross_pnl.sum()), 'fees': float(trades.fees.sum()),
        'trade_count': int(len(trades)), 'win_rate': float((trades.net_pnl > 0).mean()),
        'max_drawdown_pct': float(daily.drawdown_pct.min()),
        'profitable_days': int((daily.daily_pnl > 0).sum()), 'losing_days': int((daily.daily_pnl < 0).sum()),
        'avg_trades_per_day': float(daily.n_trades.mean()),
        'anti_lookahead': [
            'Monthly model is refit before each prediction month using only dates strictly before that month.',
            'The final validation window used for threshold selection also ends before the prediction month.',
            'Liquidity uses only the trailing 60 sessions before each signal date.',
            'Signal is EOD and entry is the next trading session open; same-day lookahead is impossible by construction.',
            'Next-day OHLC is used only to calculate trade outcome after the signal is fixed.',
            'Equal capital allocation uses starting equity for that trading day; no leverage is used.',
        ],
        'costs': cfg['costs'],
    }
    return trades, daily, summary


def main():
    cfg = yaml.safe_load((ROOT/'backtest/config.yaml').read_text())
    out = ROOT/'backtest/results'
    out.mkdir(parents=True, exist_ok=True)
    combined = []
    summaries = {}
    for mode in ['close_exit', 'tp3_close']:
        trades, daily, summary = run(cfg, mode)
        trades.to_csv(out/f'trades_{mode}.csv', index=False)
        daily.to_csv(out/f'daily_equity_{mode}.csv', index=False)
        (out/f'summary_{mode}.json').write_text(json.dumps(summary, indent=2, default=str))
        combined.append((mode, summary))
        summaries[mode] = summary
    rows = []
    for mode, s in combined:
        rows.append({'mode': mode, 'final_equity': s['final_equity'], 'net_pnl': s['net_pnl'],
                     'return_pct': s['net_return_pct'], 'gross_pnl': s['gross_pnl'], 'fees': s['fees'],
                     'trades': s['trade_count'], 'win_rate': s['win_rate'], 'max_drawdown_pct': s['max_drawdown_pct'],
                     'profitable_days': s['profitable_days'], 'losing_days': s['losing_days']})
    pd.DataFrame(rows).to_csv(out/'summary.csv', index=False)
    lines = ['# Point-in-Time Intraday Backtest', '',
             'Initial capital: ₹1,00,000; maximum 5 simultaneous trades; equal capital split among selected trades.',
             '', '## Results', '',
             '| Mode | Final equity | Net P&L | Return | Fees | Trades | Win rate | Max DD |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['mode']} | ₹{r['final_equity']:,.2f} | ₹{r['net_pnl']:,.2f} | {r['return_pct']*100:.2f}% | ₹{r['fees']:,.2f} | {r['trades']} | {r['win_rate']*100:.2f}% | {r['max_drawdown_pct']*100:.2f}% |")
    lines += ['', '## Execution assumptions',
              '', '- Entry: next trading day open after the EOD signal.',
              '- Exit 1: same-day close.',
              '- Exit 2: take profit at +3% when the next-day high reaches the target; otherwise close.',
              '- Slippage: 0.05% per side.',
              '- Paytm Money brokerage: ₹20 per executed order; two orders per completed trade.',
              '- Modeled statutory charges: exchange turnover, 0.025% STT on sell, 0.0001% SEBI turnover fee, 0.003% stamp duty on buy, and 18% GST on brokerage + exchange charges.',
              '', '## Anti-lookahead controls']
    for x in summaries['close_exit']['anti_lookahead']:
        lines.append(f'- {x}')
    lines += ['', 'The backtest uses daily OHLC data. Exact intraday timestamp/fill quality cannot be reconstructed from daily bars, so the +3% target mode is deliberately kept as a separate scenario rather than silently treated as the only result.']
    (out/'README.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
