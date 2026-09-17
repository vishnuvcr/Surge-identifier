from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.cpr_v1_engine import CPRConfig, STRATEGIES, generate_signals, period_levels

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default=str(ROOT / 'data' / 'cpr_v1' / 'intraday'))
    p.add_argument('--output-dir', default=str(ROOT / 'reports' / 'cpr_v1'))
    p.add_argument('--mode', choices=['intraday', 'daily'], default='intraday')
    p.add_argument('--symbols', default='')
    p.add_argument('--capital', type=float, default=100000)
    p.add_argument('--risk-fraction', type=float, default=0.01)
    p.add_argument('--max-position-fraction', type=float, default=0.50)
    p.add_argument('--cost-bps-side', type=float, default=8.0)
    p.add_argument('--slippage-bps-side', type=float, default=3.0)
    p.add_argument('--strategy', default='ALL')
    return p.parse_args()


def load_frames(data_dir: Path, symbols: set[str] | None) -> pd.DataFrame:
    files = sorted(list(data_dir.glob('*.csv')) + list(data_dir.glob('*.parquet')))
    if symbols:
        files = [f for f in files if f.stem.upper() in symbols]
    if not files:
        raise SystemExit(f'No intraday files found in {data_dir}')
    frames = []
    for fp in files:
        df = pd.read_parquet(fp) if fp.suffix == '.parquet' else pd.read_csv(fp)
        rename = {'datetime': 'timestamp', 'date': 'timestamp', 'Datetime': 'timestamp', 'Date': 'timestamp', 'symbol': 'symbol'}
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        if 'symbol' not in df.columns:
            df['symbol'] = fp.stem.upper()
        required = {'timestamp', 'open', 'high', 'low', 'close'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f'{fp.name}: missing {sorted(missing)}')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        for c in ['open', 'high', 'low', 'close', 'volume']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        if 'volume' not in df.columns:
            df['volume'] = 0.0
        frames.append(df[['timestamp','symbol','open','high','low','close','volume']].dropna(subset=['timestamp','open','high','low','close']))
    return pd.concat(frames, ignore_index=True).sort_values(['symbol','timestamp']).reset_index(drop=True)


def next_entry_index(g: pd.DataFrame, i: int) -> int | None:
    j = i + 1
    return j if j < len(g) else None


def backtest_one(signals: pd.DataFrame, capital: float, risk_fraction: float, max_position_fraction: float, cost_bps_side: float, slippage_bps_side: float) -> tuple[pd.DataFrame, dict]:
    trades = []
    total_cost = 2 * (cost_bps_side + slippage_bps_side) / 10000.0
    for symbol, g in signals.groupby('symbol', sort=False):
        g = g.reset_index(drop=True)
        i = 0
        while i < len(g) - 1:
            if int(g.loc[i, 'signal']) == 0:
                i += 1
                continue
            side = int(g.loc[i, 'signal'])
            entry_i = next_entry_index(g, i)
            if entry_i is None:
                break
            entry = float(g.loc[entry_i, 'open'])
            entry_day = g.loc[entry_i, 'session']
            stop = float(g.loc[i, 'd_cpr_low']) if side == 1 else float(g.loc[i, 'd_cpr_high'])
            if not np.isfinite(stop) or stop <= 0:
                i += 1
                continue
            target = float(g.loc[i, 'd_r1']) if side == 1 else float(g.loc[i, 'd_s1'])
            if not np.isfinite(target):
                i += 1
                continue
            risk_per_share = abs(entry - stop) / max(entry, 1e-9)
            if risk_per_share <= 0:
                i += 1
                continue
            notional_fraction = min(max_position_fraction, risk_fraction / risk_per_share)
            exit_i = entry_i
            reason = 'eod'
            while exit_i < len(g) and g.loc[exit_i, 'session'] == entry_day:
                hi = float(g.loc[exit_i, 'high']); lo = float(g.loc[exit_i, 'low'])
                if side == 1:
                    if lo <= stop:
                        exit_price = stop; reason = 'stop'; break
                    if hi >= target:
                        exit_price = target; reason = 'target'; break
                else:
                    if hi >= stop:
                        exit_price = stop; reason = 'stop'; break
                    if lo <= target:
                        exit_price = target; reason = 'target'; break
                exit_price = float(g.loc[exit_i, 'close'])
                exit_i += 1
            gross = side * (exit_price / entry - 1.0)
            net = gross - total_cost
            trades.append({'symbol': symbol, 'signal_time': g.loc[i, 'timestamp'], 'entry_time': g.loc[entry_i, 'timestamp'], 'exit_time': g.loc[min(exit_i, len(g)-1), 'timestamp'], 'side': 'LONG' if side == 1 else 'SHORT', 'entry': entry, 'exit': exit_price, 'gross_return': gross, 'net_return': net, 'position_fraction': notional_fraction, 'exit_reason': reason, 'strategy_id': g.loc[i, 'strategy_id']})
            i = max(exit_i, entry_i + 1)
    td = pd.DataFrame(trades)
    if td.empty:
        return td, {'trade_count': 0, 'final_capital': capital, 'net_return': 0.0, 'max_drawdown': 0.0}
    equity = float(capital); peak = equity; curve=[]
    for _, t in td.sort_values('entry_time').iterrows():
        pnl = equity * float(t.position_fraction) * float(t.net_return)
        equity += pnl; peak = max(peak, equity)
        curve.append(equity / peak - 1.0)
    r = td.net_return.astype(float)
    gross_profit = td.loc[r > 0, 'net_return'].sum(); gross_loss = -td.loc[r < 0, 'net_return'].sum()
    weekly = td.assign(week=pd.to_datetime(td.entry_time).dt.to_period('W')).groupby('week').net_return.apply(lambda s: float((1+s).prod()-1))
    monthly = td.assign(month=pd.to_datetime(td.entry_time).dt.to_period('M')).groupby('month').net_return.apply(lambda s: float((1+s).prod()-1))
    metrics = {
        'trade_count': int(len(td)),
        'long_trades': int((td.side == 'LONG').sum()),
        'short_trades': int((td.side == 'SHORT').sum()),
        'hit_rate': float((r > 0).mean()),
        'profit_factor': float(gross_profit / gross_loss) if gross_loss > 0 else None,
        'final_capital': float(equity),
        'net_return': float(equity / capital - 1.0),
        'max_drawdown': float(min(curve)),
        'weekly_avg_return': float(weekly.mean()) if len(weekly) else 0.0,
        'weekly_median_return': float(weekly.median()) if len(weekly) else 0.0,
        'positive_week_fraction': float((weekly > 0).mean()) if len(weekly) else 0.0,
        'monthly_avg_return': float(monthly.mean()) if len(monthly) else 0.0,
        'monthly_median_return': float(monthly.median()) if len(monthly) else 0.0,
        'positive_month_fraction': float((monthly > 0).mean()) if len(monthly) else 0.0,
        'avg_trade_return': float(r.mean()),
        'median_trade_return': float(r.median()),
        'worst_trade': float(r.min()),
    }
    return td, metrics


def main():
    a = parse_args()
    symbols = {s.strip().upper() for s in a.symbols.split(',') if s.strip()} or None
    data = load_frames(Path(a.data_dir), symbols)
    cfg = CPRConfig()
    prepared = period_levels(data, cfg)
    strategies = STRATEGIES if a.strategy.upper() == 'ALL' else [a.strategy]
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    summary = {'mode': a.mode, 'symbols': sorted(data.symbol.astype(str).unique()), 'strategies': {}}
    for strategy in strategies:
        sig = generate_signals(prepared, strategy, cfg)
        trades, metrics = backtest_one(sig, a.capital, a.risk_fraction, a.max_position_fraction, a.cost_bps_side, a.slippage_bps_side)
        trades.to_csv(out / f'{strategy}_trades.csv', index=False)
        summary['strategies'][strategy] = metrics
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
