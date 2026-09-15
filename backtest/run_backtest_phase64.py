from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.run_backtest_phase61 import execute, liquidity_table, load_events, summary
from src.moe_engine import add_forward_targets, add_point_in_time_events
from src.moe_engine_phase64 import (
    add_cpr_features,
    score_experts,
    select_top_predictions,
    train_experts,
    train_regime_router,
)
from src.nse_data import load_prices
from src.nse_fo import load_fo
from src.surge_model import build_features


def log(message: str, started: float) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message} | elapsed={time.perf_counter() - started:.1f}s", flush=True)


def main() -> None:
    started = time.perf_counter()
    cfg = yaml.safe_load((ROOT / 'backtest/config_phase64.yaml').read_text())
    out = Path(os.environ.get('PHASE64_OUTPUT_DIR', ROOT / 'backtest/results/phase64'))
    out.mkdir(parents=True, exist_ok=True)

    log('=== PHASE 6.4 PARALLEL REGIME-WEIGHTED MoE: START ===', started)
    log(f"Config: lookback={cfg['lookback_days']}d, train_lookback={cfg['training_lookback_sessions']} sessions, targets={cfg['target_levels']}, portfolios={cfg['portfolio_sizes']}", started)

    prices = load_prices('data/cache', int(cfg['lookback_days']))
    prices.date = pd.to_datetime(prices.date).dt.normalize()
    log(f'Cash data loaded: {len(prices):,} rows, {prices.date.min().date()} to {prices.date.max().date()}', started)

    fo = load_fo('data/cache', int(cfg['lookback_days']))
    if not fo.empty:
        fo.date = pd.to_datetime(fo.date).dt.normalize()
        data = prices.merge(fo, on=['date', 'symbol'], how='left')
    else:
        data = prices
    log(f'F&O data loaded: {len(fo):,} rows; merged={len(data):,}', started)

    feat = build_features(data)
    feat = add_cpr_features(feat)
    news, corp = load_events(cfg)
    feat = add_point_in_time_events(feat, news, corp)
    feat = add_forward_targets(feat)
    feat.date = pd.to_datetime(feat.date).dt.normalize()
    log(f'Features/targets ready: {len(feat):,} rows; news={len(news):,}; corp={len(corp):,}', started)

    days = pd.DatetimeIndex(sorted(feat.date.unique()))
    eval_days = days[int(cfg['min_train_days']):]
    if cfg.get('backtest_start'):
        eval_days = eval_days[eval_days >= pd.Timestamp(cfg['backtest_start'])]
    if cfg.get('backtest_end'):
        eval_days = eval_days[eval_days <= pd.Timestamp(cfg['backtest_end'])]
    next_day = dict(zip(days[:-1], days[1:]))
    prices_idx = prices.set_index(['date', 'symbol']).sort_index()
    price_days = set(prices_idx.index.get_level_values(0).unique())

    eligible_set = liquidity_table(prices, int(cfg['liquidity_lookback_sessions']), float(cfg['min_avg_turnover_cr']))
    eligible_by_day = {}
    for day, sym in eligible_set:
        eligible_by_day.setdefault(day, set()).add(sym)
    log(f'Universe ready: {len(eval_days)} OOS sessions; {len(eligible_set):,} eligible day-symbol pairs', started)

    sizes = [int(x) for x in cfg['portfolio_sizes']]
    states = {ps: {'capital': float(cfg['initial_capital']), 'trades': [], 'daily': []} for ps in sizes}
    bundles = {}
    router = {}
    last_period = None
    diagnostics = {
        'phase': '6.4',
        'signal_days': 0,
        'universe_rows': 0,
        'expert_prediction_rows': 0,
        'expert_top_rows': 0,
        'selected_rows': 0,
        'trade_days': 0,
        'no_trade_days': 0,
        'expert_refits': [],
        'routing_weights_by_regime': {},
        'design': 'all experts score the liquid universe in parallel; regime router weights expert predictions; no predictive eligibility gates; CPR/Camarilla is an expert',
    }

    for i, signal_day in enumerate(eval_days, 1):
        signal_day = pd.Timestamp(signal_day)
        diagnostics['signal_days'] += 1
        period = signal_day.to_period('M')
        if period != last_period:
            refit_started = time.perf_counter()
            prior = feat[(feat.date < signal_day) & feat.label_date.notna() & (feat.label_date < signal_day)].copy()
            train_days = sorted(prior.date.unique())[-int(cfg['training_lookback_sessions']):]
            prior = prior[prior.date.isin(train_days)].copy()
            if len(train_days) >= int(cfg['min_train_days']):
                bundles = train_experts(prior, tuple(cfg['target_levels']), float(cfg['stop_loss_pct']), int(cfg['seed']), int(cfg['min_positive_labels']))
                if bundles:
                    router = train_regime_router(prior, bundles, int(cfg['seed']))
                else:
                    router = {}
            else:
                bundles, router = {}, {}
            diagnostics['expert_refits'].append({'period': str(period), 'train_days': len(train_days), 'experts': sorted(bundles), 'router_regimes': sorted(router)})
            log(f"REFIT {period}: train_days={len(train_days)} experts={len(bundles)} regimes={sorted(router)} duration={time.perf_counter()-refit_started:.1f}s", started)
            last_period = period

        entry_day = next_day.get(signal_day)
        symbols = eligible_by_day.get(signal_day.date() if hasattr(signal_day, 'date') else signal_day, set())
        if not symbols:
            # liquidity_table dates are pandas timestamps in current implementation
            symbols = eligible_by_day.get(signal_day, set())
        universe = feat[(feat.date == signal_day) & feat.symbol.isin(symbols)].copy()
        diagnostics['universe_rows'] += len(universe)

        selected = pd.DataFrame()
        router_snapshot = {}
        if bundles and entry_day is not None and entry_day in price_days and not universe.empty:
            scored = score_experts(bundles, universe, tuple(cfg['target_levels']))
            diagnostics['expert_prediction_rows'] += len(scored)
            selected, router_snapshot = select_top_predictions(scored, router, int(cfg['max_per_expert']), max(sizes))
            diagnostics['expert_top_rows'] += min(len(scored), len(bundles) * int(cfg['max_per_expert'])) if len(scored) else 0
            diagnostics['selected_rows'] += len(selected)
            diagnostics['routing_weights_by_regime'].update({str(k): v for k, v in router_snapshot.items()})

        if selected.empty:
            diagnostics['no_trade_days'] += 1
            for ps in sizes:
                st = states[ps]
                st['daily'].append({'signal_date': signal_day, 'entry_date': entry_day, 'executed': 0, 'starting_equity': st['capital'], 'ending_equity': st['capital'], 'daily_pnl': 0.0, 'daily_return_pct': 0.0, 'selected_count': 0})
        else:
            diagnostics['trade_days'] += 1
            for ps in sizes:
                st = states[ps]
                chosen = selected.head(min(ps, int(cfg['max_candidates_per_day']))).copy()
                start_equity = st['capital']
                allocation = start_equity / max(len(chosen), 1)
                pnl = 0.0
                for _, pick in chosen.iterrows():
                    row = prices_idx.loc[(entry_day, pick.symbol)]
                    # Risk controls determine target/stop execution, not whether the top-ranked
                    # prediction is eligible. Every selected expert candidate is tradable.
                    result = execute(row, allocation, cfg['costs'], float(pick.target), float(cfg['stop_loss_pct']))
                    result.update(pick.to_dict())
                    result.update({'signal_date': signal_day, 'entry_date': entry_day, 'portfolio_size': ps})
                    st['trades'].append(result)
                    pnl += result['net_pnl']
                st['capital'] += pnl
                st['daily'].append({'signal_date': signal_day, 'entry_date': entry_day, 'executed': 1, 'starting_equity': start_equity, 'ending_equity': st['capital'], 'daily_pnl': pnl, 'daily_return_pct': pnl / max(start_equity, 1.0), 'selected_count': len(chosen)})
            log(f"TRADE DAY {i}/{len(eval_days)}: signal={signal_day.date()} entry={entry_day.date()} selected={len(selected)} best={selected.iloc[0].symbol if not selected.empty else 'NA'}", started)

        if i == 1 or i % 20 == 0 or i == len(eval_days):
            snapshot = ', '.join(f"P{ps}=Rs.{states[ps]['capital']:,.0f}" for ps in sizes)
            log(f"PROGRESS {i}/{len(eval_days)} trade_days={diagnostics['trade_days']} no_trade_days={diagnostics['no_trade_days']} universe={diagnostics['universe_rows']:,} | {snapshot}", started)

    summaries = {str(ps): summary(states[ps], cfg) for ps in sizes}
    result = {'phase': '6.4', 'engine': 'parallel_experts_regime_weighted_ensemble', 'config': cfg, 'diagnostics': diagnostics, 'summaries': summaries, 'runtime_seconds': time.perf_counter() - started}
    (out / 'summary.json').write_text(json.dumps(result, indent=2, default=str))
    for ps in sizes:
        pd.DataFrame(states[ps]['trades']).to_csv(out / f'trades_p{ps}.csv', index=False)
        pd.DataFrame(states[ps]['daily']).to_csv(out / f'daily_p{ps}.csv', index=False)
    log(f"=== PHASE 6.4 COMPLETE === runtime={time.perf_counter()-started:.1f}s trade_days={diagnostics['trade_days']} selected_rows={diagnostics['selected_rows']}", started)
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == '__main__':
    main()
