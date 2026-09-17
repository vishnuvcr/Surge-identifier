from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.cpr_multimode_v2 import CPRMultiModeConfig, STRATEGIES, prepare_context, generate_intraday, generate_daily

ROOT = Path(__file__).resolve().parents[1]


def args():
    p = argparse.ArgumentParser(description='CPR multi-mode: intraday, swing, BTST')
    p.add_argument('--data-dir', default=str(ROOT/'data/cpr_v1/intraday'))
    p.add_argument('--output-dir', default=str(ROOT/'reports/cpr_multimode'))
    p.add_argument('--symbols', default='RELIANCE,HDFCBANK,ICICIBANK,INFY,TCS')
    p.add_argument('--capital', type=float, default=100000)
    p.add_argument('--risk-fraction', type=float, default=0.01)
    p.add_argument('--max-position-fraction', type=float, default=0.50)
    p.add_argument('--cost-bps-side', type=float, default=8.0)
    p.add_argument('--slippage-bps-side', type=float, default=3.0)
    p.add_argument('--max-swing-days', type=int, default=10)
    p.add_argument('--mode', choices=['all','intraday','swing','btst'], default='all')
    p.add_argument('--strategy', default='ALL')
    return p.parse_args()


def load_frames(path: Path, symbols: set[str]) -> pd.DataFrame:
    files = sorted(list(path.glob('*.parquet')) + list(path.glob('*.csv')))
    if symbols:
        files = [f for f in files if f.stem.upper() in symbols]
    if not files:
        raise SystemExit(f'No data files found in {path}')
    rows=[]
    for fp in files:
        df = pd.read_parquet(fp) if fp.suffix=='.parquet' else pd.read_csv(fp)
        rename={c:'timestamp' for c in ['datetime','date','Datetime','Date'] if c in df.columns}
        if rename: df=df.rename(columns=rename)
        if 'symbol' not in df.columns: df['symbol']=fp.stem.upper()
        need={'timestamp','open','high','low','close'}
        miss=need-set(df.columns)
        if miss: raise ValueError(f'{fp.name}: missing {sorted(miss)}')
        for c in ['open','high','low','close','volume']:
            if c in df.columns: df[c]=pd.to_numeric(df[c],errors='coerce')
        if 'volume' not in df.columns: df['volume']=0.0
        df.timestamp=pd.to_datetime(df.timestamp,errors='coerce')
        rows.append(df[['timestamp','symbol','open','high','low','close','volume']].dropna(subset=['timestamp','open','high','low','close']))
    out=pd.concat(rows,ignore_index=True)
    out['symbol']=out.symbol.astype(str).str.upper()
    return out.drop_duplicates(['symbol','timestamp']).sort_values(['symbol','timestamp']).reset_index(drop=True)


def valid_geometry(side: int, entry: float, stop: float, target: float):
    if not all(np.isfinite(v) for v in [entry,stop,target]) or entry<=0: return False
    if side==1: return stop < entry < target
    if side==-1: return target < entry < stop
    return False


def trade_row(mode, strategy, symbol, signal_time, entry_time, exit_time, side, entry, stop, target, exit_price, reason, position_fraction):
    gross = side*(exit_price/entry-1.0)
    return {
        'mode':mode,'strategy_id':strategy,'symbol':symbol,
        'signal_time':signal_time,'entry_time':entry_time,'exit_time':exit_time,
        'side':'LONG' if side==1 else 'SHORT','entry':entry,'stop':stop,'target':target,
        'exit':exit_price,'gross_return':gross,'exit_reason':reason,
        'position_fraction':position_fraction,
    }


def intraday_trades(sig, capital, risk_fraction, max_position_fraction, cost_roundtrip, symbol_count):
    trades=[]; invalid=0
    alloc_cap=min(max_position_fraction, 1.0/max(symbol_count,1))
    for symbol,g in sig.groupby('symbol',sort=False):
        g=g.reset_index(drop=True); i=0
        while i < len(g)-1:
            s=int(g.loc[i,'signal'])
            if s==0: i+=1; continue
            entry_i=i+1
            if entry_i>=len(g) or g.loc[entry_i,'session']!=g.loc[i,'session']:
                i+=1; continue
            entry=float(g.loc[entry_i,'open']); stop=float(g.loc[i,'stop_level']); target=float(g.loc[i,'target_level'])
            if not valid_geometry(s,entry,stop,target): invalid+=1; i+=1; continue
            risk_pct=abs(entry-stop)/entry
            pos=min(alloc_cap,risk_fraction/max(risk_pct,1e-9))
            entry_day=g.loc[entry_i,'session']; exit_i=entry_i; exit_price=None; reason='eod'
            while exit_i<len(g) and g.loc[exit_i,'session']==entry_day:
                o=float(g.loc[exit_i,'open']); hi=float(g.loc[exit_i,'high']); lo=float(g.loc[exit_i,'low'])
                if s==1:
                    if o<=stop: exit_price=o; reason='stop_gap'; break
                    if o>=target: exit_price=target; reason='target_gap'; break
                    if lo<=stop: exit_price=stop; reason='stop'; break
                    if hi>=target: exit_price=target; reason='target'; break
                else:
                    if o>=stop: exit_price=o; reason='stop_gap'; break
                    if o<=target: exit_price=target; reason='target_gap'; break
                    if hi>=stop: exit_price=stop; reason='stop'; break
                    if lo<=target: exit_price=target; reason='target'; break
                exit_price=float(g.loc[exit_i,'close'])
                exit_i+=1
            if exit_price is None: break
            t=trade_row('intraday',g.loc[i,'strategy_id'],symbol,g.loc[i,'timestamp'],g.loc[entry_i,'timestamp'],g.loc[min(exit_i,len(g)-1),'timestamp'],s,entry,stop,target,exit_price,reason,pos)
            t['net_return']=t['gross_return']-cost_roundtrip
            trades.append(t); i=max(exit_i,entry_i+1)
    return pd.DataFrame(trades), invalid


def swing_trades(sig, capital, risk_fraction, max_position_fraction, cost_roundtrip, symbol_count, max_days):
    trades=[]; invalid=0; alloc_cap=min(max_position_fraction,1.0/max(symbol_count,1))
    for symbol,g in sig.groupby('symbol',sort=False):
        g=g.reset_index(drop=True); i=0
        while i<len(g)-1:
            s=int(g.loc[i,'signal'])
            if s==0: i+=1; continue
            entry_i=i+1
            entry=float(g.loc[entry_i,'open']); stop=float(g.loc[i,'stop_level']); target=float(g.loc[i,'target_level'])
            if not valid_geometry(s,entry,stop,target): invalid+=1; i+=1; continue
            risk_pct=abs(entry-stop)/entry; pos=min(alloc_cap,risk_fraction/max(risk_pct,1e-9))
            exit_price=None; exit_i=entry_i; reason='max_hold'
            last_i=min(len(g)-1,entry_i+max_days-1)
            while exit_i<=last_i:
                o=float(g.loc[exit_i,'open']); hi=float(g.loc[exit_i,'high']); lo=float(g.loc[exit_i,'low'])
                if s==1:
                    if o<=stop: exit_price=o; reason='stop_gap'; break
                    if o>=target: exit_price=target; reason='target_gap'; break
                    if lo<=stop: exit_price=stop; reason='stop'; break
                    if hi>=target: exit_price=target; reason='target'; break
                else:
                    if o>=stop: exit_price=o; reason='stop_gap'; break
                    if o<=target: exit_price=target; reason='target_gap'; break
                    if hi>=stop: exit_price=stop; reason='stop'; break
                    if lo<=target: exit_price=target; reason='target'; break
                exit_price=float(g.loc[exit_i,'close']); exit_i+=1
            if exit_price is None: break
            t=trade_row('swing',g.loc[i,'strategy_id'],symbol,g.loc[i,'session'],g.loc[entry_i,'session'],g.loc[exit_i,'session'],s,entry,stop,target,exit_price,reason,pos)
            t['net_return']=t['gross_return']-cost_roundtrip
            trades.append(t); i=max(exit_i+1,entry_i+1)
    return pd.DataFrame(trades), invalid


def btst_trades(daily_sig, intraday, cost_roundtrip, risk_fraction, max_position_fraction, symbol_count, capital):
    trades=[]; invalid=0; alloc_cap=min(max_position_fraction,1.0/max(symbol_count,1))
    for symbol,g in daily_sig.groupby('symbol',sort=False):
        g=g.reset_index(drop=True)
        bars=intraday[intraday.symbol==symbol].copy(); bars['session']=bars.timestamp.dt.normalize()
        sessions=sorted(bars.session.drop_duplicates())
        for i,row in g.iterrows():
            if int(row.signal)!=1: continue
            day=row.session
            same=bars[bars.session==day]
            if same.empty: continue
            idx=sessions.index(day) if day in sessions else -1
            if idx<0 or idx+1>=len(sessions): continue
            next_day=sessions[idx+1]; nxt=bars[bars.session==next_day].sort_values('timestamp')
            entry=float(same.sort_values('timestamp').iloc[-1].close)
            stop=float(row.stop_level); target=float(row.target_level)
            if not valid_geometry(1,entry,stop,target): invalid+=1; continue
            risk_pct=abs(entry-stop)/entry; pos=min(alloc_cap,risk_fraction/max(risk_pct,1e-9))
            exit_price=None; reason='eod'
            for _,bar in nxt.iterrows():
                o=float(bar.open); hi=float(bar.high); lo=float(bar.low)
                if o<=stop: exit_price=o; reason='stop_gap'; break
                if o>=target: exit_price=target; reason='target_gap'; break
                if lo<=stop: exit_price=stop; reason='stop'; break
                if hi>=target: exit_price=target; reason='target'; break
            if exit_price is None: exit_price=float(nxt.iloc[-1].close)
            trades.append(trade_row('btst',row.strategy_id,symbol,day,same.iloc[-1].timestamp,nxt.iloc[-1].timestamp,1,entry,stop,target,exit_price,reason,pos))
            trades[-1]['net_return']=trades[-1]['gross_return']-cost_roundtrip
    return pd.DataFrame(trades), invalid


def metrics(td: pd.DataFrame, capital: float):
    if td.empty:
        return {'trade_count':0,'final_capital':capital,'net_return':0.0,'hit_rate':0.0,'profit_factor':None,'max_drawdown':0.0,'avg_trade_return':0.0,'median_trade_return':0.0,'worst_trade':0.0,'target_exit_rate':0.0}
    td=td.sort_values('entry_time').copy()
    r=td.net_return.astype(float)
    eq=capital; peak=capital; dd=[]
    for _,t in td.iterrows():
        eq += eq*float(t.position_fraction)*float(t.net_return)
        peak=max(peak,eq); dd.append(eq/peak-1.0)
    gp=r[r>0].sum(); gl=-r[r<0].sum()
    return {
        'trade_count':int(len(td)),
        'long_trades':int((td.side=='LONG').sum()),
        'short_trades':int((td.side=='SHORT').sum()),
        'hit_rate':float((r>0).mean()),
        'profit_factor':float(gp/gl) if gl>0 else None,
        'final_capital':float(eq),
        'net_return':float(eq/capital-1.0),
        'max_drawdown':float(min(dd)),
        'avg_trade_return':float(r.mean()),
        'median_trade_return':float(r.median()),
        'worst_trade':float(r.min()),
        'target_exit_rate':float(td.exit_reason.str.startswith('target').mean()),
    }


def main():
    a=args(); symbols={s.strip().upper() for s in a.symbols.split(',') if s.strip()}
    raw=load_frames(Path(a.data_dir),symbols)
    cfg=CPRMultiModeConfig(max_swing_days=a.max_swing_days)
    x,daily=prepare_context(raw,cfg)
    strategies=STRATEGIES if a.strategy.upper()=='ALL' else [a.strategy]
    modes=['intraday','swing','btst'] if a.mode=='all' else [a.mode]
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    roundtrip=2*(a.cost_bps_side+a.slippage_bps_side)/10000.0
    summary={'capital':a.capital,'symbols':sorted(symbols),'cost_roundtrip':roundtrip,'modes':{}}
    rows=[]
    for mode in modes:
        summary['modes'][mode]={}
        for strategy in strategies:
            if mode=='intraday':
                sig=generate_intraday(x,strategy,cfg); td,invalid=intraday_trades(sig,a.capital,a.risk_fraction,a.max_position_fraction,roundtrip,len(symbols))
            else:
                sig=generate_daily(daily,strategy,mode,cfg)
                if mode=='swing': td,invalid=swing_trades(sig,a.capital,a.risk_fraction,a.max_position_fraction,roundtrip,len(symbols),a.max_swing_days)
                else: td,invalid=btst_trades(sig,raw,roundtrip,a.risk_fraction,a.max_position_fraction,len(symbols),a.capital)
            if not td.empty: td.to_csv(out/f'{mode}__{strategy}__trades.csv',index=False)
            m=metrics(td,a.capital); m['invalid_geometry_signals']=invalid; m['strategy_id']=strategy; summary['modes'][mode][strategy]=m
            rows.append({'mode':mode,**m})
            print(json.dumps({'mode':mode,'strategy':strategy,**m},default=str))
    pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))


if __name__=='__main__': main()
