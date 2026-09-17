from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from scripts.run_cpr_multimode import args, load_frames, intraday_trades, swing_trades, btst_trades, metrics
from src.cpr_multimode_v2 import CPRMultiModeConfig, STRATEGIES, prepare_context, generate_intraday, generate_daily

def main():
    a=args(); symbols={s.strip().upper() for s in a.symbols.split(',') if s.strip()}
    raw=load_frames(Path(a.data_dir),symbols); cfg=CPRMultiModeConfig(max_swing_days=a.max_swing_days)
    x,daily=prepare_context(raw,cfg); strategies=STRATEGIES if a.strategy.upper()=='ALL' else [a.strategy]
    modes=['intraday','swing','btst'] if a.mode=='all' else [a.mode]; out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    roundtrip=2*(a.cost_bps_side+a.slippage_bps_side)/10000.0; summary={'capital':a.capital,'symbols':sorted(symbols),'cost_roundtrip':roundtrip,'modes':{}}; rows=[]
    for mode in modes:
        summary['modes'][mode]={}
        for strategy in strategies:
            if mode=='intraday':
                sig=generate_intraday(x,strategy,cfg); sig['strategy_id']=strategy
                td,invalid=intraday_trades(sig,a.capital,a.risk_fraction,a.max_position_fraction,roundtrip,len(symbols))
            else:
                sig=generate_daily(daily,strategy,mode,cfg); sig['strategy_id']=strategy
                if mode=='swing': td,invalid=swing_trades(sig,a.capital,a.risk_fraction,a.max_position_fraction,roundtrip,len(symbols),a.max_swing_days)
                else: td,invalid=btst_trades(sig,raw,roundtrip,a.risk_fraction,a.max_position_fraction,len(symbols),a.capital)
            if not td.empty: td.to_csv(out/f'{mode}__{strategy}__trades.csv',index=False)
            m=metrics(td,a.capital); m['invalid_geometry_signals']=invalid; m['strategy_id']=strategy; summary['modes'][mode][strategy]=m; rows.append({'mode':mode,**m})
            print(json.dumps({'mode':mode,'strategy':strategy,**m},default=str))
    pd.DataFrame(rows).to_csv(out/'summary.csv',index=False); (out/'summary.json').write_text(json.dumps(summary,indent=2,default=str)); print(json.dumps(summary,indent=2,default=str))
if __name__=='__main__': main()
