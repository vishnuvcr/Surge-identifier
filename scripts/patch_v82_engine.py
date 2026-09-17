from pathlib import Path
import re

ENGINE = Path('v8/deep_iron_condor_engine.py')
INSPECTOR = Path('v8/inspector.py')

def replace_block(src, start_pat, end_pat, replacement):
    pat = re.compile(start_pat + r'.*?(?=^' + end_pat + r')', re.S | re.M)
    out, n = pat.subn(replacement.rstrip() + '\n\n', src, count=1)
    if n != 1:
        raise RuntimeError(f'patch block not found: {start_pat}')
    return out

def patch_engine():
    s = ENGINE.read_text()
    model = '''class ICTransformer(nn.Module):
    """Joint coherent quantile head with a shared range anchor.

    One shared low-end anchor is followed by positive increments.  This
    structurally guarantees low(q) <= close(q) <= high(q) and monotonicity
    across the configured quantiles.
    """
    def __init__(self, n_features):
        super().__init__()
        c = CFG['model']; d = int(c['d_model'])
        self.inp = nn.Linear(n_features, d)
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=int(c['heads']),
            dim_feedforward=4*d, dropout=float(c['dropout']), batch_first=True,
            norm_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(enc, num_layers=int(c['layers']))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 21)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h = self.norm(self.enc(self.inp(x))[:, -1])
        raw = self.head(h).view(-1, 3, len(Q))
        anchor = raw[:, 0, :1]
        low_inc = torch.nn.functional.softplus(raw[:, 0, 1:]) * 0.02 + 1e-5
        close_gap = torch.nn.functional.softplus(raw[:, 1, :]) * 0.02 + 1e-5
        high_gap = torch.nn.functional.softplus(raw[:, 2, :]) * 0.02 + 1e-5
        low = anchor + torch.cat([torch.zeros_like(low_inc[:, :1]), torch.cumsum(low_inc, dim=-1)], dim=-1)
        close = low + torch.cumsum(close_gap, dim=-1)
        high = close + torch.cumsum(high_gap, dim=-1)
        return torch.stack([low, close, high], dim=1)
'''
    s = replace_block(s, r'^class ICTransformer\(nn\.Module\):', r'^def pinball\(', model)

    fit = '''def fit_model(X, meta):
    y = meta[['path_low_return', 'target_return', 'path_high_return']].to_numpy(np.float32)
    xm = X.reshape(-1, X.shape[-1]).mean(0)
    xs = X.reshape(-1, X.shape[-1]).std(0)
    xs = np.where(xs < 1e-6, 1, xs)
    Xn = np.clip((X - xm) / xs, -8, 8).astype(np.float32)
    n = len(meta); cut = max(1, int(n * .9)); tr = np.arange(cut)
    va = np.arange(cut, n) if cut < n else tr
    model = ICTransformer(X.shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=float(CFG['model']['learning_rate']), weight_decay=float(CFG['model']['weight_decay']))
    dl = DataLoader(TensorDataset(torch.from_numpy(Xn[tr]), torch.from_numpy(y[tr])), batch_size=int(CFG['model']['batch_size']), shuffle=False)
    best = 1e99; state = None; bad = 0
    for _ in range(int(CFG['model']['epochs'])):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(set_to_none=True); pred = model(xb)
            loss = sum(pinball(pred[:, j], yb[:, j]) for j in range(3)) / 3.0
            if not torch.isfinite(loss): raise RuntimeError('V8.2 non-finite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(CFG['model']['gradient_clip'])); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(Xn[va]))
            val = float(sum(pinball(pred[:, j], torch.from_numpy(y[va, j])) for j in range(3)) / 3.0)
        if not np.isfinite(val): raise RuntimeError('V8.2 non-finite validation loss')
        if val < best - 1e-6: best = val; state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= int(CFG['model']['patience']): break
    if state is not None: model.load_state_dict(state)
    return model, (xm.astype(np.float32), xs.astype(np.float32))
'''
    s = replace_block(s, r'^def fit_model\(X,meta\):', r'^def predict\(', fit)

    pred = '''def predict(model, norm, X):
    xm, xs = norm; Xn = np.clip((X - xm) / xs, -8, 8).astype(np.float32); model.eval()
    with torch.no_grad(): raw = model(torch.from_numpy(Xn)).numpy()
    if not np.isfinite(raw).all(): raise RuntimeError('V8.2 non-finite prediction')
    dq = np.diff(raw, axis=2)
    cross = max(float(np.max(raw[:,0] - raw[:,1])), float(np.max(raw[:,1] - raw[:,2])))
    if np.any(dq < -1e-6): raise RuntimeError(f'V8.2 quantile monotonicity invariant failed: min_dq={float(dq.min())}')
    if cross > 1e-6: raise RuntimeError(f'V8.2 cross-range coherence invariant failed: max_cross={cross}')
    return raw
'''
    s = replace_block(s, r'^def predict\(model,norm,X\):', r'^def sample_q\(', pred)

    pm = '''def pred_metrics(model, norm, X, meta, train_meta):
    d = predict(model, norm, X)
    actual = np.column_stack([meta['path_low_return'], meta['target_return'], meta['path_high_return']]).astype(float)
    q50 = d[:, :, 3]; mae = np.mean(np.abs(actual - q50), axis=0)
    med = train_meta[['path_low_return', 'target_return', 'path_high_return']].median().to_numpy(float)
    naive = np.tile(med, (len(meta), 1)); nm = np.mean(np.abs(actual-naive), axis=0)
    return {'model_low_mae':float(mae[0]),'model_close_mae':float(mae[1]),'model_high_mae':float(mae[2]),'naive_low_mae':float(nm[0]),'naive_close_mae':float(nm[1]),'naive_high_mae':float(nm[2]),'close_mae_lift_vs_naive':float(1-mae[1]/nm[1]) if nm[1]>0 else 0}
'''
    s = replace_block(s, r'^def pred_metrics\(model,norm,X,meta,train_meta\):', r'^def splits\(', pm)
    s = s.replace("score=rr*pop*max(0,1-touch)", "score=(rr / max(float(CFG.get('preferred_ic_reward_risk', 2.0)), 1e-9))*pop*max(0,1-touch)")
    s = s.replace("'candidate_count':len(valid),", "'candidate_count':len(valid),'raw_candidate_count':len(latest_rows),")
    s = s.replace("'post_cost_ic_payoff':True", "'post_cost_ic_payoff':True,'joint_quantile_coherence':True,'baseline_audit':True")
    ENGINE.write_text(s)

def patch_inspector():
    s = INSPECTOR.read_text().replace('V8.1','V8.2'); INSPECTOR.write_text(s)

if __name__ == '__main__':
    patch_engine(); patch_inspector(); print('V8.2 joint-range patch applied')