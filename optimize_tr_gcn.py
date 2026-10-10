"""Grid-search Transformer (d x layers) and GCN (hidden x gc_layers)."""
import sys, json, time, itertools, math
from pathlib import Path
import numpy as np, torch, torch.nn as nn

sys.path.insert(0, "/home/hatch/workspace/hydrotwin_colab")
from train_lstm_normal import prepare, rmse_log, fit, val_loss, predict_logflow
RES = Path("/home/hatch/workspace/user/files/HydroTwin_10yr_Map/ml/results")
torch.set_num_threads(2)

class TransformerFlex(nn.Module):
    def __init__(self, d=16, layers=1, heads=2, T=29):
        super().__init__()
        self.emb = nn.Linear(9, d); self.pos = nn.Parameter(torch.zeros(1, T, d))
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict({
                "att": nn.MultiheadAttention(d, heads, batch_first=True),
                "n1": nn.LayerNorm(d),
                "ff": nn.Sequential(nn.Linear(d, 2*d), nn.GELU(), nn.Linear(2*d, d)),
                "n2": nn.LayerNorm(d)}))
        self.head = nn.Linear(d, 7)
    def forward(self, x):
        h = self.emb(x) + self.pos
        for b in self.blocks:
            a, _ = b["att"](h, h, h); h = b["n1"](h + a); h = b["n2"](h + b["ff"](h))
        return self.head(h[:, -1])

class GCNFlex(nn.Module):
    def __init__(self, hidden=16, gc_layers=2):
        super().__init__()
        self.enc = nn.LSTM(6, hidden, batch_first=True)
        self.gc = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(gc_layers)])
        self.out = nn.Linear(hidden, 7)
    def forward(self, x, A_hat):
        B, N = x.shape[:2]
        h, _ = self.enc(x.reshape(B*N, 29, 6)); h = h[:, -1].reshape(B, N, -1)
        for layer in self.gc:
            h = torch.relu(torch.einsum("ij,bjh->bih", A_hat, h)); h = layer(h)
        return self.out(h)

def build_batches(P):
    sites = list(P["sites"]); N = len(sites)
    dates = sorted(set(P["issue"].tolist()))
    lut = {(P["issue"][i], int(P["sid"][i])): i for i in range(len(P["X"]))}
    Xs, Ts, Ms, splits = [], [], [], []
    for d in dates:
        xb = np.zeros((N,29,6), np.float32); tb = np.zeros((N,7), np.float32)
        mb = np.zeros(N, bool); sp = None
        for si in range(N):
            j = lut.get((d, si))
            if j is not None: xb[si]=P["X"][j]; tb[si]=P["T"][j]; mb[si]=True; sp=P["split"][j]
        if mb.sum(): Xs.append(xb); Ts.append(tb); Ms.append(mb); splits.append(sp)
    return np.array(Xs), np.array(Ts), np.array(Ms), np.array(splits)

def main():
    P = prepare()
    va = np.where(P["split"] == "val")[0]; te = np.where(P["split"] == "test")[0]
    tr = np.where(P["split"] == "train")[0]
    Xt = torch.tensor(P["X"]); Tt = torch.tensor(P["T"])

    # ---- Transformer grid ----
    print("=== Transformer grid ===", flush=True)
    t_results = []
    for d, layers in itertools.product([16, 32, 64], [1, 2]):
        v_rmses, t_rmses = [], []
        for seed in (0, 1):
            torch.manual_seed(seed); np.random.seed(seed)
            m = TransformerFlex(d=d, layers=layers, heads=2)
            m, _, _ = fit(m, P, seed=seed)
            pred = predict_logflow(m, P)
            v_rmses.append(rmse_log(P, pred[va], va)); t_rmses.append(rmse_log(P, pred[te], te))
            np.save(RES / f"opt_tr_d{d}_l{layers}_s{seed}.npy", pred.astype(np.float32))
            print(f"d={d} L={layers} seed={seed}: val {v_rmses[-1]:.4f} test {t_rmses[-1]:.4f}", flush=True)
        t_results.append({"d": d, "layers": layers, "val": float(np.mean(v_rmses)),
                          "test": float(np.mean(t_rmses))})
    t_results.sort(key=lambda r: r["val"])
    json.dump(t_results, open(RES / "opt_transformer_grid.json", "w"), indent=1)
    bt = t_results[0]; print("BEST TR:", json.dumps(bt), flush=True)

    # ---- GCN grid ----
    print("=== GCN grid ===", flush=True)
    Xs, Ts, Ms, splits = build_batches(P)
    sites = list(P["sites"]); n = len(sites)
    idx = {s: i for i, s in enumerate(sites)}
    A = np.eye(n)
    for u, v in [("05331000","07374000"),("06934500","07374000"),("03294500","07374000"),("06214500","06934500")]:
        A[idx[u],idx[v]] = A[idx[v],idx[u]] = 1.0
    D = np.diag(1.0/np.sqrt(A.sum(1)))
    A_hat = torch.tensor(D @ A @ D, dtype=torch.float32)
    splits = np.array(splits)
    gtr = np.where(splits=="train")[0]; gva = np.where(splits=="val")[0]; gte = np.where(splits=="test")[0]
    Xt2 = torch.tensor(Xs); Tt2 = torch.tensor(Ts); Mt2 = torch.tensor(Ms)
    lut = {(P["issue"][i], int(P["sid"][i])): i for i in range(len(P["X"]))}
    ds = sorted(set(P["issue"].tolist()))
    # rebuild date list aligned with batches
    dlist = []
    for d in ds:
        if any(lut.get((d, si)) is not None for si in range(n)): dlist.append(d)
    def to_P(pred_b):
        pred = np.full((len(P["X"]),7), np.nan, np.float32)
        for bi, d in enumerate(dlist):
            for si in range(n):
                j = lut.get((d, si))
                if j is not None and Ms[bi, si]:
                    pred[j] = P["last"][j] + pred_b[bi, si] * P["scale"][j]
        return pred
    g_results = []
    for hidden, gl in itertools.product([16, 32], [2, 3]):
        v_rmses, t_rmses = [], []
        for seed in (0, 1):
            torch.manual_seed(seed); np.random.seed(seed)
            m = GCNFlex(hidden, gl)
            opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-5)
            rng = np.random.default_rng(seed)
            best, bs, bad, t0 = np.inf, None, 0, time.time()
            for ep in range(30):
                m.train(); perm = rng.permutation(gtr)
                for b in range(0, len(perm), 256):
                    i2 = perm[b:b+256]; opt.zero_grad()
                    out = m(Xt2[i2], A_hat); mk = Mt2[i2].unsqueeze(-1)
                    (((out-Tt2[i2])**2)*mk).sum().div(mk.sum().clamp(min=1)).backward()
                    nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
                m.eval()
                with torch.no_grad():
                    s2=0.0; c2=0
                    for b in range(0, len(gva), 256):
                        i2 = gva[b:b+256]; out = m(Xt2[i2], A_hat); mk = Mt2[i2].unsqueeze(-1)
                        s2 += (((out-Tt2[i2])**2)*mk).sum().item(); c2 += mk.sum().item()
                vl = s2/max(c2,1)
                if vl < best-1e-5: best, bs, bad = vl, {k:v.clone() for k,v in m.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= 5: break
            m.load_state_dict(bs); m.eval()
            with torch.no_grad():
                outs = torch.cat([m(Xt2[gva[i:i+256]], A_hat) for i in range(0, len(gva), 256)]).numpy()
                outt = torch.cat([m(Xt2[gte[i:i+256]], A_hat) for i in range(0, len(gte), 256)]).numpy()
            pv = to_P(np.concatenate([np.zeros((0,n,7)), outs], 0) if False else np.full((len(Xs),n,7), np.nan))
            # simpler: full predict then slice
            with torch.no_grad():
                full = torch.cat([m(Xt2[i:i+256], A_hat) for i in range(0, len(Xs), 256)]).numpy()
            pred = to_P(full)
            te_idx = np.where(P["split"]=="test")[0]; va_idx = np.where(P["split"]=="val")[0]
            mv = ~np.isnan(pred[va_idx]).any(1); mt_ = ~np.isnan(pred[te_idx]).any(1)
            v_rmses.append(rmse_log(P, pred[va_idx][mv], va_idx[mv]))
            t_rmses.append(rmse_log(P, pred[te_idx][mt_], te_idx[mt_]))
            np.save(RES / f"opt_gcn_h{hidden}_g{gl}_s{seed}.npy", pred.astype(np.float32))
            print(f"h={hidden} g={gl} seed={seed}: val {v_rmses[-1]:.4f} test {t_rmses[-1]:.4f}", flush=True)
        g_results.append({"hidden": hidden, "gc_layers": gl, "val": float(np.mean(v_rmses)),
                          "test": float(np.mean(t_rmses))})
    g_results.sort(key=lambda r: r["val"])
    json.dump(g_results, open(RES / "opt_gcn_grid.json", "w"), indent=1)
    bg = g_results[0]; print("BEST GCN:", json.dumps(bg), flush=True)
    import shutil
    for seed in (0, 1):
        shutil.copy(RES / f"opt_gcn_h{bg['hidden']}_g{bg['gc_layers']}_s{seed}.npy",
                    RES / f"pred_gcn_s{seed}.npy")
    # transformer best -> need canonical? transformer preds aren't in WebXR; skip
    print("done", flush=True)

if __name__ == "__main__":
    main()
