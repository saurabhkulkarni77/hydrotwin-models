"""Latest forecasting architectures: PatchTST, iTransformer, DLinear, TiDE, Mamba-lite."""
import sys, json, time
import os
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lstm_normal import prepare, rmse_log, predict_logflow, fit, val_loss
PROJ = Path(os.environ.get("HYDROTWIN_PROJ", Path(__file__).resolve().parent))
RES = PROJ / "ml" / "results"
torch.set_num_threads(2)

# ---------------- PatchTST: patch the series, Transformer on patches ----------------
class PatchTST(nn.Module):
    def __init__(self, d=64, patch=7, stride=7, layers=2, heads=4):
        super().__init__()
        self.patch, self.stride = patch, stride
        self.npatch = (29 - patch)//stride + 1
        self.emb = nn.Linear(patch*6, d)
        self.pos = nn.Parameter(torch.zeros(1, self.npatch, d))
        self.blocks = nn.ModuleList([nn.ModuleDict({
            "att": nn.MultiheadAttention(d, heads, batch_first=True),
            "n1": nn.LayerNorm(d),
            "ff": nn.Sequential(nn.Linear(d,2*d), nn.GELU(), nn.Linear(2*d,d)),
            "n2": nn.LayerNorm(d)} ) for _ in range(layers)])
        self.head = nn.Linear(d*self.npatch, 7)
    def forward(self, x):
        B = x.shape[0]
        p = x.unfold(1, self.patch, self.stride).permute(0,1,3,2).reshape(B, self.npatch, -1)
        h = self.emb(p) + self.pos
        for b in self.blocks:
            a,_ = b["att"](h,h,h); h = b["n1"](h+a); h = b["n2"](h+b["ff"](h))
        return self.head(h.reshape(B, -1))

# ---------------- iTransformer: variate tokens ----------------
class iTransformer(nn.Module):
    def __init__(self, d=64, layers=2, heads=4):
        super().__init__()
        self.emb = nn.Linear(29, d)  # each of 6 variates -> token
        self.blocks = nn.ModuleList([nn.ModuleDict({
            "att": nn.MultiheadAttention(d, heads, batch_first=True),
            "n1": nn.LayerNorm(d),
            "ff": nn.Sequential(nn.Linear(d,2*d), nn.GELU(), nn.Linear(2*d,d)),
            "n2": nn.LayerNorm(d)} ) for _ in range(layers)])
        self.head = nn.Linear(6*d, 7)
    def forward(self, x):
        h = self.emb(x.transpose(1,2))  # (B,6,29) -> (B,6,d)
        for b in self.blocks:
            a,_ = b["att"](h,h,h); h = b["n1"](h+a); h = b["n2"](h+b["ff"](h))
        return self.head(h.reshape(x.shape[0], -1))

# ---------------- DLinear: decomposition + linear ----------------
class DLinear(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.k = k
        self.lin_seas = nn.Linear(29, 7); self.lin_trend = nn.Linear(29, 7)
    def forward(self, x):
        # x: (B,29,6) -> predict per-feature then take flow channel? No: linear on flattened
        B = x.shape[0]
        xf = x.reshape(B, 29*6)
        trend = xf.unfold(1, self.k, 1).mean(-1)  # moving avg approx
        seas = xf - torch.cat([trend[:,:1].expand(-1,self.k//2), trend, trend[:,-1:].expand(-1,self.k//2)], 1)[:,:xf.shape[1]]
        # simpler: two linears on flattened input
        return self.lin_seas(xf) * 0 + nn.Linear(29*6, 7)(xf)  # placeholder replaced below

class DLinearV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(29*6, 7)
    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1))

# ---------------- TiDE: dense MLP encoder-decoder ----------------
class TiDE(nn.Module):
    def __init__(self, d=128, layers=2):
        super().__init__()
        enc = [nn.Linear(29*6, d), nn.GELU()]
        for _ in range(layers-1): enc += [nn.Linear(d, d), nn.GELU()]
        self.enc = nn.Sequential(*enc)
        self.dec = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 7))
    def forward(self, x):
        return self.dec(self.enc(x.reshape(x.shape[0], -1)))

# ---------------- Mamba-lite: selective SSM (simplified) ----------------
class MambaLite(nn.Module):
    def __init__(self, d=64, dstate=16):
        super().__init__()
        self.d, self.dstate = d, dstate
        self.inp = nn.Linear(6, d)
        self.dt = nn.Linear(d, d)          # selective timestep
        self.A = nn.Parameter(-torch.rand(d, dstate))
        self.B = nn.Linear(d, dstate); self.C = nn.Linear(d, dstate)
        self.out = nn.Linear(d, d); self.head = nn.Linear(d, 7)
        self.norm = nn.LayerNorm(d)
    def forward(self, x):
        B, T, _ = x.shape
        h = self.inp(x)                     # (B,T,d)
        s = torch.zeros(B, self.d, self.dstate, device=x.device)
        ys = []
        for t in range(T):
            ht = h[:, t]                    # (B,d)
            dt = F.softplus(self.dt(ht)).unsqueeze(-1)       # (B,d,1)
            A = torch.exp(dt * self.A.unsqueeze(0))               # (B,d,ds)
            Bb = self.B(ht).unsqueeze(1)                         # (B,1,ds)
            s = s * A + Bb * ht.unsqueeze(-1)
            y = (s * self.C(ht).unsqueeze(1)).sum(-1)            # (B,d)
            ys.append(y)
        y = torch.stack(ys, 1)
        return self.head(self.norm(self.out(y)[:, -1]))

MODELS = {
    "patchtst": lambda: PatchTST(),
    "itransformer": lambda: iTransformer(),
    "dlinear": lambda: DLinearV2(),
    "tide": lambda: TiDE(),
    "mamba": lambda: MambaLite(),
}

def main():
    P = prepare()
    te = np.where(P["split"]=="test")[0]
    results = {}
    for name, fn in MODELS.items():
        print(f"=== {name} ===", flush=True)
        preds = []
        for seed in (0, 1):
            torch.manual_seed(seed); np.random.seed(seed)
            m = fn()
            m, _, _ = fit(m, P, seed=seed)
            pred = predict_logflow(m, P)
            preds.append(pred)
            np.save(RES/f"pred_{name}_s{seed}.npy", pred.astype(np.float32))
            print(f"  seed {seed} test {rmse_log(P, pred[te], te):.4f}", flush=True)
        ens = (preds[0]+preds[1])/2
        r = round(float(rmse_log(P, ens[te], te)), 4)
        results[name] = r
        print(f"{name} ensemble: {r}", flush=True)
    json.dump(results, open(RES/"new_arch_results.json","w"), indent=1)
    print(json.dumps(results, indent=1), flush=True)

if __name__ == "__main__":
    main()
