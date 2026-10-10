"""8 more SOTA forecasters: N-HiTS, TimesNet-lite, TFT-lite, N-BEATS, SegRNN, TimeMixer-lite, ModernTCN-lite, FITS."""
import sys, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, "/home/hatch/workspace/hydrotwin_colab")
from train_lstm_normal import prepare, rmse_log, predict_logflow, fit
RES = Path("/home/hatch/workspace/user/files/HydroTwin_10yr_Map/ml/results")
torch.set_num_threads(2)

# ---------------- N-HiTS (hierarchical MLP) ----------------
class NHiTSBlock(nn.Module):
    def __init__(self, in_len, h, n_theta):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_len, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.theta_f = nn.Linear(h, n_theta)
        self.n_theta = n_theta
    def forward(self, x):
        h = self.mlp(x)
        theta = self.theta_f(h)
        # interpolate theta to horizon 7
        idx = torch.linspace(0, 1, 7, device=x.device)
        grid = torch.linspace(0, 1, self.n_theta, device=x.device)
        w = torch.softmax(-((idx[:, None]-grid[None])**2)*self.n_theta, dim=1)
        return theta @ w.T

class NHiTS(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = NHiTSBlock(29*6, 128, 14)
        self.b2 = NHiTSBlock(29*6, 128, 7)
    def forward(self, x):
        xf = x.reshape(x.shape[0], -1)
        return self.b1(xf) + self.b2(xf)

# ---------------- TimesNet-lite ----------------
class Inception(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.c1 = nn.Conv2d(d, d, 1); self.c3 = nn.Conv2d(d, d, 3, padding=1); self.c5 = nn.Conv2d(d, d, 5, padding=2)
    def forward(self, x):
        return F.gelu(self.c1(x) + self.c3(x) + self.c5(x))

class TimesNetLite(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.emb = nn.Linear(6, d)
        self.inc = Inception(d)
        self.head = nn.Linear(29*d, 7)
    def forward(self, x):
        h = self.emb(x)  # (B,29,d)
        # find dominant period via FFT on mean series
        with torch.no_grad():
            f = torch.fft.rfft(h.mean(-1), dim=1).abs()[:, 1:15]
            per = torch.clamp((29 // (f.argmax(1)+1)).clamp(min=2, max=14), min=2)
        B, T, Dd = h.shape
        outs = []
        for b in range(B):
            p = int(per[b].item()); n = (T + p - 1)//p
            pad = h[b].T  # (d,29)
            if pad.shape[1] < n*p: pad = F.pad(pad, (0, n*p - pad.shape[1]))
            img = pad.reshape(Dd, n, p).unsqueeze(0)
            o = self.inc(img).reshape(Dd, -1)[:, :T].T  # (29,d)
            outs.append(o)
        h2 = torch.stack(outs)
        return self.head((h + h2).reshape(B, -1))

# ---------------- TFT-lite ----------------
class GRN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, d); self.fc2 = nn.Linear(d, d)
        self.gate = nn.Linear(d, d); self.norm = nn.LayerNorm(d)
    def forward(self, x):
        h = F.elu(self.fc1(x)); h = self.fc2(h)
        return self.norm(x + torch.sigmoid(self.gate(x)) * h)

class TFTLite(nn.Module):
    def __init__(self, d=48, heads=4):
        super().__init__()
        self.emb = nn.Linear(9, d)
        self.lstm = nn.LSTM(d, d, batch_first=True)
        self.grn = GRN(d)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n = nn.LayerNorm(d)
        self.head = nn.Linear(d, 7)
    def forward(self, x):
        h = self.emb(x)
        h, _ = self.lstm(h)
        h = self.grn(h)
        a, _ = self.att(h, h, h)
        return self.head(self.n(h + a)[:, -1])

# ---------------- N-BEATS (generic) ----------------
class NBeatsBlock(nn.Module):
    def __init__(self, in_len, h=128, layers=3):
        super().__init__()
        ls = [nn.Linear(in_len, h), nn.ReLU()]
        for _ in range(layers-1): ls += [nn.Linear(h, h), nn.ReLU()]
        self.fc = nn.Sequential(*ls)
        self.back = nn.Linear(h, in_len); self.fore = nn.Linear(h, 7)
    def forward(self, x):
        h = self.fc(x)
        return x - self.back(h), self.fore(h)

class NBeats(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([NBeatsBlock(29*6) for _ in range(3)])
    def forward(self, x):
        xf = x.reshape(x.shape[0], -1); f = 0
        for b in self.blocks:
            xf, fo = b(xf); f = f + fo
        return f

# ---------------- SegRNN ----------------
class SegRNN(nn.Module):
    def __init__(self, seg=7, d=64):
        super().__init__()
        self.seg = seg; self.nseg = 29//seg
        self.proj = nn.Linear(seg*6, d)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.head = nn.Linear(d, 7)
    def forward(self, x):
        B = x.shape[0]
        s = x[:, :self.nseg*self.seg].reshape(B, self.nseg, -1)
        h = self.proj(s)
        _, hn = self.gru(h)
        return self.head(hn[-1])

# ---------------- TimeMixer-lite ----------------
class TimeMixerLite(nn.Module):
    def __init__(self, d=48):
        super().__init__()
        self.down1 = nn.AvgPool1d(2); self.down2 = nn.AvgPool1d(4)
        self.m0 = nn.Sequential(nn.Linear(29*6, d), nn.GELU())
        self.m1 = nn.Sequential(nn.Linear(14*6, d), nn.GELU())
        self.m2 = nn.Sequential(nn.Linear(7*6, d), nn.GELU())
        self.head = nn.Linear(3*d, 7)
    def forward(self, x):
        B = x.shape[0]
        x1 = x.transpose(1,2); x_14 = self.down1(x1).transpose(1,2); x_7 = self.down2(x1).transpose(1,2)
        h = torch.cat([self.m0(x.reshape(B,-1)), self.m1(x_14.reshape(B,-1)), self.m2(x_7.reshape(B,-1))], 1)
        return self.head(h)

# ---------------- ModernTCN-lite ----------------
class TCNBlock(nn.Module):
    def __init__(self, d, k=5):
        super().__init__()
        self.dw = nn.Conv1d(d, d, k, padding=k//2, groups=d)
        self.pw = nn.Conv1d(d, d, 1)
        self.n = nn.LayerNorm(d)
    def forward(self, x):
        h = x.transpose(1,2)
        h = self.pw(F.gelu(self.dw(h))).transpose(1,2)
        return self.n(x + h)

class ModernTCN(nn.Module):
    def __init__(self, d=48, layers=3):
        super().__init__()
        self.emb = nn.Linear(6, d)
        self.blocks = nn.Sequential(*[TCNBlock(d) for _ in range(layers)])
        self.head = nn.Linear(29*d, 7)
    def forward(self, x):
        h = self.blocks(self.emb(x))
        return self.head(h.reshape(x.shape[0], -1))

# ---------------- FITS ----------------
class FITS(nn.Module):
    def __init__(self, n_freq=12):
        super().__init__()
        self.nf = n_freq
        self.cmplx = nn.Linear(n_freq, n_freq)  # real-valued on concat(re,im)
        self.head = nn.Linear(6*n_freq*2, 7)
    def forward(self, x):
        B = x.shape[0]
        Xf = torch.fft.rfft(x, dim=1)[:, :self.nf]  # (B,nf,6)
        re = self.cmplx(Xf.real.transpose(1, 2)).transpose(1, 2)
        im = self.cmplx(Xf.imag.transpose(1, 2)).transpose(1, 2)
        return self.head(torch.cat([re, im], -1).reshape(B, -1))

MODELS = {
    "nhits": NHiTS, "timesnet": TimesNetLite, "tft": TFTLite, "nbeats": NBeats,
    "segrnn": SegRNN, "timemixer": TimeMixerLite, "moderntcn": ModernTCN, "fits": FITS,
}

def main():
    P = prepare(); te = np.where(P["split"]=="test")[0]
    results = {}
    for name, cls in MODELS.items():
        print(f"=== {name} ===", flush=True)
        preds = []
        for seed in (0, 1):
            torch.manual_seed(seed); np.random.seed(seed)
            m = cls()
            m, _, _ = fit(m, P, seed=seed)
            pred = predict_logflow(m, P); preds.append(pred)
            np.save(RES/f"pred_{name}_s{seed}.npy", pred.astype(np.float32))
            print(f"  seed {seed} test {rmse_log(P, pred[te], te):.4f}", flush=True)
        ens = (preds[0]+preds[1])/2
        r = round(float(rmse_log(P, ens[te], te)), 4)
        results[name] = r
        print(f"{name} ensemble: {r}", flush=True)
    json.dump(results, open(RES/"sota8_results.json","w"), indent=1)
    print(json.dumps(results, indent=1), flush=True)

if __name__ == "__main__":
    main()
