"""Train a NORMAL (untuned) classical LSTM on the pooled 14-river USGS data.
Mirrors ml/common.py prepare()/fit() and ml/data10.py windows() exactly,
but reads the daily tables from the local project copy (no /home/claude).

Normal = LSTMNet defaults: hidden=16, dropout=0.0, lr=3e-3, wd=1e-5,
epochs=30, patience=5, batch=256. Two seeds (0, 1), ensemble = mean in log space.
Saves pred_lstm_s{SEED}.npy + res_lstm_s{SEED}.json next to the other results.
Usage: python3 train_lstm_normal.py [seed]   (default: 0 1)
"""
import sys, json, time, os
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn

PROJ = Path(os.environ.get("HYDROTWIN_PROJ", Path(__file__).resolve().parent))
DAILY = PROJ / "data" / "daily"
RES = PROJ / "ml" / "results"
DAYS = pd.date_range("2016-10-01", "2026-09-30", freq="D")
COLS = ["tmax_c", "tmin_c", "tmean_c", "precip_mm", "flow_cfs", "rising"]
WIN, LEADS = 29, 7
TRAIN_END = pd.Timestamp(os.environ.get("HYDROTWIN_TRAIN_END", "2023-09-30"))
VAL_END = pd.Timestamp(os.environ.get("HYDROTWIN_VAL_END", "2024-09-30"))
torch.set_num_threads(1)


class LSTMNet(nn.Module):
    def __init__(self, hidden=16, dropout=0.0):
        super().__init__()
        self.rnn = nn.LSTM(6, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 7)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.head(self.drop(h[:, -1]))


def windows(d):
    v = d[COLS].to_numpy(float); lf = np.log(d.flow_cfs.to_numpy(float))
    ok = ~np.isnan(v).any(1)
    okc = np.concatenate([[0], np.cumsum(ok)])
    X, Y, issue, ft = [], [], [], []
    for t in range(WIN - 1, len(d) - LEADS):
        if okc[t + 1] - okc[t - WIN + 1] != WIN or np.isnan(lf[t + 1:t + 1 + LEADS]).any():
            continue
        X.append(v[t - WIN + 1:t + 1]); Y.append(lf[t + 1:t + 1 + LEADS])
        issue.append(d.date.iloc[t]); ft.append(d.date.iloc[t + 1])
    ft = pd.Series(ft)
    split = np.where(ft <= TRAIN_END, "train", np.where(ft <= VAL_END, "val", "test"))
    return np.array(X), np.array(Y), np.array(issue), split


def prepare():
    Xs, Ts, last, sid, issue, split, scale = [], [], [], [], [], [], []
    sites = sorted(f.stem.replace("_daily", "") for f in DAILY.glob("*_daily.csv"))
    stats = {}
    for i, s in enumerate(sites):
        d = pd.read_csv(DAILY / f"{s}_daily.csv", parse_dates=["date"])
        X, Y, iss, sp = windows(d)
        if len(X) == 0:
            continue
        X = X.copy(); X[:, :, 4] = np.log(X[:, :, 4])
        lst = X[:, -1, 4]; T = Y - lst[:, None]
        tr = sp == "train"
        mu = X[tr].reshape(-1, 6).mean(0); sd = X[tr].reshape(-1, 6).std(0) + 1e-6
        tsd = T[tr].std(0) + 1e-6
        stats[s] = {"mu": mu.tolist(), "sd": sd.tolist(), "tsd": tsd.tolist()}
        Xs.append(((X - mu) / sd).astype(np.float32)); Ts.append((T / tsd).astype(np.float32))
        last.append(lst); sid.append(np.full(len(X), i))
        issue.append(iss.astype("datetime64[D]").astype(str)); split.append(sp)
        scale.append(np.repeat(tsd[None], len(X), 0))
    return dict(X=np.concatenate(Xs), T=np.concatenate(Ts), last=np.concatenate(last),
                sid=np.concatenate(sid), issue=np.concatenate(issue), split=np.concatenate(split),
                scale=np.concatenate(scale).astype(np.float32), sites=np.array(sites), stats=stats)


def val_loss(model, Xt, Tt, idx, batch=2048):
    model.eval(); s = 0.0
    with torch.no_grad():
        for b in range(0, len(idx), batch):
            i = idx[b:b + batch]
            s += ((model(Xt[i]) - Tt[i]) ** 2).sum().item()
    return s / len(idx)


def fit(model, P, lr=3e-3, wd=1e-5, epochs=30, patience=5, batch=256, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    tr = np.where(P["split"] == "train")[0]; va = np.where(P["split"] == "val")[0]
    Xt, Tt = torch.tensor(P["X"]), torch.tensor(P["T"])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best, best_state, bad, t0, eps = np.inf, None, 0, time.time(), 0
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train(); perm = rng.permutation(tr)
        for b in range(0, len(perm), batch):
            idx = perm[b:b + batch]
            opt.zero_grad(); loss = ((model(Xt[idx]) - Tt[idx]) ** 2).mean()
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        vl = val_loss(model, Xt, Tt, va)
        print(f"  epoch {ep + 1} val {vl:.4f} [{time.time() - t0:.0f}s]", flush=True)
        if vl < best - 1e-5:
            best, best_state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
        eps = ep + 1
    model.load_state_dict(best_state)
    return model, best, eps


def predict_logflow(model, P, batch=2048):
    """Predicted log flow for every window (unstandardized, like ml/common.py)."""
    model.eval(); out = []
    Xt = torch.tensor(P["X"])
    with torch.no_grad():
        for b in range(0, len(Xt), batch):
            out.append(model(Xt[b:b + batch]).numpy())
    Tp = np.concatenate(out)
    return P["last"][:, None] + Tp * P["scale"]


def rmse_log(P, pred, idx):
    true = P["last"][idx, None] + P["T"][idx] * P["scale"][idx]
    return float(np.sqrt(((pred - true) ** 2).mean()))


def main():
    seeds = [int(a) for a in sys.argv[1:]] or [0, 1]
    print("preparing data...", flush=True)
    P = prepare()
    n = len(P["X"])
    print(f"windows: {n}  train {(P['split'] == 'train').sum()}  val {(P['split'] == 'val').sum()}  test {(P['split'] == 'test').sum()}", flush=True)
    for seed in seeds:
        print(f"=== lstm (normal) seed {seed} ===", flush=True)
        t0 = time.time()
        model = LSTMNet()  # defaults: hidden=16, dropout=0.0
        nparams = sum(p.numel() for p in model.parameters())
        model, best, eps = fit(model, P, seed=seed)
        pred = predict_logflow(model, P)
        np.save(RES / f"pred_lstm_s{seed}.npy", pred.astype(np.float32))
        va = np.where(P["split"] == "val")[0]; te = np.where(P["split"] == "test")[0]
        res = {"model": "lstm", "seed": seed, "params": int(nparams), "epochs": int(eps),
               "minutes": round((time.time() - t0) / 60, 1),
               "val_rmse_log": rmse_log(P, pred[va], va), "test_rmse_log": rmse_log(P, pred[te], te)}
        json.dump(res, open(RES / f"res_lstm_s{seed}.json", "w"))
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
