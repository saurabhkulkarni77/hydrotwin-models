"""Physics-Informed LSTM (PILSTM) on the 14 HydroTwin rivers.

Adapts the folder's PIML idea to HydroTwin's USGS-only features.
Architecture: same LSTMNet as the normal LSTM (hidden=16).
Loss = data MSE + λ * physics_loss, where the physics loss encodes two
hydrological priors:

1. Recession prior: on dry days (no recent rain) streamflow recedes — it should
   not jump up. Penalizes predicted day-to-day rises weighted by dryness.
2. Water-balance prior: over the 7-day horizon, total outflow volume cannot
   exceed current channel storage (yesterday's flow × 7, loose) plus total
   precip input. Penalizes the excess.

Both are soft constraints (differentiable, ReLU-based).
Usage: python3 train_pilstm.py [seed...]
"""
import sys, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn

PROJ = Path("/home/hatch/workspace/user/files/HydroTwin_10yr_Map")
RES = PROJ / "ml" / "results"
torch.set_num_threads(1)

LAMBDA_PHYS = 0.1  # weight of the physics term


class LSTMNet(nn.Module):
    def __init__(self, hidden=16, dropout=0.0):
        super().__init__()
        self.rnn = nn.LSTM(9, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 7)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.head(self.drop(h[:, -1]))


def physics_loss(pred_lf, precip_mm, last_lf):
    """pred_lf: (B,7) predicted log flow; precip_mm: (B,) recent precip; last_lf: (B,) log flow at issue.
    Returns scalar physics penalty."""
    B = pred_lf.shape[0]
    # 1. Recession: on dry days flow should not rise day-to-day
    dry = torch.clamp(1.0 - precip_mm / 10.0, 0, 1).unsqueeze(1)  # 1 when dry, 0 when >=10mm
    rises = torch.relu(pred_lf[:, 1:] - pred_lf[:, :-1])          # (B,6) positive rises
    recession = (rises * dry).mean()
    # 2. Water balance: 7-day outflow <= 7 * Q0 (storage proxy) + precip
    q = torch.exp(pred_lf)                                       # cfs
    q0 = torch.exp(last_lf)
    excess = torch.relu(q.sum(1) - (7 * q0 + precip_mm * 50))    # 50 cfs per mm: loose
    balance = (excess / (7 * q0 + 1)).mean()
    return recession + balance


def main():
    seeds = [int(a) for a in sys.argv[1:]] or [0, 1]
    sys.path.insert(0, "/home/hatch/workspace/hydrotwin_colab")
    from train_lstm_normal import prepare, val_loss, predict_logflow, rmse_log
    print("preparing data...", flush=True)
    P = prepare()
    # recent precip per window (unstandardized, mm): feature 3 of last day; need train stats to invert
    # rebuild quickly: use raw daily files
    import pandas as pd
    DAILY = PROJ / "data" / "daily"
    prcp = []
    for si, s in enumerate(P["sites"]):
        d = pd.read_csv(DAILY / f"{s}_daily.csv", parse_dates=["date"])
        # map P windows back: use issue date -> precip on issue date
        pm = dict(zip(d.date.dt.strftime("%Y-%m-%d"), d.precip_mm.fillna(0).to_numpy()))
        prcp.append(np.array([pm[dt] for dt in P["issue"][P["sid"] == si]]))
    prcp = np.concatenate(prcp)
    # order prcp like P["X"] (sites in order, windows in date order) — prepare() already does that
    assert len(prcp) == len(P["X"])

    tr = np.where(P["split"] == "train")[0]
    va = np.where(P["split"] == "val")[0]
    te = np.where(P["split"] == "test")[0]
    Xt = torch.tensor(P["X"]); Tt = torch.tensor(P["T"])
    last_t = torch.tensor(P["last"], dtype=torch.float32)
    scale_t = torch.tensor(P["scale"], dtype=torch.float32)
    prcp_t = torch.tensor(prcp, dtype=torch.float32)

    for seed in seeds:
        print(f"=== pilstm seed {seed} ===", flush=True)
        torch.manual_seed(seed); np.random.seed(seed)
        model = LSTMNet()
        opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
        rng = np.random.default_rng(seed)
        best, best_state, bad, t0 = np.inf, None, 0, time.time()
        for ep in range(30):
            model.train(); perm = rng.permutation(tr)
            for b in range(0, len(perm), 256):
                idx = perm[b:b + 256]
                opt.zero_grad()
                out = model(Xt[idx])
                data = ((out - Tt[idx]) ** 2).mean()
                # physics on predicted log flow
                pred_lf = last_t[idx, None] + out * scale_t[idx]
                phys = physics_loss(pred_lf, prcp_t[idx], last_t[idx])
                (data + LAMBDA_PHYS * phys).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            vl = val_loss(model, Xt, Tt, va)
            print(f"  epoch {ep + 1} val {vl:.4f} [{time.time() - t0:.0f}s]", flush=True)
            if vl < best - 1e-5:
                best, best_state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 5:
                    break
        model.load_state_dict(best_state)
        pred = predict_logflow(model, P)
        np.save(RES / f"pred_pilstm_s{seed}.npy", pred.astype(np.float32))
        res = {"model": "pilstm", "seed": seed,
               "val_rmse_log": rmse_log(P, pred[va], va), "test_rmse_log": rmse_log(P, pred[te], te)}
        json.dump(res, open(RES / f"res_pilstm_s{seed}.json", "w"))
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
