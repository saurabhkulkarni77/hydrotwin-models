#!/usr/bin/env python3
"""Ablation study: train key models on feature subsets.

Subsets (indices into the 9-feature prepare9() windows):
  6: [0,1,2,3,4,5]       USGS+Open-Meteo (baseline)
  7: [0,1,2,3,4,5,8]     + snow (swe)
  8: [0,1,2,3,4,5,6,7]   + soil (sms, stmp)
  9: [0..8]              all (already done, for reference)

Models: LSTM, PILSTMv1, PILSTMv2, Transformer, GP_Matern
Protocol: best_lr from per_station_results_opt.json, seed 0, patience 7,
          60 epochs, wd 1e-5. Same as the 9-feature run.

For GP: kernel features are subset of the 11 (8 base + sms, stmp, swe):
  6 -> 8 base only; 7 -> 8 + swe; 8 -> 8 + sms,stmp; 9 -> all 11.

Saves per-station test RMSE to ablation_results.json (merged with existing).
Usage: python3 train_ablation.py --model LSTM --subset 7
       python3 train_ablation.py --model GP_Matern --subset 8
"""
import sys, json, time, os, argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn

sys.path.insert(0, "/home/hatch/workspace/hydrotwin_colab")
for v in ("HYDROTWIN_TRAIN_END", "HYDROTWIN_VAL_END", "HYDROTWIN_TEST_START"):
    os.environ.pop(v, None)

import train_lstm_normal as tln

PROJ = Path("/home/hatch/workspace/user/files/HydroTwin_10yr_Map")
RES = PROJ / "ml" / "results"
ABL = Path("/home/hatch/workspace/hydrotwin_colab/ablation_results.json")

SUBSETS = {
    6: [0, 1, 2, 3, 4, 5],
    7: [0, 1, 2, 3, 4, 5, 8],
    8: [0, 1, 2, 3, 4, 5, 6, 7],
    9: list(range(9)),
}
# GP kernel feature indices (into the 11: 8 base + [sms, stmp, swe])
GP_SUBSETS = {
    6: list(range(8)),
    7: list(range(8)) + [10],
    8: list(range(8)) + [8, 9],
    9: list(range(11)),
}

WD, MAX_EPOCHS, PATIENCE, BATCH = 1e-5, 60, 7, 256


class LSTMNetFlex(nn.Module):
    def __init__(self, in_dim, hidden=16, dropout=0.0):
        super().__init__()
        self.rnn = nn.LSTM(in_dim, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 7)
    def forward(self, x):
        h, _ = self.rnn(x)
        return self.head(self.drop(h[:, -1]))


class TransformerFlexIn(nn.Module):
    def __init__(self, in_dim, d=16, layers=1, heads=2, T=29):
        super().__init__()
        self.emb = nn.Linear(in_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, T, d))
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


class PILSTMv2Flex(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 7)
    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h[:, -1])


def subset_P(P, idx):
    """Return P with X sliced to feature indices idx and stats subsetted."""
    Q = dict(P)
    Q["X"] = P["X"][:, :, idx].copy()
    stats = {}
    for site, st in P["stats"].items():
        s = dict(st)
        s["mu"] = [st["mu"][i] for i in idx]
        s["sd"] = [st["sd"][i] for i in idx]
        stats[site] = s
    Q["stats"] = stats
    return Q


def subset_station(P, si):
    m = P["sid"] == si
    return {k: (v[m] if isinstance(v, np.ndarray) and len(v) == len(P["X"]) else v)
            for k, v in P.items()}


def eval_test_rmse(model, Ps):
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(Ps["X"]); Tt = torch.tensor(Ps["T"])
        out = model(Xt)
    pred = tln.predict_logflow(model, Ps)  # uses model internally; simpler: compute directly
    te = np.where(Ps["split"] == "test")[0]
    return float(tln.rmse_log(Ps, pred, te))


def train_neural(name, in_dim, subset_idx):
    from train_pilstm import physics_loss, LAMBDA_PHYS
    from train_pilstm_v2 import physics_v2, load_soil_snow, LAM
    import pandas as pd

    P9 = tln.prepare9()
    P = subset_P(P9, subset_idx)
    assert P["X"].shape[2] == in_dim

    # physics lookups (same as train_export_lstm9.py)
    DAILY = PROJ / "data" / "daily"
    prcp, tm = [], []
    for si, s in enumerate(P9["sites"]):
        d = pd.read_csv(DAILY / f"{s}_daily.csv", parse_dates=["date"])
        pm = dict(zip(d.date.dt.strftime("%Y-%m-%d"), d.precip_mm.fillna(0).to_numpy()))
        tmm = dict(zip(d.date.dt.strftime("%Y-%m-%d"), d.tmean_c.fillna(10).to_numpy()))
        prcp.append(np.array([pm[dt] for dt in P9["issue"][P9["sid"] == si]]))
        tm.append(np.array([tmm[dt] for dt in P9["issue"][P9["sid"] == si]]))
    prcp = np.concatenate(prcp); tm = np.concatenate(tm)
    sms, stmp, swe = load_soil_snow(P9)

    res = json.loads((RES / "per_station_results_opt.json").read_text())

    def make_model():
        if name == "LSTM":
            return LSTMNetFlex(in_dim)
        if name == "PILSTMv1":
            return LSTMNetFlex(in_dim)
        if name == "PILSTMv2":
            return PILSTMv2Flex(in_dim)
        if name == "Transformer":
            return TransformerFlexIn(in_dim)
        raise ValueError(name)

    out_rmse = {}
    for si, site in enumerate(P9["sites"]):
        key = "LSTM" if name == "PILSTMv1" else name
        try:
            best_lr = res[key][site]["best_lr"]
        except (KeyError, TypeError):
            print(f"{name} {site}: no best_lr, skip", flush=True)
            continue
        Ps = subset_station(P, si)
        if np.sum(Ps["split"] == "train") < 200:
            print(f"{name} {site}: too little train data, skip", flush=True)
            continue
        m_mask = (P9["sid"] == si)
        t0 = time.time()
        torch.manual_seed(0); np.random.seed(0)
        model = make_model()

        # training loop (mirrors train_export_lstm9.py fitters)
        tr = np.where(Ps["split"] == "train")[0]; va = np.where(Ps["split"] == "val")[0]
        Xt, Tt = torch.tensor(Ps["X"]), torch.tensor(Ps["T"])
        last_t = torch.tensor(Ps["last"], dtype=torch.float32)
        scale_t = torch.tensor(Ps["scale"], dtype=torch.float32)
        opt = torch.optim.Adam(model.parameters(), lr=best_lr, weight_decay=WD)
        rng = np.random.default_rng(0)
        best, best_state, bad, eps = np.inf, None, 0, 0
        use_phys = name in ("PILSTMv1", "PILSTMv2")
        if use_phys:
            prcp_t = torch.tensor(prcp[m_mask], dtype=torch.float32)
        if name == "PILSTMv2":
            tm_t = torch.tensor(tm[m_mask], dtype=torch.float32)
            sms_t = torch.tensor(sms[m_mask], dtype=torch.float32)
            stmp_t = torch.tensor(stmp[m_mask], dtype=torch.float32)
            swe_t = torch.tensor(swe[m_mask], dtype=torch.float32)
        for ep in range(MAX_EPOCHS):
            model.train(); perm = rng.permutation(tr)
            for b in range(0, len(perm), BATCH):
                idx = perm[b:b + BATCH]
                opt.zero_grad()
                out = model(Xt[idx])
                data = ((out - Tt[idx]) ** 2).mean()
                if name == "PILSTMv1":
                    pred_lf = last_t[idx, None] + out * scale_t[idx]
                    phys = physics_loss(pred_lf, prcp_t[idx], last_t[idx])
                    loss = data + LAMBDA_PHYS * phys
                elif name == "PILSTMv2":
                    pred_lf = last_t[idx, None] + out * scale_t[idx]
                    phys = physics_v2(pred_lf, last_t[idx], prcp_t[idx], tm_t[idx],
                                      sms_t[idx], stmp_t[idx], swe_t[idx])
                    loss = data + LAM * phys
                else:
                    loss = data
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            vl = tln.val_loss(model, Xt, Tt, va)
            if vl < best - 1e-5:
                best, best_state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
            eps = ep + 1
        model.load_state_dict(best_state)
        # test RMSE
        model.eval()
        with torch.no_grad():
            pred = tln.predict_logflow(model, Ps)
        te = np.where(Ps["split"] == "test")[0]
        rmse = float(tln.rmse_log(Ps, pred[te], te))
        out_rmse[site] = round(rmse, 4)
        print(f"{name} sub{len(subset_idx)} {site}: test_rmse={rmse:.4f} "
              f"ep={eps} [{time.time()-t0:.0f}s]", flush=True)
    return out_rmse


def train_gp(subset_n):
    """GP Matérn on kernel-feature subset, with per-station hyperparameter
    grid search mirroring refine_gp.py. Subset selects from the 11 kernel
    features (8 base + sms, stmp, swe)."""
    from scipy.linalg import cho_factor, cho_solve
    import pandas as pd

    def matern(A, B, ls):
        r = np.sqrt(np.maximum(((A[:, None] - B[None]) ** 2).sum(-1), 0)) / ls
        return (1 + np.sqrt(5) * r + 5 / 3 * r ** 2) * np.exp(-np.sqrt(5) * r)

    def gp_pred(Ktr, Kte, Y, noise):
        c = cho_factor(Ktr + noise * np.eye(len(Ktr)))
        return Kte @ cho_solve(c, Y)

    def kfeatures(X9):
        """11 kernel features from 9-channel windows (same as refine_gp.features)."""
        lf = np.log(X9[:, :, 4]); rain = X9[:, :, 3]; tm = X9[:, :, 2]
        base = np.stack([lf[:, -1], lf[:, -1] - lf[:, -2], lf[:, -1] - lf[:, -8],
                         rain[:, -1], rain[:, -3:].sum(1), rain[:, -7:].sum(1),
                         rain.sum(1), tm[:, -7:].mean(1)], 1)
        return np.concatenate([base, X9[:, -1, 6:9]], 1)

    P9 = tln.prepare9()
    kidx = GP_SUBSETS[subset_n]
    # starting hypers from 9-feat refined
    G = json.loads((RES / "gp_refined9.json").read_text())

    out_rmse = {}
    for si, site in enumerate(P9["sites"]):
        m = P9["sid"] == si
        st = P9["stats"][site]
        mu = np.array(st["mu"]); sd = np.array(st["sd"])
        Xr = P9["X"][m] * sd[None, None, :] + mu[None, None, :]  # unnormalize
        F = kfeatures(Xr)[:, kidx]
        tr = np.where(P9["split"][m] == "train")[0]
        va = np.where(P9["split"][m] == "val")[0]
        te = np.where(P9["split"][m] == "test")[0]
        fmu, fsd = F[tr].mean(0), F[tr].std(0) + 1e-9
        Z = (F - fmu) / fsd
        # targets: standardized log-flow deltas (match refine_gp Ts convention)
        T = P9["T"][m]; scale = P9["scale"][m]; last = P9["last"][m]
        # grid search on val
        cur = G.get(site, {}).get("matern", {})
        cur_ls = float(cur.get("setting", 1.0)); cur_nz = float(cur.get("noise", 0.1))
        ls_cands = sorted(set([cur_ls * f for f in (0.5, 0.7, 1.0, 1.4, 2.0)]))
        nz_cands = sorted(set([cur_nz * f for f in (0.3, 1.0, 3.0)]))
        t0 = time.time()
        best = (np.inf, cur_ls, cur_nz)
        K_full = None
        for ls in ls_cands:
            K = matern(Z, Z, ls)
            Ktr = K[np.ix_(tr, tr)]; Kva = K[np.ix_(va, tr)]
            for nz in nz_cands:
                try:
                    mu_p = gp_pred(Ktr, Kva, T[tr], nz)
                except Exception:
                    continue
                rmse = float(np.sqrt(((mu_p - T[va]) ** 2).mean()))
                if rmse < best[0]:
                    best = (rmse, ls, nz)
        bls, bnz = best[1], best[2]
        # test RMSE in log-flow units
        Ktr = matern(Z[tr], Z[tr], bls); Kte = matern(Z[te], Z[tr], bls)
        mup = gp_pred(Ktr, Kte, T[tr], bnz)
        pred_lf = last[te, None] + mup * scale[te]
        true_lf = last[te, None] + T[te] * scale[te]
        rmse = float(np.sqrt(((pred_lf - true_lf) ** 2).mean()))
        out_rmse[site] = round(rmse, 4)
        print(f"GP_Matern sub{subset_n} {site}: test={rmse:.4f} "
              f"ls={bls:.2f} nz={bnz:.4f} [{time.time()-t0:.0f}s]", flush=True)
    return out_rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["LSTM", "PILSTMv1", "PILSTMv2", "Transformer", "GP_Matern"])
    ap.add_argument("--subset", type=int, required=True, choices=[6, 7, 8, 9])
    a = ap.parse_args()
    torch.set_num_threads(2)

    if a.model == "GP_Matern":
        rmse = train_gp(a.subset)
    else:
        rmse = train_neural(a.model, len(SUBSETS[a.subset]), SUBSETS[a.subset])

    # merge into ablation_results.json
    if ABL.exists():
        abl = json.loads(ABL.read_text())
    else:
        abl = {}
    key = f"{a.subset}feat"
    abl.setdefault(key, {})[a.model] = rmse
    # mean
    vals = list(rmse.values())
    abl[key][a.model + "_mean"] = round(float(np.mean(vals)), 4) if vals else None
    ABL.write_text(json.dumps(abl, indent=1))
    print(f"ABLATION DONE: {a.model} subset {a.subset}, mean={np.mean(vals):.4f}")


if __name__ == "__main__":
    main()
