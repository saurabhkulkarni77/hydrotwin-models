"""Training loop + Ray Tune trainable.

train_one() mirrors the repo fit() semantics (Adam, MSE on standardized
ln-delta targets, grad-clip 1.0, best-checkpoint restore) with one
deliberate change: checkpoint selection and per-epoch reporting use the
objective metric itself — validation RMSE in log10(cfs) — so ASHA races
trials on the number the quality gate will judge.

PILSTMv1: same loop plus train_pilstm.py's physics loss verbatim
(recession + water-balance priors, lambda from the config, production
0.1), fed by P["precip"] (issue-date precip, exactly as production
rebuilds it) and the predicted ln-flow levels.
GP: exact GP regression (Matern-5/2, scipy Cholesky) on the 11 kernel
features P["gpF"], targets = standardized ln-deltas P["T"] — the
refine_gp.py / train_ablation.py protocol, no epochs.
"""
import time

import numpy as np
import torch
import torch.nn as nn

from architectures import build_model
from data_adapter import build_station_P, rmse_log10
from spaces import flatten


def physics_loss(pred_lf, precip_mm, last_lf):
    """train_pilstm.py physics_loss, verbatim: recession prior on dry
    days + water-balance prior over the 7-day horizon."""
    # 1. Recession: on dry days flow should not rise day-to-day
    dry = torch.clamp(1.0 - precip_mm / 10.0, 0, 1).unsqueeze(1)
    rises = torch.relu(pred_lf[:, 1:] - pred_lf[:, :-1])
    recession = (rises * dry).mean()
    # 2. Water balance: 7-day outflow <= 7 * Q0 (storage proxy) + precip
    q = torch.exp(pred_lf)
    q0 = torch.exp(last_lf)
    excess = torch.relu(q.sum(1) - (7 * q0 + precip_mm * 50))
    balance = (excess / (7 * q0 + 1)).mean()
    return recession + balance


def _matern(A, B, ls):
    """Matern-5/2 kernel, verbatim from refine_gp.py / train_ablation.py."""
    r = np.sqrt(np.maximum(((A[:, None] - B[None]) ** 2).sum(-1), 0)) / ls
    return (1 + np.sqrt(5) * r + 5 / 3 * r ** 2) * np.exp(-np.sqrt(5) * r)


def fit_gp(params, P):
    """Exact GP fit on P's TRAIN rows (features standardized on train,
    targets = standardized ln-deltas, kernel solved via Cholesky)."""
    from scipy.linalg import cho_factor, cho_solve
    tr = np.where(P["split"] == "train")[0]
    F = P["gpF"]
    fmu, fsd = F[tr].mean(0), F[tr].std(0) + 1e-9
    Z = (F - fmu) / fsd
    K = _matern(Z[tr], Z[tr], float(params["ls"]))
    c = cho_factor(K + float(params["noise"]) * np.eye(len(tr)))
    alpha = cho_solve(c, P["T"][tr])
    return {"family": "GP", "ls": float(params["ls"]),
            "noise": float(params["noise"]), "Z_tr": Z[tr],
            "alpha": alpha, "fmu": fmu, "fsd": fsd, "n_train": int(len(tr))}


def predict_gp(gp, P):
    """Predicted ln-flow levels for ALL rows of P."""
    F = P["gpF"]
    Z = (F - gp["fmu"]) / gp["fsd"]
    mu = _matern(Z, gp["Z_tr"], gp["ls"]) @ gp["alpha"]
    return P["last"][:, None] + mu * P["scale"]


def predict_ln(model, P, batch=2048):
    if isinstance(model, dict) and model.get("family") == "GP":
        return predict_gp(model, P)
    model.eval()
    Xt = torch.tensor(P["X"])
    out = []
    with torch.no_grad():
        for b in range(0, len(Xt), batch):
            out.append(model(Xt[b:b + batch]).numpy())
    Tp = np.concatenate(out)
    return P["last"][:, None] + Tp * P["scale"]


def train_one(family, params, P, seed=0, max_epochs=30, patience=8,
              report=None):
    """Train one model on P's train split; early stop on P's val split.

    report(epoch, val_rmse_log10) is called after every epoch if given.
    Returns (model, best_val_rmse_log10, epochs_run).
    """
    if family == "GP":
        gp = fit_gp(params, P)
        va = np.where(P["split"] == "val")[0]
        return gp, rmse_log10(P, predict_gp(gp, P), va), 1
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    window = int(params["window"])
    model = build_model(family, flatten(family, params), window)
    tr = np.where(P["split"] == "train")[0]
    va = np.where(P["split"] == "val")[0]
    Xt, Tt = torch.tensor(P["X"]), torch.tensor(P["T"])
    opt = torch.optim.Adam(model.parameters(), lr=float(params["lr"]),
                           weight_decay=float(params["wd"]))
    batch = int(params["batch"])
    rng = np.random.default_rng(seed)
    phys = family == "PILSTMv1"
    if phys:
        lam = float(params.get("lambda_phys", 0.1))
        prcp_t = torch.tensor(P["precip"], dtype=torch.float32)
        last_t = torch.tensor(P["last"], dtype=torch.float32)
        scale_t = torch.tensor(P["scale"], dtype=torch.float32)
    best, best_state, bad, eps = np.inf, None, 0, 0
    for ep in range(max_epochs):
        model.train()
        perm = rng.permutation(tr)
        for b in range(0, len(perm), batch):
            idx = perm[b:b + batch]
            opt.zero_grad()
            out = model(Xt[idx])
            loss = ((out - Tt[idx]) ** 2).mean()
            if phys:
                pred_lf = last_t[idx, None] + out * scale_t[idx]
                loss = loss + lam * physics_loss(pred_lf, prcp_t[idx],
                                                 last_t[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        r = rmse_log10(P, predict_ln(model, P), va)
        eps = ep + 1
        if report is not None:
            report(eps, r)
        if r < best - 1e-6:
            best, bad = r, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best), eps


def tune_trainable(config, station=None, family=None, proj=None,
                   max_epochs=30):
    """Ray Tune entry point. `config` is the flat search-space dict."""
    from ray import tune as _tune
    params = dict(config)
    P = build_station_P(station, int(params["window"]), "tune", proj=proj)
    t0 = time.time()
    _, best, eps = train_one(
        family, params, P, seed=0, max_epochs=max_epochs, patience=8,
        report=lambda ep, r: _tune.report({"rmse": r, "epoch": ep}))
    _tune.report({"rmse": best, "epoch": eps,
                  "train_s": time.time() - t0})
