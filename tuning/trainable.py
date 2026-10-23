"""Training loop + Ray Tune trainable.

train_one() mirrors the repo fit() semantics (Adam, MSE on standardized
ln-delta targets, grad-clip 1.0, best-checkpoint restore) with one
deliberate change: checkpoint selection and per-epoch reporting use the
objective metric itself — validation RMSE in log10(cfs) — so ASHA races
trials on the number the quality gate will judge.
"""
import time

import numpy as np
import torch
import torch.nn as nn

from architectures import build_model
from data_adapter import build_station_P, rmse_log10
from spaces import flatten


def predict_ln(model, P, batch=2048):
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
    best, best_state, bad, eps = np.inf, None, 0, 0
    for ep in range(max_epochs):
        model.train()
        perm = rng.permutation(tr)
        for b in range(0, len(perm), batch):
            idx = perm[b:b + batch]
            opt.zero_grad()
            loss = ((model(Xt[idx]) - Tt[idx]) ** 2).mean()
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
