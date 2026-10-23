"""Search spaces + incumbent configs for HydroTwin tuning.

Incumbents are the CURRENT dev recipe from train_export.py + best_lr.json:
LSTMNet(hidden=16, dropout=0) / TransformerFlex(d=16, layers=1, heads=2) /
TFTLite(d=48, heads=4) / PatchTST(d=64, patch=7, layers=2, heads=4),
window 29, batch 256, wd 1e-5, per-family best_lr for the station.
PILSTMv1 = LSTMNet(hidden=16) + physics loss (train_pilstm.py), lr 3e-3
(best_lr.json has no PILSTMv1 entry; train_export.py falls back to 3e-3
and train_pilstm.py hardcodes 3e-3). GP = exact GP regression with a
Matern-5/2 kernel (refine_gp.py / train_ablation.py): the incumbent is
the per-station (ls, noise) in classical_hyper.json — the settings
export_params.py ships to the pages' Compare panel — on the fixed
29-day window its 11 kernel features are defined on.
Spaces are bounded for CPU tractability; widen in the Actions workflow.
"""
import json
from pathlib import Path

import optuna

# (d_model, heads) pairs with heads dividing d_model.
TR_PAIRS = [(16, 2), (16, 4), (32, 2), (32, 4), (32, 8),
            (48, 2), (48, 4), (64, 2), (64, 4), (64, 8)]
TFT_PAIRS = [(24, 2), (24, 4), (32, 2), (32, 4), (48, 2), (48, 4),
             (48, 6), (64, 2), (64, 4), (64, 8), (96, 4), (96, 8)]
PATCH_PAIRS = [(32, 2), (32, 4), (64, 2), (64, 4), (64, 8),
               (96, 4), (96, 8), (128, 4), (128, 8)]

WINDOWS = [14, 21, 29, 45, 60]
BATCHES = [128, 256, 512]

ALL_FAMILIES = ["LSTM", "Transformer", "TFT", "PatchTST", "PILSTMv1", "GP"]


def space_for(family):
    """OptunaSearch space (dict of optuna distributions)."""
    F = optuna.distributions.FloatDistribution
    I = optuna.distributions.IntDistribution
    C = optuna.distributions.CategoricalDistribution
    common = {
        "lr": F(1e-4, 1e-2, log=True),
        "wd": F(1e-6, 1e-3, log=True),
        "batch": C(BATCHES),
        "window": C(WINDOWS),
    }
    if family == "LSTM":
        return {**common,
                "hidden": C([16, 32, 48, 64, 96]),
                "layers": I(1, 3),
                "dropout": F(0.0, 0.4)}
    if family == "PILSTMv1":
        return {**common,
                "hidden": C([16, 32, 48, 64]),
                "layers": I(1, 2),
                "dropout": F(0.0, 0.3),
                # physics-loss weight (train_pilstm.py LAMBDA_PHYS=0.1)
                "lambda_phys": F(0.01, 0.5, log=True)}
    if family == "Transformer":
        return {**common,
                "d_heads": C([f"{d}_{h}" for d, h in TR_PAIRS]),
                "layers": I(1, 3),
                "dropout": F(0.0, 0.3)}
    if family == "TFT":
        return {**common,
                "d_heads": C([f"{d}_{h}" for d, h in TFT_PAIRS]),
                "layers": I(1, 2),
                "dropout": F(0.0, 0.3)}
    if family == "PatchTST":
        return {**common,
                "d_heads": C([f"{d}_{h}" for d, h in PATCH_PAIRS]),
                "patch": C([5, 7, 10, 14]),
                "layers": I(1, 3),
                "dropout": F(0.0, 0.3)}
    if family == "GP":
        # Exact GP (Matern-5/2): only the kernel length scale and the
        # noise level exist (production kernel has unit signal variance).
        # Window is fixed at 29 — the 11 kernel features are defined on
        # 29-day windows in refine_gp.py / train_ablation.py.
        return {"ls": F(0.25, 64.0, log=True),
                "noise": F(1e-3, 3.0, log=True),
                "window": C([29])}
    raise ValueError(family)


def incumbent_for(family, best_lr):
    """Incumbent config in the SAME flat format trials report.

    best_lr: per-station learning rate from best_lr.json for this family
    (PILSTMv1: the 3e-3 production fallback — see module docstring).
    GP does not come through here; use gp_incumbent_for().
    """
    base = {"lr": float(best_lr), "wd": 1e-5, "batch": 256, "window": 29,
            "layers": 1, "dropout": 0.0}
    if family == "LSTM":
        return {**base, "hidden": 16}
    if family == "PILSTMv1":
        return {**base, "hidden": 16, "lambda_phys": 0.1}
    if family == "Transformer":
        return {**base, "d_heads": "16_2"}
    if family == "TFT":
        return {**base, "d_heads": "48_4"}
    if family == "PatchTST":
        return {**base, "d_heads": "64_4", "patch": 7, "layers": 2}
    raise ValueError(family)


def gp_incumbent_for(proj, station):
    """GP incumbent: per-station (ls, noise) from classical_hyper.json —
    the refined production settings export_params.py ships to the site."""
    hyp = json.loads((Path(proj) / "classical_hyper.json").read_text())
    g = hyp[station]["GP_Matern"]
    return {"ls": float(g["ls"]), "noise": float(g["noise"]), "window": 29}


def flatten(family, cfg):
    """Expand a stored config into build_model() params."""
    out = dict(cfg)
    if family in ("Transformer", "TFT", "PatchTST"):
        d, h = out.pop("d_heads").split("_")
        out["d"], out["heads"] = int(d), int(h)
    return out
