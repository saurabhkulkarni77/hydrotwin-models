"""Search spaces + incumbent configs for HydroTwin tuning.

Incumbents are the CURRENT dev recipe from train_export.py + best_lr.json:
LSTMNet(hidden=16, dropout=0) / TransformerFlex(d=16, layers=1, heads=2),
window 29, batch 256, wd 1e-5, per-family best_lr for the station.
Spaces are bounded for CPU tractability; widen in the Actions workflow.
"""
import optuna

# (d_model, heads) pairs with heads dividing d_model.
TR_PAIRS = [(16, 2), (16, 4), (32, 2), (32, 4), (32, 8),
            (48, 2), (48, 4), (64, 2), (64, 4), (64, 8)]

WINDOWS = [14, 21, 29, 45, 60]
BATCHES = [128, 256, 512]


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
    if family == "Transformer":
        return {**common,
                "d_heads": C([f"{d}_{h}" for d, h in TR_PAIRS]),
                "layers": I(1, 3),
                "dropout": F(0.0, 0.3)}
    raise ValueError(family)


def incumbent_for(family, best_lr):
    """Incumbent config in the SAME flat format trials report.

    best_lr: per-station learning rate from best_lr.json for this family.
    """
    base = {"lr": float(best_lr), "wd": 1e-5, "batch": 256, "window": 29,
            "layers": 1, "dropout": 0.0}
    if family == "LSTM":
        return {**base, "hidden": 16}
    if family == "Transformer":
        return {**base, "d_heads": "16_2"}
    raise ValueError(family)


def flatten(family, cfg):
    """Expand a stored config into build_model() params."""
    out = dict(cfg)
    if family == "Transformer":
        d, h = out.pop("d_heads").split("_")
        out["d"], out["heads"] = int(d), int(h)
    return out
