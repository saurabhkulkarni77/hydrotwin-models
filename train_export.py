#!/usr/bin/env python3
"""Train the 4 neural architectures per station and save checkpoints + stats for ONNX export.

Replicates train_per_station_opt.main() exactly (same fit(), seed 0, patience 7,
wd 1e-5, 60 epochs) but with a single LR per model/station: the validation-picked
best_lr from ml/results/per_station_results_opt.json. Uses ORIGINAL splits
(defaults 2023-09-30 / 2024-09-30); any HYDROTWIN_* env overrides are cleared.

Resume-safe: skips stations whose checkpoint + stats files already exist.
"""
import sys, json, time, os
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
for v in ("HYDROTWIN_TRAIN_END", "HYDROTWIN_VAL_END", "HYDROTWIN_TEST_START"):
    os.environ.pop(v, None)

from train_lstm_normal import prepare, fit, LSTMNet
from optimize_tr_gcn import TransformerFlex
from train_sota8 import TFTLite
from train_new_arch import PatchTST
from train_pilstm_v2 import PILSTMv2

PROJ = Path(os.environ.get("HYDROTWIN_PROJ", Path(__file__).resolve().parent))
RES = PROJ / "ml" / "results"
CKPT = PROJ / "ml" / "checkpoints"

MODELS = [
    ("LSTM", LSTMNet),
    ("Transformer", lambda: TransformerFlex(d=16, layers=1, heads=2)),
    ("TFT", TFTLite),
    ("PatchTST", PatchTST),
    ("PILSTMv2", PILSTMv2),
    ("PILSTMv1", LSTMNet),  # v1 shares LSTM arch; physics in loss (see train_pilstm.py)
]
WD, MAX_EPOCHS, PATIENCE = 1e-5, 60, 7


def subset(P, si):
    m = P["sid"] == si
    return {k: (v[m] if isinstance(v, np.ndarray) and len(v) == len(P["X"]) else v)
            for k, v in P.items()}


def main():
    torch.set_num_threads(2)
    print("building windows...", flush=True)
    P = prepare()
    print("windows:", len(P["X"]), "| stations:", list(P["sites"]), flush=True)
    blr_path = Path(__file__).resolve().parent / "best_lr.json"
    if blr_path.exists():
        res = json.loads(blr_path.read_text())
        print("using best_lr.json for learning rates", flush=True)
    else:
        res = json.loads((RES / "per_station_results_opt.json").read_text())
    done, skipped = 0, 0
    for name, fn in MODELS:
        for si, site in enumerate(P["sites"]):
            out = CKPT / name / f"{site}.pt"
            stf = CKPT / name / f"{site}_stats.json"
            if out.exists() and stf.exists():
                skipped += 1
                continue
            try:
                best_lr = res[name][site]["best_lr"]
            except (KeyError, TypeError):
                print(f"{name} {site}: no best_lr in results, skip", flush=True)
                continue
            Ps = subset(P, si)
            if np.sum(Ps["split"] == "train") < 200:
                print(f"{name} {site}: too little train data, skip", flush=True)
                continue
            t0 = time.time()
            torch.manual_seed(0); np.random.seed(0)
            m = fn()
            m, bv, eps = fit(m, Ps, lr=best_lr, wd=WD, epochs=MAX_EPOCHS,
                             patience=PATIENCE, seed=0)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(m.state_dict(), out)
            stf.write_text(json.dumps(P["stats"][site]))
            done += 1
            print(f"{name} {site}: saved val={bv:.4f} ep={eps} "
                  f"[{time.time() - t0:.0f}s] ({done} done)", flush=True)
    print(f"EXPORT-TRAIN DONE: {done} trained, {skipped} already had checkpoints")


if __name__ == "__main__":
    main()
