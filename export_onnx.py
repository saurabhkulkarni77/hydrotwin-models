#!/usr/bin/env python3
"""Export the 56 trained checkpoints to ONNX and verify each against PyTorch.

Reads ml/checkpoints/<Model>/<site>.pt, exports to ml/onnx/<site>/<Model>.onnx
(input "window", dynamic batch x 29 x 6), then runs onnxruntime on one REAL
window from prepare() and requires max abs diff vs PyTorch < 1e-4.
"""
import sys, json, os
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lstm_normal import prepare, LSTMNet
from optimize_tr_gcn import TransformerFlex
from train_sota8 import TFTLite
from train_new_arch import PatchTST
from train_pilstm_v2 import PILSTMv2
import onnxruntime as ort

PROJ = Path(os.environ.get("HYDROTWIN_PROJ", Path(__file__).resolve().parent))
CKPT = PROJ / "ml" / "checkpoints"
ONNXD = PROJ / "ml" / "onnx"

MODELS = [
    ("LSTM", LSTMNet),
    ("Transformer", lambda: TransformerFlex(d=16, layers=1, heads=2)),
    ("TFT", TFTLite),
    ("PatchTST", PatchTST),
    ("PILSTMv2", PILSTMv2),
    ("PILSTMv1", LSTMNet),
]


def main(out_dir=None):
    global ONNXD
    if out_dir:
        ONNXD = Path(out_dir)
    torch.set_num_threads(2)
    P = prepare()
    site_idx = {s: i for i, s in enumerate(P["sites"])}
    ok, fail = 0, []
    for name, fn in MODELS:
        for site in P["sites"]:
            pt = CKPT / name / f"{site}.pt"
            dst = ONNXD / site / f"{name}.onnx"
            if not pt.exists():
                fail.append((name, site, "missing checkpoint"))
                continue
            m = fn()
            m.load_state_dict(torch.load(pt, map_location="cpu"))
            m.eval()
            si = site_idx[site]
            xw = P["X"][P["sid"] == si][:1]  # one real standardized window
            dst.parent.mkdir(parents=True, exist_ok=True)
            torch.onnx.export(m, torch.tensor(xw), str(dst),
                              input_names=["window"], output_names=["out"],
                              dynamic_axes={"window": {0: "batch"}, "out": {0: "batch"}},
                              opset_version=17, dynamo=False)
            sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
            y_ort = sess.run(None, {"window": xw.astype(np.float32)})[0]
            with torch.no_grad():
                y_pt = m(torch.tensor(xw)).numpy()
            d = float(np.abs(y_ort - y_pt).max())
            good = y_ort.shape == (1, 7) and np.isfinite(y_ort).all() and d < 1e-4
            if good:
                ok += 1
            else:
                fail.append((name, site, f"diff={d:.2e} shape={y_ort.shape}"))
            print(f"{name} {site}: diff={d:.2e} -> {'OK' if good else 'FAIL'}", flush=True)
    print(f"ONNX EXPORT DONE: {ok} ok, {len(fail)} failed")
    for f in fail:
        print("  FAILED:", f, flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output dir for onnx files")
    main(ap.parse_args().out)
