#!/usr/bin/env python3
"""Champion/challenger QUALITY GATE — enforced in the dev path.

For one (station, family): takes the latest tuned winner from
tuning/registry/, retrains incumbent and challenger under the gate
protocol, and backtests both on the IDENTICAL untouched holdout
(WY2026 windows, RMSE in log10 cfs — the drift-fix A-vs-B convention).

Gate protocol: normalization stats and training span = all data with
first-forecast date <= 2025-09-30; early stopping on the Apr-Sep 2025
tail; WY2026 is never used for fitting, stopping, or selection.

ENFORCEMENT:
- PASS (candidate RMSE < incumbent RMSE): the candidate checkpoint is
  copied to tuning/dev_weights/<family>/<station>.pt (+ config JSON).
  Dev weights only — production is never touched by this script.
- FAIL: NOTHING is copied (incumbent dev weights stay), the verdict is
  still written to the registry and the run summary, and this script
  exits 1 so the calling workflow step/job FAILS.

Usage: python gate.py --station 08057000 --family LSTM [--epochs 60]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from data_adapter import build_station_P, rmse_log10  # noqa: E402
from trainable import predict_ln, train_one  # noqa: E402
from tune_models import incumbent  # noqa: E402


def evaluate(station, family, params, proj, epochs, patience):
    P = build_station_P(station, int(params["window"]), "gate", proj=proj)
    model, _, eps = train_one(family, params, P, seed=0,
                              max_epochs=epochs, patience=patience)
    te = np.where(P["split"] == "test")[0]
    rmse = rmse_log10(P, predict_ln(model, P), te)
    return model, rmse, len(te), eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--proj", default=os.environ.get(
        "HYDROTWIN_PROJ", str(HERE.parent)))
    args = ap.parse_args()
    args.proj = str(Path(args.proj).resolve())

    reg_path = HERE / "registry" / f"{args.station}_{args.family}.json"
    if not reg_path.exists():
        print(f"::error::no registry entry for {args.station} "
              f"{args.family} — run tune_models.py first")
        return 2
    entry = json.loads(reg_path.read_text())
    run = entry["history"][-1]
    cand = dict(run["best_params"])
    inc = incumbent(args.proj, args.family, args.station)

    t0 = time.time()
    inc_model, inc_rmse, n_te, inc_eps = evaluate(
        args.station, args.family, inc, args.proj, args.epochs,
        args.patience)
    cand_model, cand_rmse, n_te2, cand_eps = evaluate(
        args.station, args.family, cand, args.proj, args.epochs,
        args.patience)
    margin = inc_rmse - cand_rmse
    passed = bool(cand_rmse < inc_rmse)

    verdict = {
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "holdout": "WY2026 (first-forecast 2025-10-01..2026-09-30), "
                   "RMSE log10(cfs)",
        "protocol": "train span <= 2025-09-30, early stop Apr-Sep 2025",
        "n_holdout_windows_incumbent": int(n_te),
        "n_holdout_windows_candidate": int(n_te2),
        "incumbent_rmse": round(float(inc_rmse), 4),
        "candidate_rmse": round(float(cand_rmse), 4),
        "margin": round(float(margin), 4),
        "margin_pct": round(float(100 * margin / inc_rmse), 2),
        "incumbent_epochs": inc_eps, "candidate_epochs": cand_eps,
        "verdict": "PASS" if passed else "FAIL",
        "published_protocol_B_reference_rmse": json.loads(
            (HERE / "registry" / "incumbents.json").read_text())
        ["stations"].get(args.station, {}).get(args.family, {})
        .get("published_protocol_B_rmse_log10_WY2026"),
    }
    run["gate"] = verdict
    entry["updated"] = verdict["date"]
    reg_path.write_text(json.dumps(entry, indent=1))

    if passed:
        out_dir = HERE / "dev_weights" / args.family
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.family == "GP":
            # An exact GP has no torch weights: its "checkpoint" is the
            # winning kernel config + the feature standardization the
            # fit used (the training rows themselves stay in the repo
            # dataset, reproducible from data/daily + the NRCS LUT).
            (out_dir / f"{args.station}.json").write_text(json.dumps(
                {"station": args.station, "family": args.family,
                 "params": cand, "gate": verdict,
                 "gp_fit": {"n_train": cand_model["n_train"],
                            "feature_mu": cand_model["fmu"].tolist(),
                            "feature_sd": cand_model["fsd"].tolist()}},
                indent=1))
        else:
            torch.save(cand_model.state_dict(),
                       out_dir / f"{args.station}.pt")
            (out_dir / f"{args.station}.json").write_text(json.dumps(
                {"station": args.station, "family": args.family,
                 "params": cand, "gate": verdict}, indent=1))
        action = f"candidate promoted to DEV weights {out_dir}"
    else:
        action = "incumbent dev weights kept; candidate NOT promoted"

    line = (f"QUALITY GATE {args.station} {args.family}: "
            f"candidate {verdict['candidate_rmse']} vs incumbent "
            f"{verdict['incumbent_rmse']} (margin "
            f"{verdict['margin']:+}, {verdict['margin_pct']:+}%) -> "
            f"{verdict['verdict']}; {action}")
    print(line, flush=True)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write(f"## Quality gate\n\n{line}\n")
    print(f"gate took {time.time() - t0:.0f}s", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
