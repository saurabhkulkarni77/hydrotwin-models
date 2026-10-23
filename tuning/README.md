# HydroTwin hyperparameter tuning (Optuna + Ray Tune) — DEV ONLY

Dynamic parameter optimization for the daily neural models, wired into
the dev environment only. Nothing here can touch production: winners
reach **dev weights** (`tuning/dev_weights/`) only through the enforced
quality gate, and production weights change solely through the
existing main-branch retrain flow on a human decision.

## Pieces

| file | role |
|---|---|
| `spaces.py` | Search spaces per family + the incumbent configs (current recipe from `train_export.py` / `best_lr.json`; GP from `classical_hyper.json`). |
| `architectures.py` | Tunable supersets of the repo architectures (`LSTMNetX`, `TransformerX`, `TFTX`, `PatchTSTX`). Identical to the repo classes at incumbent knobs (state_dict-compatible). PILSTMv1 is `LSTMNetX` (its production architecture is `LSTMNet`); GP is an exact GP, no torch module. |
| `data_adapter.py` | Single-station datasets replicating `prepare9()` conventions (9 features: USGS + NRCS soil + SNOTEL snow, all data combined), with `tune` and `gate` folds. Also carries `gpF` (the 11 GP kernel features on raw windows) and `precip` (issue-date precip for the PILSTMv1 physics loss). |
| `trainable.py` | Training loop mirroring repo `fit()` semantics; reports val RMSE (log10 cfs) per epoch. PILSTMv1 adds `train_pilstm.py`'s physics loss; GP is fit exactly (Matern-5/2, Cholesky) in one shot. |
| `tune_models.py` | Ray Tune + OptunaSearch (TPE) + ASHAScheduler. Studies persist in `studies/*.journal` (append-only, resumable, committed). The incumbent config is evaluated as trial 0 of every study. |
| `gate.py` | **Enforced champion/challenger quality gate** (below). |
| `registry/` | Versioned JSON per (station, family): winner params, tuned val RMSE, incumbent val RMSE, gate verdicts. `incumbents.json` holds the published protocol-B backtest references. |

## Objective + folds (water-year convention)

- Tuning objective: validation RMSE in **log10(cfs)** on **WY2025**
  (train ≤ 2024-09-30) — the same metric family as the drift panel's
  A-vs-B backtest.
- Gate holdout: **WY2026**, untouched by tuning, fitting, stopping, or
  selection (both models train on data ≤ 2025-09-30, early-stop on the
  Apr–Sep 2025 tail) — identical-holdout champion/challenger.

## The quality gate (enforced, not advisory)

`gate.py` exits **1** when the candidate does not beat the incumbent:

- FAIL → no file is copied; incumbent dev weights stay; the job/step
  calling the gate fails. Verdict (candidate RMSE, incumbent RMSE,
  margin, PASS/FAIL) is written to the registry entry and to the
  GitHub run summary before the failure is raised.
- PASS → candidate checkpoint + config are written to
  `tuning/dev_weights/<family>/<station>.pt|.json` — **dev only**.

## Drift hookup

The dev retrain workflow runs `drift_check.py` before training.
`tune_models.py --from-drift-report ../drift_report.json` tunes only
stations whose worst-feature PSI exceeds 0.25 (the project's "drift"
cutoff), so re-optimization follows the drift signal instead of a
blind schedule.

## Run it

Supported path: push a `tuning/**` change to the dev branch (or, once
this file also exists on the default branch, dispatch
`.github/workflows/tuning-dev.yml`). The workflow tunes + gates all
six families — LSTM, Transformer, TFT, PatchTST, PILSTMv1, GP — and
commits registry entries, study journals, and any gate-passed dev
weights back to dev. GP note: GP Matérn is not one of the five served
ONNX models — it is the Compare-panel / ablation model (exact GP
regression on 11 window-summary features, per-station `ls`/`noise` in
`classical_hyper.json`); it is tuned and gated under the identical
protocol, and a PASS publishes a config JSON (kernel + feature stats)
instead of a `.pt`, since an exact GP has no torch weights.

Locally (any normal host):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r tuning/requirements-tuning.txt
export HYDROTWIN_PROJ=$PWD HYDROTWIN_NRCS_XLSX=$PWD/data/nrcs_export.xlsx
python tuning/tune_models.py --station 08057000 --family LSTM --trials 16
python tuning/gate.py --station 08057000 --family LSTM   # exits 1 on FAIL
```

Dev-VM caveat: Ray advertises the host's primary interface IP for all
internal traffic and converts loopback node IPs back to it by design;
the agent VM's runtime intercepts non-loopback TCP, so `ray.init()`
cannot complete there (verified Oct 22, 2026 — GCS dials the host IP
and times out). That is why the Actions workflow is the execution
path; the identical code runs unchanged in both places.

<!-- tuning batch trigger: station 01646500 (drift PSI > 0.25 batch, Oct 22-23 2026) -->

<!-- tuning batch trigger: station 02320500 (drift PSI > 0.25 batch, Oct 22-23 2026) -->

<!-- tuning batch trigger: station 05331000 (drift PSI > 0.25 batch, Oct 22-23 2026) -->

<!-- tuning batch trigger: station 06214500 (drift PSI > 0.25 batch, Oct 22-23 2026) -->

<!-- tuning batch trigger: station 06934500 (drift PSI > 0.25 batch, Oct 22-23 2026) -->

<!-- tuning batch trigger: station 07374000 (drift PSI > 0.25 batch, Oct 22-23 2026) -->

<!-- tuning batch trigger: station 08158000 (drift PSI > 0.25 batch, Oct 22-23 2026) -->

<!-- tuning batch trigger: station 08313000 (drift PSI > 0.25 batch, Oct 22-23 2026) -->

<!-- tuning batch trigger: station 09380000 (drift PSI > 0.25 batch, Oct 22-23 2026) -->

<!-- tuning batch trigger: station 11447650 (drift PSI > 0.25 batch, Oct 22-23 2026) -->

<!-- tuning batch trigger: station 14105700 (drift PSI > 0.25 batch, Oct 22-23 2026) -->

<!-- tuning batch trigger: station 01463500 (all-models expansion sweep, Oct 23 2026) -->

<!-- tuning batch trigger: station 01646500 (all-models expansion sweep, Oct 23 2026) -->

<!-- tuning batch trigger: station 02320500 (all-models expansion sweep, Oct 23 2026) -->

<!-- tuning batch trigger: station 03294500 (all-models expansion sweep, Oct 23 2026) -->

<!-- tuning batch trigger: station 05331000 (all-models expansion sweep, Oct 23 2026) -->

<!-- tuning batch trigger: station 06214500 (all-models expansion sweep, Oct 23 2026) -->

<!-- tuning batch trigger: station 06934500 (all-models expansion sweep, Oct 23 2026) -->
