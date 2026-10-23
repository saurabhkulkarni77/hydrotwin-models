# HydroTwin hyperparameter tuning (Optuna + Ray Tune) — DEV ONLY

Dynamic parameter optimization for the daily neural models, wired into
the dev environment only. Nothing here can touch production: winners
reach **dev weights** (`tuning/dev_weights/`) only through the enforced
quality gate, and production weights change solely through the
existing main-branch retrain flow on a human decision.

## Pieces

| file | role |
|---|---|
| `spaces.py` | Search spaces per family + the incumbent configs (current recipe from `train_export.py` / `best_lr.json`). |
| `architectures.py` | Tunable supersets of the repo architectures (`LSTMNetX`, `TransformerX`). Identical to the repo classes at incumbent knobs (state_dict-compatible). |
| `data_adapter.py` | Single-station datasets replicating `prepare9()` conventions (9 features: USGS + NRCS soil + SNOTEL snow, all data combined), with `tune` and `gate` folds. |
| `trainable.py` | Training loop mirroring repo `fit()` semantics; reports val RMSE (log10 cfs) per epoch. |
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
`.github/workflows/tuning-dev.yml`). The workflow tunes + gates both
families and commits registry entries, study journals, and any
gate-passed dev weights back to dev.

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
