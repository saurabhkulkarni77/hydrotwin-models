#!/usr/bin/env python3
"""Optuna + Ray Tune hyperparameter tuning for HydroTwin — DEV ONLY.

Optuna (via Ray Tune's OptunaSearch, TPE sampler) proposes configs;
ASHAScheduler races them, culling the bottom half at each rung.
Studies persist in an append-only journal under tuning/studies/ so
they accumulate and resume across runs (local or Actions).

Drift hookup: the dev retrain workflow already runs drift_check.py
first. Pass --from-drift-report drift_report.json and only stations
whose worst-feature PSI > --psi-threshold (default 0.25, the project
"drift" cutoff) are tuned — tuning compute follows the drift signal.

Winners are written to tuning/registry/<station>_<family>.json.
NOTHING here touches served weights: promotion to dev weights happens
only through gate.py, and production is never touched.

Usage:
  python tune_models.py --station 08057000 --family LSTM --trials 16
  python tune_models.py --from-drift-report ../drift_report.json \
      --family Transformer --trials 12
"""
import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

SEED = 0  # repo convention (train_export.py seeds 0)


def study_storage(study):
    (HERE / "studies").mkdir(exist_ok=True)
    from optuna.storages import JournalStorage
    try:
        from optuna.storages.journal import JournalFileBackend
        backend = JournalFileBackend(str(HERE / "studies" / f"{study}.journal"))
    except ImportError:  # optuna 3.x
        from optuna.storages import JournalFileStorage
        backend = JournalFileStorage(str(HERE / "studies" / f"{study}.journal"))
    return JournalStorage(backend)


def init_ray(proj):
    """In-process Ray cluster.

    NOTE (dev VM constraint): Ray advertises the host's primary
    interface IP for all internal traffic and refuses loopback node
    IPs by design; the dev VM's runtime intercepts non-loopback TCP,
    so ray.init() cannot complete there. Run this on a normal host —
    the tuning-dev GitHub Actions workflow is the supported path —
    or any workstation. No code change is needed: the same entry
    point runs in both places.
    """
    import ray
    ray.init(ignore_reinit_error=True, log_to_driver=False,
             include_dashboard=False,
             runtime_env={"env_vars": {"PYTHONPATH": f"{proj}:{HERE}"}})


def load_best_lr(proj, family, station):
    res = json.loads((Path(proj) / "best_lr.json").read_text())
    v = res[family][station]
    return float(v["best_lr"] if isinstance(v, dict) else v)


def run_study(station, family, trials, max_epochs, proj, storage_path):
    import optuna
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch

    from spaces import incumbent_for, space_for
    from trainable import tune_trainable

    study = f"hydrotwin-{station}-{family.lower()}"
    inc = incumbent_for(family, load_best_lr(proj, family, station))
    algo = OptunaSearch(
        space_for(family), metric="rmse", mode="min",
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=5),
        storage=study_storage(study), study_name=study,
        points_to_evaluate=[inc])
    scheduler = ASHAScheduler(max_t=max_epochs, grace_period=4,
                              reduction_factor=2)
    trainable = functools.partial(tune_trainable, station=station,
                                  family=family, proj=str(proj),
                                  max_epochs=max_epochs)
    init_ray(proj)
    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(num_samples=trials,
                                    max_concurrent_trials=2,
                                    search_alg=algo, scheduler=scheduler),
        run_config=tune.RunConfig(
            name=study, storage_path=str(storage_path),
            failure_config=tune.FailureConfig(max_failures=2),
            verbose=1))
    results = tuner.fit()
    best = results.get_best_result("rmse", "min")
    return study, inc, best


def write_registry(station, family, study, trials, best_cfg, best_rmse,
                   incumbent_cfg, incumbent_val_rmse):
    reg_dir = HERE / "registry"
    reg_dir.mkdir(exist_ok=True)
    path = reg_dir / f"{station}_{family}.json"
    entry = {"schema": 1, "station": station, "family": family,
             "history": []}
    if path.exists():
        entry = json.loads(path.read_text())
    run = {
        "run_id": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "study": study, "trials": trials, "seed": SEED,
        "objective": "val RMSE log10(cfs), WY2025, tune fold "
                     "(train <= 2024-09-30)",
        "best_params": {k: best_cfg[k] for k in sorted(best_cfg)},
        "best_val_rmse_log10": round(float(best_rmse), 4),
        "incumbent_params": incumbent_cfg,
        "incumbent_val_rmse_log10": incumbent_val_rmse,
        "gate": None,
    }
    entry["history"].append(run)
    entry["updated"] = run["run_id"]
    path.write_text(json.dumps(entry, indent=1))
    return path, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", action="append", default=[])
    ap.add_argument("--family", action="append", required=True,
                    choices=["LSTM", "Transformer", "TFT", "PatchTST"])
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--max-epochs", type=int, default=24)
    ap.add_argument("--from-drift-report")
    ap.add_argument("--psi-threshold", type=float, default=0.25)
    ap.add_argument("--proj", default=os.environ.get(
        "HYDROTWIN_PROJ", str(HERE.parent)))
    ap.add_argument("--storage-path",
                    default=str(Path.home() / "ray_tune_runs"))
    args = ap.parse_args()

    stations = list(args.station)
    if args.from_drift_report:
        rep = json.loads(Path(args.from_drift_report).read_text())
        drifted = [s for s, v in rep["stations"].items()
                   if (v.get("worst_psi") or 0) > args.psi_threshold]
        print(f"drift report: {len(drifted)} station(s) above PSI "
              f"{args.psi_threshold}: {drifted}", flush=True)
        stations = drifted or stations
    if not stations:
        print("no stations to tune", flush=True)
        return 0

    args.proj = str(Path(args.proj).resolve())
    rc = 0
    for station in stations:
        for family in args.family:
            if family not in ("LSTM", "Transformer"):
                print(f"{family}: trainable not wired yet (spaces + "
                      f"architectures cover LSTM/Transformer in v1); "
                      f"skipping", flush=True)
                continue
            print(f"=== tuning {station} {family}: {args.trials} trials "
                  f"===", flush=True)
            t0 = time.time()
            study, inc, best = run_study(station, family, args.trials,
                                         args.max_epochs, args.proj,
                                         args.storage_path)
            # Incumbent's own score inside the study: Optuna evaluated it
            # as trial 0 (points_to_evaluate), so read it back honestly.
            import optuna
            st = optuna.load_study(study_name=study,
                                   storage=study_storage(study))
            inc_val = None
            for t in st.trials:
                if t.params == inc:
                    inc_val = t.value
            path, run = write_registry(
                station, family, study, len(st.trials), best.config,
                best.metrics["rmse"], inc, inc_val)
            print(f"{station} {family}: best val RMSE "
                  f"{run['best_val_rmse_log10']} params "
                  f"{run['best_params']} | incumbent val RMSE {inc_val} "
                  f"| {time.time() - t0:.0f}s | registry {path}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
