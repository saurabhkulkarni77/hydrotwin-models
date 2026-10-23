#!/usr/bin/env python3
"""Build the public tuning status JSON from tuning/registry/*.json.

The HydroTwin station pages poll

    https://saurabhkulkarni77.github.io/hydrotwin-models/tuning_status.json

(the gh-pages branch of this repo — the same Pages source that serves
the ONNX weights) to render the "Optuna + Ray Tune" badge: an animated
RUNNING state while the tuning-dev workflow runs, otherwise the latest
quality-gate verdict per station/family from the registry.

The tuning-dev workflow calls this twice per run, both times with
continue-on-error so a status-publish failure can never fail tuning:

  * near the start:  --running "$STATION" --families "LSTM,Transformer"
  * at the end (if: always()): no --running flag, so the JSON carries
    the registry results the run just produced.

Usage:
  python tuning/write_status.py [--running STATION]
                                [--families LSTM,Transformer]
                                [--out PATH] [--registry DIR]

Stdlib only. The registry directory defaults to the one next to this
script, so it works from any cwd.
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone


def _latest_result(reg):
    """Latest gated result from one registry document, or None."""
    history = reg.get("history") or []
    for entry in reversed(history):
        gate = entry.get("gate") or {}
        if not gate.get("verdict"):
            continue
        return {
            "verdict": gate["verdict"],
            "margin_pct": gate.get("margin_pct"),
            "candidate_rmse": gate.get("candidate_rmse"),
            "incumbent_rmse": gate.get("incumbent_rmse"),
            "trials": entry.get("trials"),
            "best_val_rmse": entry.get("best_val_rmse_log10"),
            "holdout": "WY2026",
            "updated": gate.get("date") or reg.get("updated"),
        }
    return None


def build_status(registry_dir, running_station=None, families=None):
    stations = {}
    for path in sorted(glob.glob(os.path.join(registry_dir, "*.json"))):
        if os.path.basename(path) == "incumbents.json":
            continue  # reference numbers only, not a run registry
        try:
            with open(path, encoding="utf-8") as fh:
                reg = json.load(fh)
        except Exception:
            continue
        station, family = reg.get("station"), reg.get("family")
        result = _latest_result(reg)
        if not station or not family or result is None:
            continue
        stations.setdefault(station, {})[family] = result
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    current = None
    if running_station:
        current = {
            "station": running_station,
            "families": families or ["LSTM", "Transformer"],
            "since": now,
            "workflow": "tuning-dev",
        }
        run_id = os.environ.get("GITHUB_RUN_ID")
        repo = os.environ.get("GITHUB_REPOSITORY")
        if run_id and repo:
            current["run_url"] = (
                f"https://github.com/{repo}/actions/runs/{run_id}")
    return {
        "schema": 1,
        "generated_at": now,
        "running": bool(running_station),
        "current": current,
        "stations": stations,
        "source": "hydrotwin-models tuning/registry (dev branch), "
                  "published by tuning/write_status.py",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--running", metavar="STATION", default=None,
                    help="station id currently being tuned (sets the "
                         "running flag); omit for the idle snapshot")
    ap.add_argument("--families", default=None,
                    help="comma-separated families being tuned "
                         "(default: LSTM,Transformer)")
    ap.add_argument("--out", default=None,
                    help="output path (default: stdout)")
    ap.add_argument("--registry", default=None,
                    help="registry directory (default: registry/ next "
                         "to this script)")
    args = ap.parse_args(argv)
    registry_dir = args.registry or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "registry")
    families = None
    if args.families:
        families = [f.strip() for f in args.families.split(",") if f.strip()]
    status = build_status(registry_dir, args.running, families)
    text = json.dumps(status, indent=1) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"write_status: wrote {args.out} "
              f"(running={status['running']}, "
              f"stations={sorted(status['stations'])})")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
