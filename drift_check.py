#!/usr/bin/env python3
"""Pre-training drift check for HydroTwin (runs BEFORE training in the
weekly retrain workflow).

Data drift: PSI of each input feature at each station, baseline = train era
(2016-10-01..2023-09-30) vs the most recent 2 years of refreshed data.
A drifting input distribution is the earliest warning that model accuracy
(concept drift) will follow, so the report lands before any GPU/CPU time is
spent training, and the retrain can be prioritised per station.

Concept drift proper (forecast-error trend) is measured on the site side
from the served forecast series; see drift/drift_bake.json in hydrotwin-site.

Usage: python drift_check.py [--data data/daily] [--out drift_report.json]
       [--strict]   (strict: exit 1 if any station's worst PSI > 0.25)
Writes the JSON report and prints a one-line-per-station summary.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE = ("2016-10-01", "2023-09-30")
FEATURES = ["tmax_c", "tmin_c", "tmean_c", "precip_mm", "log10_flow"]


def psi(expected, actual, bins=10):
    e = np.asarray(expected, float); a = np.asarray(actual, float)
    e = e[np.isfinite(e)]; a = a[np.isfinite(a)]
    if len(e) < 100 or len(a) < 100:
        return None
    qs = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(qs) < 4:
        return None
    qs[0], qs[-1] = -np.inf, np.inf
    ec = np.clip(np.histogram(e, qs)[0] / len(e), 1e-4, None)
    ac = np.clip(np.histogram(a, qs)[0] / len(a), 1e-4, None)
    return float(np.sum((ac - ec) * np.log(ac / ec)))


def status(v):
    if v is None: return "no data"
    if v < 0.10: return "stable"
    if v <= 0.25: return "watch"
    return "drift"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/daily")
    ap.add_argument("--out", default="drift_report.json")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    files = sorted(Path(args.data).glob("*_daily.csv"))
    if not files:
        print(f"::error::no daily CSVs found under {args.data}")
        return 2
    report = {"baseline": BASE, "stations": {}}
    worst_overall = 0.0
    lines = []
    for f in files:
        st = f.stem.replace("_daily", "")
        d = pd.read_csv(f, parse_dates=["date"])
        d["log10_flow"] = np.log10(d["flow_cfs"].where(d["flow_cfs"] > 0))
        last = d["date"].max()
        recent_start = last - pd.DateOffset(years=2) + pd.Timedelta(days=1)
        b = d[(d.date >= BASE[0]) & (d.date <= BASE[1])]
        r = d[d.date >= recent_start]
        feats = {}
        for col in FEATURES:
            v = psi(b[col].to_numpy(), r[col].to_numpy())
            feats[col] = {"psi": None if v is None else round(v, 4), "status": status(v)}
        worst_f = max((c for c in FEATURES if feats[c]["psi"] is not None),
                      key=lambda c: feats[c]["psi"], default=None)
        wv = feats[worst_f]["psi"] if worst_f else None
        st_status = status(wv)
        worst_overall = max(worst_overall, wv or 0.0)
        report["stations"][st] = {"recent_window": [str(recent_start.date()), str(last.date())],
                                  "worst_feature": worst_f, "worst_psi": wv,
                                  "status": st_status, "features": feats}
        lines.append(f"{st}: {st_status} (worst {worst_f} PSI {wv})")
        if st_status == "drift":
            print(f"::warning::Station {st} input drift: {worst_f} PSI {wv} (> 0.25)")
    Path(args.out).write_text(json.dumps(report, indent=1))
    summary = "Drift check — " + "; ".join(lines)
    print(summary)
    gh_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if gh_summary:
        with open(gh_summary, "a") as fh:
            fh.write("## Pre-training drift check\n\n" + "\n\n".join(lines) + "\n")
    if args.strict and worst_overall > 0.25:
        print("::error::strict mode: drift above threshold")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
