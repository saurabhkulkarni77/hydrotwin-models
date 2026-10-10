"""Refine GP hyperparameters: finer lengthscale grid around current best + noise refinement.
For each (river, kernel): try ls in {0.5, 0.7, 1.0, 1.4, 2.0} x current best (extended if at boundary),
and noise in {0.3, 1.0, 3.0} x current noise. Pick by validation RMSE.
Saves refined settings + new predictions."""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve

sys.path.insert(0, "/home/hatch/workspace/user/files/HydroTwin_10yr_Map/ml")
# patch the /home/claude import inside gp10
import importlib.util
spec = importlib.util.spec_from_file_location("gp10", "/home/hatch/workspace/user/files/HydroTwin_10yr_Map/ml/gp10.py")
# gp10 imports data10 which imports /home/claude/ml10 — stub it
import types
sys.modules["data10"] = types.ModuleType("data10")
gp10 = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(gp10)
except ImportError:
    pass  # data10 stub may break features; define locally instead

# --- local copies (avoid the /home/claude import chain) ---
def matern(A, B, ls):
    r = np.sqrt(np.maximum(((A[:, None] - B[None]) ** 2).sum(-1), 0)) / ls
    return (1 + np.sqrt(5) * r + 5 / 3 * r ** 2) * np.exp(-np.sqrt(5) * r)

def rbf(A, B, ls):
    return np.exp(-((A[:, None] - B[None]) ** 2).sum(-1) / (2 * ls ** 2))

def gp_pred(Ktr, Kte, Y, noise):
    c = cho_factor(Ktr + noise * np.eye(len(Ktr)))
    return Kte @ cho_solve(c, Y)

def features(X):
    """Summary features from a raw window X (N,29,C).
    8 original features from the 6 USGS channels + (when present) the 3
    soil/snow channels 6,7,8 at the last timestep (issue-date values):
    [last log flow, 1-day change, 7-day change, last rain, 3-day rain sum,
     7-day rain sum, 29-day rain sum, 7-day temp mean,
     soil_sms_8in_pct, soil_temp_2in_F, swe_in] -> (N,11).
    Old 6-channel windows still return the original 8 features."""
    lf = np.log(X[:, :, 4]); rain = X[:, :, 3]; tm = X[:, :, 2]
    base = np.stack([lf[:, -1], lf[:, -1] - lf[:, -2], lf[:, -1] - lf[:, -8], rain[:, -1], rain[:, -3:].sum(1),
                     rain[:, -7:].sum(1), rain.sum(1), tm[:, -7:].mean(1)], 1)
    if X.shape[2] >= 9:
        return np.concatenate([base, X[:, -1, 6:9]], 1)
    return base

PROJ = Path("/home/hatch/workspace/user/files/HydroTwin_10yr_Map")
RES = PROJ / "ml" / "results"
sys.path.insert(0, "/home/hatch/workspace/hydrotwin_colab")
from train_lstm_normal import prepare9, load_soilsnow_lut


def soil_triple(site, issue, lut):
    """Raw (sms_8in_pct, soil_temp_2in_F, swe_in) for an issue date; NaN kept for median fallback."""
    v = lut.get((site, issue))
    if v is None:
        return (np.nan, np.nan, 0.0)
    return v


def main():
    P = prepare9()
    lut = load_soilsnow_lut()
    G = json.load(open(RES / "gp10_results.json"))["stations"]
    refined = {}
    results = {"stations": {}, "features": 11, "split": "train/val/test per refine_gp (issue-date based)"}
    for si, site in enumerate(P["sites"]):
        print(f"=== {site} ===", flush=True)
        # rebuild raw windows like before (6 USGS channels), then attach soil/snow as raw ch 6-8
        import pandas as pd
        d = pd.read_csv(PROJ / "data" / "daily" / f"{site}_daily.csv", parse_dates=["date"])
        cols = ["tmax_c", "tmin_c", "tmean_c", "precip_mm", "flow_cfs", "rising"]
        v = d[cols].to_numpy(float)
        ok = ~np.isnan(v).any(1)
        WIN, LEADS = 29, 7
        Xs, Ys, iss, sps = [], [], [], []
        TRAIN_END = pd.Timestamp("2023-09-30"); TEST_START = pd.Timestamp("2024-09-30")
        lfall = np.log(d.flow_cfs.to_numpy(float))
        for t in range(WIN - 1, len(d) - LEADS):
            if not ok[t - WIN + 1:t + 1].all(): continue
            lf = lfall
            if np.isnan(lf[t + 1:t + 1 + LEADS]).any(): continue
            Xs.append(v[t - WIN + 1:t + 1]); Ys.append(lf[t + 1:t + 1 + LEADS] - lf[t])
            iss.append(d.date.iloc[t].strftime("%Y-%m-%d"))
            dt = d.date.iloc[t + 1]
            sps.append("train" if d.date.iloc[t] <= TRAIN_END else ("test" if dt > TEST_START else "val"))
        X6 = np.array(Xs); Y = np.array(Ys); iss = np.array(iss); sps = np.array(sps)
        tr = sps == "train"; va = sps == "val"; te = sps == "test"
        # soil/snow raw values at issue date; missing soil -> per-site train median, missing swe -> 0
        raw = np.array([soil_triple(site, i, lut) for i in iss], float)
        sms, stmp, swe = raw[:, 0], raw[:, 1], raw[:, 2]
        med_sms = float(np.nanmedian(sms[tr])) if tr.sum() else 50.0
        med_stmp = float(np.nanmedian(stmp[tr])) if tr.sum() else 60.0
        sms = np.where(np.isnan(sms), med_sms, sms)
        stmp = np.where(np.isnan(stmp), med_stmp, stmp)
        X = np.zeros((len(X6), WIN, 9)); X[:, :, :6] = X6
        X[:, :, 6] = sms[:, None]; X[:, :, 7] = stmp[:, None]; X[:, :, 8] = swe[:, None]
        F = features(X)
        assert F.shape[1] == 11, F.shape
        Z = (F - F[tr].mean(0)) / (F[tr].std(0) + 1e-9)
        mt, st = Y[tr].mean(0), Y[tr].std(0) + 1e-9
        Ts = (Y - mt) / st
        refined[site] = {}
        sres = {"n": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())}}
        for kind in ["matern", "rbf"]:
            cur_ls = G[site][kind]["setting"]; cur_nz = G[site][kind]["noise"]
            kfn = matern if kind == "matern" else rbf
            # finer grid: around current best, extended at boundaries
            ls_cands = sorted(set([cur_ls * f for f in (0.5, 0.7, 1.0, 1.4, 2.0)]))
            if cur_ls >= 32: ls_cands += [48, 64]
            if cur_ls <= 0.5: ls_cands = [0.25, 0.35] + ls_cands
            nz_cands = sorted(set([cur_nz * f for f in (0.3, 1.0, 3.0)]))
            best = (np.inf, cur_ls, cur_nz)
            for ls in ls_cands:
                K = kfn(Z, Z, ls)
                Ktr = K[np.ix_(tr, tr)]; Kva = K[np.ix_(va, tr)]
                for nz in nz_cands:
                    try:
                        mu = gp_pred(Ktr, Kva, Ts[tr], nz)
                    except Exception:
                        continue
                    # val RMSE in standardized units, mean over leads
                    rmse = float(np.sqrt(((mu - Ts[va]) ** 2).mean()))
                    if rmse < best[0]:
                        best = (rmse, ls, nz)
            print(f"  {kind}: ls {cur_ls}->{best[1]:.2f}, noise {cur_nz}->{best[2]:.4f}, val {best[0]:.4f}", flush=True)
            refined[site][kind] = {"setting": float(best[1]), "noise": float(best[2])}
            # test RMSE (log-flow units, gp10 protocol: fit on train, score test, mean over 7 leads)
            Ktr = kfn(Z[tr], Z[tr], best[1]); Kte = kfn(Z[te], Z[tr], best[1])
            mup = gp_pred(Ktr, Kte, Ts[tr], best[2])
            err = (mup - Ts[te]) * st
            test_rmse = float(np.sqrt((err ** 2).mean()))
            test_lead = np.sqrt((err ** 2).mean(0)).round(4).tolist()
            sres[kind] = {"setting": float(best[1]), "noise": float(best[2]),
                          "val": float(best[0]), "test": test_rmse, "test_lead": test_lead}
            print(f"  {kind} test log-RMSE {test_rmse:.4f}", flush=True)
        results["stations"][site] = sres
    json.dump(refined, open(RES / "gp_refined9.json", "w"), indent=1)
    json.dump(results, open(RES / "gp9_results.json", "w"), indent=1)
    print("saved gp_refined9.json, gp9_results.json", flush=True)
    for kind in ["matern", "rbf"]:
        m = np.mean([results["stations"][s][kind]["test"] for s in results["stations"]])
        print(f"mean test RMSE {kind}: {m:.4f}", flush=True)
    # refresh the export hyperparameters (classical_hyper.json) with the retrained settings
    hyp_path = Path("/home/hatch/workspace/hydrotwin_colab/classical_hyper.json")
    hyp = json.loads(hyp_path.read_text())
    for site in refined:
        hyp[site]["GP_RBF"] = {"ls": refined[site]["rbf"]["setting"], "noise": refined[site]["rbf"]["noise"]}
        hyp[site]["GP_Matern"] = {"ls": refined[site]["matern"]["setting"], "noise": refined[site]["matern"]["noise"]}
    hyp_path.write_text(json.dumps(hyp, indent=1))
    print("classical_hyper.json GP entries updated", flush=True)

if __name__ == "__main__":
    main()
