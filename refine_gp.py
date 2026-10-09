"""Refine GP hyperparameters: finer lengthscale grid around current best + noise refinement.
For each (river, kernel): try ls in {0.5, 0.7, 1.0, 1.4, 2.0} x current best (extended if at boundary),
and noise in {0.3, 1.0, 3.0} x current noise. Pick by validation RMSE.
Saves refined settings + new predictions."""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve

sys.path.insert(0, str(Path(os.environ.get("HYDROTWIN_PROJ", Path(__file__).resolve().parent)) / "ml"))
# patch the /home/claude import inside gp10
import importlib.util
spec = importlib.util.spec_from_file_location("gp10", str(Path(os.environ.get("HYDROTWIN_PROJ", Path(__file__).resolve().parent)) / "ml" / "gp10.py"))
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
    lf = np.log(X[:, :, 4]); rain = X[:, :, 3]; tm = X[:, :, 2]
    return np.stack([lf[:, -1], lf[:, -1] - lf[:, -2], lf[:, -1] - lf[:, -8], rain[:, -1], rain[:, -3:].sum(1),
                     rain[:, -7:].sum(1), rain.sum(1), tm[:, -7:].mean(1)], 1)

PROJ = Path(os.environ.get("HYDROTWIN_PROJ", Path(__file__).resolve().parent))
RES = PROJ / "ml" / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lstm_normal import prepare

def main():
    P = prepare()
    G = json.load(open(RES / "gp10_results.json"))["stations"]
    refined = {}
    for si, site in enumerate(P["sites"]):
        print(f"=== {site} ===", flush=True)
        # rebuild features like gp_valtest
        import pandas as pd
        d = pd.read_csv(PROJ / "data" / "daily" / f"{site}_daily.csv", parse_dates=["date"])
        cols = ["tmax_c", "tmin_c", "tmean_c", "precip_mm", "flow_cfs", "rising"]
        v = d[cols].to_numpy(float)
        ok = ~np.isnan(v).any(1)
        WIN, LEADS = 29, 7
        Xs, Ys, iss, sps = [], [], [], []
        TRAIN_END = pd.Timestamp("2023-09-30"); TEST_START = pd.Timestamp("2024-09-30")
        for t in range(WIN - 1, len(d) - LEADS):
            if not ok[t - WIN + 1:t + 1].all(): continue
            lf = np.log(d.flow_cfs.to_numpy(float))
            if np.isnan(lf[t + 1:t + 1 + LEADS]).any(): continue
            Xs.append(v[t - WIN + 1:t + 1]); Ys.append(lf[t + 1:t + 1 + LEADS] - lf[t])
            iss.append(d.date.iloc[t].strftime("%Y-%m-%d"))
            dt = d.date.iloc[t + 1]
            sps.append("train" if d.date.iloc[t] <= TRAIN_END else ("test" if dt > TEST_START else "val"))
        X = np.array(Xs); Y = np.array(Ys); sps = np.array(sps)
        F = features(X)
        tr = sps == "train"; va = sps == "val"; te = sps == "test"
        Z = (F - F[tr].mean(0)) / (F[tr].std(0) + 1e-9)
        mt, st = Y[tr].mean(0), Y[tr].std(0) + 1e-9
        Ts = (Y - mt) / st
        refined[site] = {}
        for kind in ["matern", "rbf"]:
            cur_ls = G[site][kind]["setting"]; cur_nz = G[site][kind]["noise"]
            kfn = matern if kind == "matern" else rbf
            # finer grid: around current best, extended at boundaries
            ls_cands = sorted(set([cur_ls * f for f in (0.5, 0.7, 1.0, 1.4, 2.0)]))
            if cur_ls >= 32: ls_cands += [48, 64]
            if cur_ls <= 0.5: ls_cands = [0.25, 0.35] + ls_cands
            nz_cands = sorted(set([cur_nz * f for f in (0.3, 1.0, 3.0)]))
            best = (np.inf, cur_ls, cur_nz)
            Ktr_cache = {}
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
    json.dump(refined, open(RES / "gp_refined.json", "w"), indent=1)
    print("saved gp_refined.json", flush=True)

if __name__ == "__main__":
    main()
