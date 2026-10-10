#!/usr/bin/env python3
"""Export ARIMA/SARIMA/GP parameters as JSON for browser-side live forecasting.

Per station, writes <OUT>/<site>/{ARIMA,SARIMA,GP_RBF,GP_Matern}.json:
- ARIMA/SARIMA: order, seasonal_order, const, ar/ma/sar/sma coefficients.
  JS: difference series -> ARMA residual recursion -> 7-step forecast -> undifference.
- GP: kernel, ls, noise, feature mu/sd (11), inducing points Z (<=1000x11),
  alpha weights (<=1000x7), tsd (7).
  JS: 11 window features -> standardize -> k(z,Z) @ alpha -> *tsd + last -> exp.
  The 11 features = 8 original USGS summary features + soil_sms_8in_pct,
  soil_temp_2in_F, swe_in at the issue date (broadcast across the window).

Uses FIXED hyperparameters from classical_hyper.json (no grid search).
ARIMA/SARIMA fit on train+val log-flow; GP fit on train+val windows (original protocol).
"""
import json, os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
PROJ = Path(os.environ.get("HYDROTWIN_PROJ", "/home/hatch/workspace/user/files/HydroTwin_10yr_Map"))
DAILY = PROJ / "data" / "daily"
OUT = Path(os.environ.get("HYDROTWIN_PARAMS_OUT", "/home/hatch/workspace/hydrotwin_colab/params_out"))
HYPER = Path(os.environ.get("HYDROTWIN_HYPER", HERE / "classical_hyper.json"))

TRAIN_END = pd.Timestamp(os.environ.get("HYDROTWIN_TRAIN_END", "2023-09-30"))
TEST_START = pd.Timestamp(os.environ.get("HYDROTWIN_TEST_START", "2024-09-30"))
N_IND = 1000
WIN, LEADS = 29, 7


def features(X):
    lf = np.log(X[:, :, 4]); rain = X[:, :, 3]; tm = X[:, :, 2]
    base = np.stack([lf[:, -1], lf[:, -1] - lf[:, -2], lf[:, -1] - lf[:, -8], rain[:, -1],
                     rain[:, -3:].sum(1), rain[:, -7:].sum(1), rain.sum(1),
                     tm[:, -7:].mean(1)], 1)
    if X.shape[2] >= 9:
        return np.concatenate([base, X[:, -1, 6:9]], 1)
    return base


def matern(A, B, ls):
    r = np.sqrt(np.maximum(((A[:, None] - B[None]) ** 2).sum(-1), 0)) / ls
    return (1 + np.sqrt(5) * r + 5 / 3 * r ** 2) * np.exp(-np.sqrt(5) * r)


def rbf(A, B, ls):
    return np.exp(-((A[:, None] - B[None]) ** 2).sum(-1) / (2 * ls ** 2))


def build_windows(site):
    d = pd.read_csv(DAILY / f"{site}_daily.csv", parse_dates=["date"])
    cols = ["tmax_c", "tmin_c", "tmean_c", "precip_mm", "flow_cfs", "rising"]
    v = d[cols].to_numpy(float)
    lf = np.log(d.flow_cfs.to_numpy(float))
    ok = ~np.isnan(v).any(1)
    # soil/snow LUT: (site, issue date) -> raw (sms, stmp, swe); missing soil -> train median, swe -> 0
    sys.path.insert(0, str(HERE))
    from train_lstm_normal import load_soilsnow_lut
    lut = load_soilsnow_lut()
    Xs, Ys, issues = [], [], []
    for t in range(WIN - 1, len(d) - LEADS):
        if not ok[t - WIN + 1:t + 1].all():
            continue
        if np.isnan(lf[t + 1:t + 1 + LEADS]).any():
            continue
        Xs.append(v[t - WIN + 1:t + 1])
        Ys.append(lf[t + 1:t + 1 + LEADS])
        issues.append(d.date.iloc[t].strftime("%Y-%m-%d"))
    X6 = np.array(Xs); Y = np.array(Ys)
    ft = pd.to_datetime(issues) + pd.Timedelta(days=1)
    split = np.where(ft <= TRAIN_END, "train", np.where(ft <= TEST_START, "val", "test"))
    tr = split == "train"
    raw = np.array([lut.get((site, i), (np.nan, np.nan, 0.0)) for i in issues], float)
    med_sms = float(np.nanmedian(raw[tr, 0])) if tr.sum() else 50.0
    med_stmp = float(np.nanmedian(raw[tr, 1])) if tr.sum() else 60.0
    sms = np.where(np.isnan(raw[:, 0]), med_sms, raw[:, 0])
    stmp = np.where(np.isnan(raw[:, 1]), med_stmp, raw[:, 1])
    X = np.zeros((len(X6), WIN, 9)); X[:, :, :6] = X6
    X[:, :, 6] = sms[:, None]; X[:, :, 7] = stmp[:, None]; X[:, :, 8] = raw[:, 2][:, None]
    return X, Y, split


def export_arima(site, order, seasonal_order):
    from statsmodels.tsa.arima.model import ARIMA
    d = pd.read_csv(DAILY / f"{site}_daily.csv", parse_dates=["date"])
    lf = np.log(d.flow_cfs.to_numpy(float))
    s = pd.Series(lf).ffill().bfill().to_numpy()
    y = s[(d.date <= TEST_START).to_numpy()]
    kw = {} if not seasonal_order else {"seasonal_order": tuple(seasonal_order)}
    fit = ARIMA(y, order=tuple(order), **kw).fit()
    names = list(fit.param_names)
    vals = dict(zip(names, map(float, fit.params)))
    return {
        "order": [int(x) for x in order],
        "seasonal_order": [int(x) for x in seasonal_order] if seasonal_order else None,
        "const": float(vals.get("const", 0.0)),
        "ar": [vals[k] for k in names if k.startswith("ar.L")],
        "ma": [vals[k] for k in names if k.startswith("ma.L")],
        "sar": [vals[k] for k in names if k.startswith("ar.S")],
        "sma": [vals[k] for k in names if k.startswith("ma.S")],
    }


def export_gp(site, kernel, ls, noise):
    from scipy.linalg import cho_factor, cho_solve
    kfn = matern if kernel == "matern" else rbf
    X, Y, split = build_windows(site)
    F = features(X)
    last = np.log(X[:, -1, 4])
    Traw = Y - last[:, None]
    tr = split == "train"
    trv = tr | (split == "val")
    if tr.sum() < 200 or trv.sum() < 200:
        return None
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
    Z = (F - mu) / sd
    tsd = Traw[tr].std(0) + 1e-9
    T = Traw / tsd
    rng = np.random.default_rng(0)
    ind = rng.choice(np.where(trv)[0], size=min(N_IND, trv.sum()), replace=False)
    Ztr, Ttr = Z[ind], T[ind]
    K = kfn(Ztr, Ztr, ls)
    alpha = cho_solve(cho_factor(K + noise * np.eye(len(K))), Ttr)
    return {
        "kernel": kernel,
        "ls": float(ls),
        "noise": float(noise),
        "fmu": [round(float(x), 6) for x in mu],
        "fsd": [round(float(x), 6) for x in sd],
        "tsd": [round(float(x), 6) for x in tsd],
        "Z": [[round(float(x), 4) for x in row] for row in Ztr],
        "alpha": [[round(float(x), 6) for x in row] for row in alpha],
    }


def main():
    hyper = json.loads(HYPER.read_text())
    sites = sorted(hyper.keys())
    for site in sites:
        h = hyper[site]
        sdir = OUT / site
        sdir.mkdir(parents=True, exist_ok=True)
        # ARIMA / SARIMA
        for name, key in (("ARIMA", "ARIMA"), ("SARIMA", "SARIMA")):
            hh = h[key]
            p = export_arima(site, hh["order"], hh["seasonal_order"])
            (sdir / f"{name}.json").write_text(json.dumps(p))
            print(f"{site} {name}: order={p['order']} seas={p['seasonal_order']}", flush=True)
        # GPs
        for name, key, kernel in (("GP_RBF", "GP_RBF", "rbf"), ("GP_Matern", "GP_Matern", "matern")):
            hh = h[key]
            p = export_gp(site, kernel, hh["ls"], hh["noise"])
            if p is None:
                print(f"{site} {name}: skipped (too little data)", flush=True)
                continue
            (sdir / f"{name}.json").write_text(json.dumps(p))
            print(f"{site} {name}: ls={p['ls']} noise={p['noise']} ind={len(p['Z'])}", flush=True)
    print("PARAM EXPORT DONE:", OUT)


if __name__ == "__main__":
    main()
