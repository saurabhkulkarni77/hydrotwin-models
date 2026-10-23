"""Single-station dataset builder for tuning, replicating the repo's
prepare9() conventions (train_lstm_normal.py) exactly:

- 9 features: 6 USGS/Open-Meteo windowed columns (flow column in ln) +
  soil_sms / soil_temp / SWE from the NRCS lookup, issue-date values
  broadcast across the window (ALL data combined, per the standing rule).
- Targets: ln-flow deltas vs the last window day, 7 leads.
- Per-station normalization stats from the fold's TRAIN rows only.
- Missing soil -> train median; missing SWE -> 0.

The only deliberate differences from prepare9(): it builds ONE station
(pooled training normalizes per station anyway, so the station's tensors
are identical), and splits are assigned by fold instead of the fixed
2023/2024 cutoffs:

  fold "tune": train ft <= 2024-09-30, val = WY2025  (tuning objective)
  fold "gate": stats/train span ft <= 2025-09-30; fit-train ft <=
               2025-03-31, early-stop val = 2025-04-01..2025-09-30,
               test = WY2026 (the untouched champion/challenger holdout,
               same convention as the drift-fix A-vs-B backtest)

ft = first forecast date (issue date + 1). WY = water year (Oct-Sep).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train_lstm_normal as TL  # noqa: E402  (repo module: COLS, LUT loader)

LEADS = 7
LN10 = np.log(10.0)

_cache = {}


def _windows(d, W):
    """Parameterized copy of TL.windows() (which fixes WIN=29)."""
    v = d[TL.COLS].to_numpy(float)
    lf = np.log(d.flow_cfs.to_numpy(float))
    ok = ~np.isnan(v).any(1)
    okc = np.concatenate([[0], np.cumsum(ok)])
    X, Y, issue, ft = [], [], [], []
    for t in range(W - 1, len(d) - LEADS):
        if okc[t + 1] - okc[t - W + 1] != W or np.isnan(lf[t + 1:t + 1 + LEADS]).any():
            continue
        X.append(v[t - W + 1:t + 1])
        Y.append(lf[t + 1:t + 1 + LEADS])
        issue.append(d.date.iloc[t])
        ft.append(d.date.iloc[t + 1])
    return (np.array(X), np.array(Y), np.array(issue),
            pd.Series(pd.to_datetime(ft)))


def _gp_features(X9raw):
    """The 11 GP kernel features of refine_gp.features() /
    train_ablation.kfeatures(), verbatim: 8 summary stats of the RAW
    (unstandardized) window + the raw soil/snow issue-date values that
    ride in channels 6-8. X9raw: (N, W, 9) raw daily values with the
    soil/snow channels broadcast across the window, exactly as
    refine_gp.py builds them."""
    lf = np.log(X9raw[:, :, 4])
    rain = X9raw[:, :, 3]
    tm = X9raw[:, :, 2]
    base = np.stack([lf[:, -1], lf[:, -1] - lf[:, -2],
                     lf[:, -1] - lf[:, -8], rain[:, -1],
                     rain[:, -3:].sum(1), rain[:, -7:].sum(1),
                     rain.sum(1), tm[:, -7:].mean(1)], 1)
    return np.concatenate([base, X9raw[:, -1, 6:9]], 1)


def build_station_P(site, window, fold, proj=None):
    """Return a prepare()-shaped dict for one station and fold."""
    key = (site, window, fold, str(proj))
    if key in _cache:
        return _cache[key]
    proj = Path(proj or os.environ.get("HYDROTWIN_PROJ",
                                       Path(__file__).resolve().parent.parent))
    d = pd.read_csv(proj / "data" / "daily" / f"{site}_daily.csv",
                    parse_dates=["date"])
    X, Y, issue, ft = _windows(d, window)
    X = X.copy()
    raw6 = X.copy()  # raw daily values (ch 4 = flow_cfs) for gpF/precip
    X[:, :, 4] = np.log(X[:, :, 4])
    last = X[:, -1, 4]
    T = Y - last[:, None]

    if fold == "tune":
        stats_end = pd.Timestamp("2024-09-30")
        split = np.where(ft <= stats_end, "train",
                         np.where(ft <= pd.Timestamp("2025-09-30"), "val",
                                  "test"))
    elif fold == "gate":
        stats_end = pd.Timestamp("2025-09-30")
        split = np.where(ft <= pd.Timestamp("2025-03-31"), "train",
                         np.where(ft <= stats_end, "val",
                                  np.where(ft <= pd.Timestamp("2026-09-30"),
                                           "test", "excl")))
    else:
        raise ValueError(fold)
    tr_stats = (ft <= stats_end).to_numpy()

    mu = X[tr_stats].reshape(-1, 6).mean(0)
    sd = X[tr_stats].reshape(-1, 6).std(0) + 1e-6
    tsd = T[tr_stats].std(0) + 1e-6

    lut = TL.load_soilsnow_lut()
    n = len(X)
    sms = np.full(n, np.nan)
    stmp = np.full(n, np.nan)
    swe = np.zeros(n)
    iss = pd.Series(issue).dt.strftime("%Y-%m-%d")
    for i in range(n):
        v = lut.get((site, iss.iloc[i]))
        if v:
            sms[i], stmp[i], swe[i] = v[0], v[1], v[2]
    med_sms = float(np.nanmedian(sms[tr_stats])) if tr_stats.sum() else 50.0
    med_stmp = float(np.nanmedian(stmp[tr_stats])) if tr_stats.sum() else 60.0
    s_sms = np.where(np.isnan(sms), med_sms, sms)
    s_stmp = np.where(np.isnan(stmp), med_stmp, stmp)
    mu9 = np.array([s_sms[tr_stats].mean(), s_stmp[tr_stats].mean(),
                    swe[tr_stats].mean()])
    sd9 = np.array([s_sms[tr_stats].std(), s_stmp[tr_stats].std(),
                    swe[tr_stats].std()]) + 1e-6

    X9 = np.zeros((n, window, 9), dtype=np.float32)
    X9[:, :, :6] = ((X - mu) / sd).astype(np.float32)
    for k, arr in enumerate([s_sms, s_stmp, swe]):
        X9[:, :, 6 + k] = (((arr - mu9[k]) / sd9[k]).astype(np.float32)
                           [:, None] * np.ones((1, window), dtype=np.float32))

    # Raw-window extras for the non-standard families:
    # - gpF: the 11 GP kernel features (refine_gp.features) computed on
    #   RAW windows with RAW soil/snow broadcast, as production builds.
    # - precip: issue-date precip_mm (raw), the PILSTMv1 physics input
    #   (train_pilstm.py rebuilds exactly this map from the daily CSV).
    raw9 = np.zeros((n, window, 9))
    raw9[:, :, :6] = raw6
    for k, arr in enumerate([s_sms, s_stmp, swe]):
        raw9[:, :, 6 + k] = arr[:, None]
    gpF = _gp_features(raw9)
    pmap = dict(zip(d.date.dt.strftime("%Y-%m-%d"),
                    d.precip_mm.fillna(0).to_numpy(float)))
    precip = np.array([pmap[s] for s in iss], dtype=np.float64)

    P = dict(X=X9, T=(T / tsd).astype(np.float32), last=last,
             gpF=gpF, precip=precip,
             sid=np.zeros(n, dtype=int), issue=iss.to_numpy(), split=split,
             scale=np.repeat(tsd[None], n, 0).astype(np.float32),
             sites=np.array([site]),
             stats={site: {"mu": mu.tolist() + mu9.tolist(),
                           "sd": sd.tolist() + sd9.tolist(),
                           "tsd": tsd.tolist(),
                           "med_sms": med_sms, "med_stmp": med_stmp}},
             n_feat=9)
    _cache[key] = P
    return P


def rmse_log10(P, pred_ln, idx):
    """RMSE in log10(cfs) — the drift-backtest metric convention.

    pred_ln: predictions for ALL windows (predict_ln output); idx
    selects the rows to score.
    """
    true_ln = P["last"][idx, None] + P["T"][idx] * P["scale"][idx]
    return float(np.sqrt(((pred_ln[idx] - true_ln) ** 2).mean()) / LN10)
