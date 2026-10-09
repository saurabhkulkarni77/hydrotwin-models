"""PILSTM v2: adds soil + snow physics rules.
New terms (all soft penalties):
1. saturation_excess: wet soil (SMS>70%) + rain -> amplified runoff expected
2. snowmelt: SWE>0.5in + tmean>0C -> degree-day melt contributes to flow
3. rain_on_snow: rain + deep snowpack + warm -> amplified response
4. frozen_soil: soil temp<32F -> infiltration blocked, rain=direct runoff
5. snow_accum: tmean<0C + precip -> snow (not runoff), penalize predicted spikes
"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn

sys.path.insert(0, "/home/hatch/workspace/hydrotwin_colab")
from train_lstm_normal import prepare, rmse_log, predict_logflow
RES = Path("/home/hatch/workspace/user/files/HydroTwin_10yr_Map/ml/results")
torch.set_num_threads(2)

LAM = 0.15  # physics weight

class PILSTMv2(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(6, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 7)
    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h[:, -1])

def load_soil_snow(P):
    """Load soil/snow per window (issue date values)."""
    d = pd.read_excel("/home/hatch/workspace/user/files/HydroTwin_NRCS_export.xlsx",
                      sheet_name="Features+targets NRCS",
                      usecols=["site","date","soil_sms_8in_pct","soil_temp_2in_degF","snow1_swe_in"])
    d["site"] = d["site"].astype(str).str.zfill(8)
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    lut = {(r.site, r.date): (r.soil_sms_8in_pct, r.soil_temp_2in_degF, r.snow1_swe_in)
           for r in d.itertuples()}
    n = len(P["X"])
    sms = np.full(n, np.nan); stmp = np.full(n, np.nan); swe = np.zeros(n)
    for i in range(n):
        site = P["sites"][P["sid"][i]]
        v = lut.get((site, P["issue"][i]))
        if v:
            sms[i], stmp[i], swe[i] = v[0], v[1], (v[2] if v[2]==v[2] else 0)
    # fill NaN with median
    sms = np.where(np.isnan(sms), np.nanmedian(sms), sms)
    stmp = np.where(np.isnan(stmp), np.nanmedian(stmp), stmp)
    return sms, stmp, swe

def physics_v2(pred_lf, last_lf, prcp, tmean, sms, stmp, swe):
    """pred_lf: (B,7) log-flow forecasts. All others: (B,) issue-day values."""
    # --- original terms ---
    dry = torch.clamp(1.0 - prcp / 10.0, 0, 1).unsqueeze(1)
    rises = torch.relu(pred_lf[:, 1:] - pred_lf[:, :-1])
    recession = (rises * dry).mean()
    q = torch.exp(pred_lf); q0 = torch.exp(last_lf)
    excess = torch.relu(q.sum(1) - (7*q0 + prcp*50))
    balance = (excess / (7*q0 + 1)).mean()
    # --- 1. saturation excess: wet soil + rain -> expect rise ---
    wet = torch.clamp((sms - 70) / 30, 0, 1)          # 0 dry -> 1 saturated
    sat_drive = wet * torch.clamp(prcp / 20, 0, 2)     # strong when wet+rainy
    # penalize too-small day-1 rise when saturation says runoff should jump
    rise1 = pred_lf[:, 0] - last_lf
    sat_loss = torch.relu(sat_drive*0.3 - rise1).mean()
    # --- 2. snowmelt: degree-day ---
    melt = torch.clamp(tmean, 0, 10) * torch.clamp(swe/2, 0, 1)  # 0..10
    # during melt, flow should not crash: penalize day-7 far below day-1
    melt_loss = torch.relu((pred_lf[:, 0] - pred_lf[:, 6]) * torch.clamp(melt/5,0,1)).mean() * 0.3
    # --- 3. rain on snow: amplified ---
    ros = torch.clamp(prcp/15,0,2) * torch.clamp(swe/3,0,1) * (tmean > 0).float()
    ros_loss = torch.relu(ros*0.4 - rise1).mean()
    # --- 4. frozen soil: rain -> direct runoff ---
    frozen = (stmp < 32).float()
    fz_loss = torch.relu(frozen*torch.clamp(prcp/15,0,1)*0.3 - rise1).mean()
    # --- 5. snow accumulation: cold+precip -> NO spike ---
    cold_snow = ((tmean < 0) & (prcp > 2)).float()
    acc_loss = (torch.relu(rise1 - 0.1) * cold_snow).mean()
    return recession + balance + sat_loss + melt_loss + ros_loss + fz_loss + acc_loss

def main():
    P = prepare()
    print("loading soil/snow...", flush=True)
    sms, stmp, swe = load_soil_snow(P)
    # precip + tmean per window (from daily)
    prcp = []; tm = []
    for si, s in enumerate(P["sites"]):
        d = pd.read_csv(f"/home/hatch/workspace/user/files/HydroTwin_10yr_Map/data/daily/{s}_daily.csv", parse_dates=["date"])
        pm = dict(zip(d.date.dt.strftime("%Y-%m-%d"), d.precip_mm.fillna(0).to_numpy()))
        tmm = dict(zip(d.date.dt.strftime("%Y-%m-%d"), d.tmean_c.fillna(10).to_numpy()))
        prcp.append(np.array([pm[dt] for dt in P["issue"][P["sid"]==si]]))
        tm.append(np.array([tmm[dt] for dt in P["issue"][P["sid"]==si]]))
    prcp = np.concatenate(prcp); tm = np.concatenate(tm)
    va = np.where(P["split"]=="val")[0]; te = np.where(P["split"]=="test")[0]; tr = np.where(P["split"]=="train")[0]
    Xt = torch.tensor(P["X"]); Tt = torch.tensor(P["T"])
    last_t = torch.tensor(P["last"], dtype=torch.float32)
    scale_t = torch.tensor(P["scale"], dtype=torch.float32)
    prcp_t = torch.tensor(prcp, dtype=torch.float32); tm_t = torch.tensor(tm, dtype=torch.float32)
    sms_t = torch.tensor(sms, dtype=torch.float32); stmp_t = torch.tensor(stmp, dtype=torch.float32)
    swe_t = torch.tensor(swe, dtype=torch.float32)
    for seed in (0, 1):
        print(f"=== pilstm-v2 seed {seed} ===", flush=True)
        torch.manual_seed(seed); np.random.seed(seed)
        m = PILSTMv2()
        opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-5)
        rng = np.random.default_rng(seed)
        best, bs, bad, t0 = np.inf, None, 0, time.time()
        for ep in range(30):
            m.train(); perm = rng.permutation(tr)
            for b in range(0, len(perm), 256):
                idx = perm[b:b+256]; opt.zero_grad()
                out = m(Xt[idx])
                data_loss = ((out - Tt[idx])**2).mean()
                pred_lf = last_t[idx,None] + out*scale_t[idx]
                phys = physics_v2(pred_lf, last_t[idx], prcp_t[idx], tm_t[idx], sms_t[idx], stmp_t[idx], swe_t[idx])
                (data_loss + LAM*phys).backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            # val
            m.eval()
            with torch.no_grad():
                s2=0; n2=0
                for b in range(0, len(va), 512):
                    i2=va[b:b+512]; e=m(Xt[i2])-Tt[i2]; s2+=(e**2).sum().item(); n2+=e.numel()
            vl=s2/n2
            print(f"  epoch {ep+1} val {vl:.4f} [{time.time()-t0:.0f}s]", flush=True)
            if vl < best-1e-5: best,bs,bad=vl,{k:v.clone() for k,v in m.state_dict().items()},0
            else:
                bad+=1
                if bad>=5: break
        m.load_state_dict(bs)
        pred = predict_logflow(m, P)
        np.save(RES/f"pred_pilstm_v2_s{seed}.npy", pred.astype(np.float32))
        print(f"seed {seed} test {rmse_log(P, pred[te], te):.4f}", flush=True)

if __name__ == "__main__":
    main()
