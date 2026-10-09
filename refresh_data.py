#!/usr/bin/env python3
"""HydroTwin daily data refresh (standalone).

For each of the 14 USGS gauges, fetches new daily mean discharge (USGS NWIS)
and daily weather (Open-Meteo archive) since the last row in its CSV, and
APPENDS complete days (flow + weather both present). History is never
rewritten. Only stdlib + pandas.

Usage: python refresh_data.py
"""
import json
import os
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

PROJ = Path(os.environ.get("HYDROTWIN_PROJ", Path(__file__).resolve().parent))
DAILY = PROJ / "data" / "daily"

# gauge lat/lon (river anchor; same as the project's NRCS mapping)
LATLON = {
    '01463500': (40.2217, -74.7781),
    '01646500': (38.9498, -77.1276),
    '02320500': (29.9558, -82.9276),
    '03294500': (38.2803, -85.7991),
    '05331000': (44.9444, -93.0881),
    '06214500': (45.8001, -108.468),
    '06934500': (38.7098, -91.4385),
    '07374000': (30.4457, -91.1916),
    '08057000': (32.7749, -96.8219),
    '08158000': (30.2461, -97.6801),
    '08313000': (35.8745, -106.1424),
    '09380000': (36.8643, -111.5879),
    '11447650': (38.4557, -121.5016),
    '14105700': (45.6083, -121.1899),
}


def http_json(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "HydroTwin-refresh/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_usgs(site, d1, d2):
    """Daily mean discharge {date: cfs} from USGS NWIS."""
    url = (f"https://waterservices.usgs.gov/nwis/dv/?format=json&sites={site}"
           f"&parameterCd=00060&startDT={d1}&endDT={d2}")
    d = http_json(url)
    out = {}
    for ts in d["value"]["timeSeries"]:
        for v in ts["values"][0]["value"]:
            try:
                out[v["dateTime"][:10]] = float(v["value"])
            except (TypeError, ValueError):
                pass
    return out


def fetch_wx(lat, lon, d1, d2):
    """Daily weather {date: (tmax_c, tmin_c, tmean_c, precip_mm)} from Open-Meteo archive."""
    url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
           f"&start_date={d1}&end_date={d2}"
           f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
           f"&temperature_unit=celsius&precipitation_unit=mm&timezone=auto")
    d = http_json(url)["daily"]
    out = {}
    for i in range(len(d["time"])):
        tx, tn, pr = d["temperature_2m_max"][i], d["temperature_2m_min"][i], d["precipitation_sum"][i]
        if tx is None or tn is None:
            continue  # archive lags ~5 days; skip days with no data yet
        out[d["time"][i]] = (tx, tn, (tx + tn) / 2.0, 0.0 if pr is None else pr)
    return out


def refresh_station(site):
    """Append new complete rows to one station's CSV. Returns n_added."""
    f = DAILY / f"{site}_daily.csv"
    df = pd.read_csv(f, parse_dates=["date"])
    last = df["date"].max().date()
    d1, d2 = last + timedelta(days=1), date.today()
    if d1 > d2:
        return 0
    flows = fetch_usgs(site, d1.isoformat(), d2.isoformat())
    lat, lon = LATLON[site]
    wx = fetch_wx(lat, lon, d1.isoformat(), d2.isoformat())
    dates = sorted(set(flows) & set(wx))  # complete days only
    if not dates:
        return 0
    prev = float(df["flow_cfs"].iloc[-1])
    lines = []
    for dt in dates:
        fl = flows[dt]
        tx, tn, tm, pr = wx[dt]
        rising = 1.0 if fl > prev else 0.0
        lines.append(f"{dt},{tx:.2f},{tn:.2f},{tm:.2f},{pr:.2f},{fl:.1f},{rising:.1f}")
        prev = fl
    with open(f, "a") as fh:  # append-only
        fh.write("\n".join(lines) + "\n")
    return len(lines)


def main():
    total = 0
    for f in sorted(DAILY.glob("*_daily.csv")):
        site = f.stem.replace("_daily", "")
        try:
            n = refresh_station(site)
        except Exception as e:
            print(f"{site}: FAILED ({str(e)[:80]})", flush=True)
            n = 0
        total += n
        print(f"{site}: +{n} rows", flush=True)
    print(f"total new rows: {total}", flush=True)


if __name__ == "__main__":
    main()
