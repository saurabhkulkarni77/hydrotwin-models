#!/usr/bin/env python3
"""Refresh NRCS soil/snow data for the 9-feature pipeline.

Fetches SCAN soil moisture/temp and SNOTEL SWE from the USDA AWDB API for
each site's mapped stations, and appends to data/nrcs_daily.csv
(site, date, sms_pct, stmp_f, swe_in). Only stdlib + pandas.

The training pipeline (prepare9) merges this CSV over the baseline Excel,
so weekly retrains get fresh soil/snow.

Usage: python refresh_nrcs.py
"""
import json
import os
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

PROJ = Path(os.environ.get("HYDROTWIN_PROJ", Path(__file__).resolve().parent))
OUT = PROJ / "data" / "nrcs_daily.csv"

SOILST = {
    '01463500': '2039:VA:SCAN', '01646500': '2039:VA:SCAN', '02320500': '2012:FL:SCAN',
    '03294500': '2075:TN:SCAN', '05331000': '2002:MN:SCAN', '06214500': '2119:MT:SCAN',
    '06934500': '2220:MO:SCAN', '07374000': '2086:MS:SCAN', '08057000': '2022:OK:SCAN',
    '08158000': '2200:TX:SCAN', '08313000': '708:NM:SNTL', '09380000': '2162:UT:SCAN',
    '11447650': '463:CA:SNTL', '14105700': '679:WA:SNTL',
}
SNOTEL = {
    '06214500': ['981:MT:SNTL', '326:WY:SNTL', '1105:MT:SNTL'],
    '06934500': ['920:SD:SNTL', '354:SD:SNTL', '1045:WY:SNTL'],
    '07374000': ['934:NM:SNTL', '857:CO:SNTL', '303:CO:SNTL'],
    '08313000': ['1254:NM:SNTL', '1083:NM:SNTL', '491:NM:SNTL'],
    '09380000': ['1249:UT:SNTL', '1269:UT:SNTL', '452:UT:SNTL'],
    '11447650': ['1067:CA:SNTL', '428:CA:SNTL', '301:CA:SNTL'],
    '14105700': ['401:OR:SNTL', '599:WA:SNTL', '502:WA:SNTL'],
}


def http_json(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "HydroTwin-refresh/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        print(f"  fetch failed: {e}")
        return None


def fetch_awdb(triplets, elements, duration, d1, d2):
    """{triplet: {date: value}} for the given elements (last value wins per day)."""
    url = ("https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data"
           f"?stationTriplets={','.join(triplets)}&elements={elements}"
           f"&duration={duration}&beginDate={d1}&endDate={d2}&unitSystem=ENGLISH")
    data = http_json(url)
    out = {}
    if not isinstance(data, list):
        return out
    for s in data:
        trip = (s.get("stationTriplet") or "")
        for dd in (s.get("data") or []):
            code = (dd.get("stationElement") or {}).get("elementCode")
            for v in (dd.get("values") or []):
                if v.get("value") is None:
                    continue
                dt = str(v.get("date", ""))[:10]
                out.setdefault(trip, {}).setdefault(dt, {})[code] = v["value"]
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # last date we have
    last = {}
    if OUT.exists():
        df = pd.read_csv(OUT, dtype={"site": str})
        for site, g in df.groupby("site"):
            last[site] = g["date"].max()
    today = date.today()
    d1 = (today - timedelta(days=14)).isoformat()  # 2-week overlap for safety
    d2 = today.isoformat()
    rows = []
    for site in sorted(SOILST):
        if last.get(site, "") >= d2:
            continue
        print(f"{site}...", flush=True)
        # soil from SCAN/SNTL station
        soil = fetch_awdb([SOILST[site]], "SMS:-8,SMS:-2,STO:-2", "DAILY", d1, d2)
        # SWE from SNOTEL triplets (basin mean)
        swe = {}
        trips = SNOTEL.get(site, [])
        if trips:
            swe_data = fetch_awdb(trips, "WTEQ", "DAILY", d1, d2)
            # average across stations per date
            by_date = {}
            for trip, dates in swe_data.items():
                for dt, vals in dates.items():
                    if "WTEQ" in vals:
                        by_date.setdefault(dt, []).append(vals["WTEQ"])
            swe = {dt: sum(v) / len(v) for dt, v in by_date.items()}
        # combine per date
        dates = set()
        for trip, dd in soil.get(SOILST[site], {}).items():
            dates.update(dd.keys()) if isinstance(dd, dict) else None
        # soil structure is {trip: {date: {code: val}}}
        sdata = soil.get(SOILST[site], {})
        dates = set(sdata.keys()) | set(swe.keys())
        for dt in sorted(dates):
            if dt <= last.get(site, ""):
                continue
            vals = sdata.get(dt, {})
            sms = vals.get("SMS:-8", vals.get("SMS:-2"))
            stmp = vals.get("STO:-2")
            rows.append({"site": site, "date": dt, "sms_pct": sms,
                         "stmp_f": stmp, "swe_in": swe.get(dt, 0.0)})
    if rows:
        df = pd.DataFrame(rows)
        if OUT.exists():
            old = pd.read_csv(OUT, dtype={"site": str})
            df = pd.concat([old, df]).drop_duplicates(["site", "date"], keep="last")
        df.to_csv(OUT, index=False)
        print(f"wrote {len(rows)} new rows to {OUT}")
    else:
        print("no new NRCS data")


if __name__ == "__main__":
    main()
