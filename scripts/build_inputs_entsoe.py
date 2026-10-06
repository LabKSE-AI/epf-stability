"""Build data/inputs/<MARKET>.parquet for the additional markets: DE (EPEX SPOT; bidding zone
DE-AT-LU until 2018-09-30, DE-LU from 2018-10-01) and SE4 (Nord Pool, Sweden bidding zone 4).

  price    SDAC day-ahead price, EUR/MWh (ENTSO-E A44), hourly. Taken from the hourly price table
           produced by the project's data pipeline (ENTSO-E A44; the DE document contains, since
           1.10.2025, also 15-minute products and a second auction — only the hourly SDAC price
           series is used; 15-minute prices are averaged to hours only when all four quarter-hours
           exist).
  load_fc  day-ahead total-load forecast of the bidding zone (ENTSO-E A65/A01), downloaded with
           scripts/fetch_entsoe.py; sub-hourly values averaged to hours only when complete.
Missing values stay missing; nothing is interpolated or imputed. Local time: CET/CEST.

Usage: python scripts/build_inputs_entsoe.py --prices <neighbours_hourly_utc.parquet>
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epfstab.paths import INPUTS, RAW  # noqa: E402

RES_MIN = {"PT15M": 15, "PT30M": 30, "PT60M": 60}
START_UTC = pd.Timestamp("2015-12-31 23:00", tz="UTC")
END_UTC = pd.Timestamp("2025-12-31 22:00", tz="UTC")
DE_SPLIT_UTC = pd.Timestamp("2018-09-30 22:00", tz="UTC")     # DE-LU zone starts 2018-10-01 00:00 CET
MARKETS = {"DE": {"tz": "Europe/Berlin", "price_col": "price_DE",
                  "loadfc": [("DE_AT_LU_loadfc", None, DE_SPLIT_UTC), ("DE_loadfc", DE_SPLIT_UTC, None)]},
           "SE4": {"tz": "Europe/Stockholm", "price_col": "price_SE4", "loadfc": [("SE4_loadfc", None, None)]}}


def parse_points(xml: bytes) -> pd.DataFrame:
    """All points of all TimeSeries: (time_utc, resolution_min, value). curveType A03 omits
    points whose value equals the previous one — they are restored by the format definition."""
    root = etree.fromstring(xml)
    ns = {"n": root.nsmap[None]} if None in root.nsmap else {}
    q = (lambda p: p.replace("/", "/n:").replace("//n:", "//n:")) if ns else (lambda p: p)
    rows = []
    for ts in root.iterfind(".//n:TimeSeries" if ns else ".//TimeSeries", ns):
        curve = ts.findtext("n:curveType", default="A01", namespaces=ns)
        for per in ts.iterfind("n:Period" if ns else "Period", ns):
            start = pd.Timestamp(per.findtext("n:timeInterval/n:start", namespaces=ns))
            end = pd.Timestamp(per.findtext("n:timeInterval/n:end", namespaces=ns))
            res = RES_MIN[per.findtext("n:resolution", namespaces=ns)]
            n = int((end - start) / pd.Timedelta(minutes=res))
            vals = np.full(n, np.nan)
            for pt in per.iterfind("n:Point", ns):
                vals[int(pt.findtext("n:position", namespaces=ns)) - 1] = float(pt.findtext("n:quantity", namespaces=ns))
            if curve == "A03":
                vals = pd.Series(vals).ffill().to_numpy()       # format definition, not interpolation
            t = start + pd.to_timedelta(np.arange(n) * res, unit="min")
            rows.append(pd.DataFrame({"time": t, "res": res, "value": vals}))
    return pd.concat(rows) if rows else pd.DataFrame(columns=["time", "res", "value"])


def hourly(series_dir: Path) -> tuple[pd.Series, dict]:
    parts, sha = [], {}
    for f in sorted(series_dir.glob("*.xml.gz")):
        b = gzip.decompress(f.read_bytes())
        sha[f.name] = hashlib.sha256(b).hexdigest()
        if b"<TimeSeries>" in b:
            parts.append(parse_points(b))
    P = pd.concat(parts).dropna(subset=["value"])
    P["time"] = pd.to_datetime(P["time"], utc=True)
    conflicts = int((P.groupby(["time", "res"])["value"].nunique() > 1).sum())
    P = P.drop_duplicates(["time", "res"], keep="last")
    out = {}
    for res, g in P.groupby("res"):
        s = g.set_index("time")["value"].sort_index()
        hr = s.index.floor("h")
        cnt = s.groupby(hr).count()
        mean = s.groupby(hr).mean()
        out[res] = mean[cnt == 60 // res]                      # complete hours only
    # prefer the finest complete resolution available for each hour
    h = None
    for res in sorted(out):
        h = out[res] if h is None else h.combine_first(out[res])
    return h, {"files": len(sha), "duplicate_points_with_different_values": conflicts, "sha256": sha}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True)
    a = ap.parse_args()
    prices = pd.read_parquet(a.prices)
    idx = pd.date_range(START_UTC - pd.Timedelta(days=7), END_UTC, freq="h", tz="UTC")
    for mkt, spec in MARKETS.items():
        loc = idx.tz_convert(spec["tz"])
        df = pd.DataFrame(index=idx)
        df.index.name = "time"
        df["price"] = prices[spec["price_col"]].reindex(idx).astype(float)
        fl = pd.Series(np.nan, index=idx)
        prov = {}
        for name, t0, t1 in spec["loadfc"]:
            s, info = hourly(RAW / "entsoe" / name)
            m = np.ones(len(idx), bool)
            if t0 is not None:
                m &= idx >= t0
            if t1 is not None:
                m &= idx < t1
            fl[m] = s.reindex(idx[m]).to_numpy()
            prov[name] = {k: v for k, v in info.items() if k != "sha256"}
            prov[name]["sha256_sum_of_files"] = hashlib.sha256("".join(sorted(info["sha256"].values())).encode()).hexdigest()
        df["load_fc"] = fl
        df["origin"] = loc.hour == 0
        df["dow"] = loc.dayofweek
        df["lday"] = (loc.normalize().tz_localize(None) - pd.Timestamp("1970-01-01")).days
        df.to_parquet(INPUTS / f"{mkt}.parquet")
        span = df.index >= START_UTC
        meta = {"market": mkt, "timezone": spec["tz"], "price": "SDAC day-ahead price (ENTSO-E A44), EUR/MWh",
                "load_fc": "day-ahead total load forecast (ENTSO-E A65/A01), MW",
                "range_utc": [str(idx.min()), str(idx.max())],
                "missing_in_2016_2025": {c: int(df.loc[span, c].isna().sum()) for c in ["price", "load_fc"]},
                "load_fc_provenance": prov, "built_by": "scripts/build_inputs_entsoe.py"}
        (INPUTS / f"{mkt}.json").write_text(json.dumps(meta, indent=1))
        print(mkt, meta["missing_in_2016_2025"], {k: v["duplicate_points_with_different_values"] for k, v in prov.items()})


if __name__ == "__main__":
    main()
