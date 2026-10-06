"""Download day-ahead prices and day-ahead load forecasts from the ENTSO-E Transparency Platform.

Access: a free ENTSO-E account and a personal security token are required
(https://transparency.entsoe.eu -> "My Account Settings" -> "Web API Security Token"; see the
platform's "Restful API" guide). The token is read from the environment variable ENTSOE_TOKEN or
from the file ~/.entsoe_token. It is never written to disk by this script.

Documents (ENTSO-E codes):
  price    A44            day-ahead price (EUR/MWh) of a bidding zone
  loadfc   A65 / A01      day-ahead total load forecast of a bidding zone (MW)

Raw XML responses are stored verbatim (gzip) with SHA-256 checksums under
data/raw/entsoe/<series>/<YYYYMM>.xml.gz, so every number can be traced back to the source.
Requests are made per calendar month; existing files are skipped (resumable).

Usage:
  python scripts/fetch_entsoe.py --series DE_price DE_AT_LU_price DE_loadfc DE_AT_LU_loadfc SE4_price SE4_loadfc \
      --start 2015-12 --end 2025-12
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("EPFSTAB_DATA", ROOT / "data")) / "raw" / "entsoe"
API = "https://web-api.tp.entsoe.eu/api"

ZONES = {  # EIC codes of bidding zones
    "DE_LU": "10Y1001A1001A82H",      # Germany-Luxembourg, from 2018-10-01
    "DE_AT_LU": "10Y1001A1001A63L",   # Germany-Austria-Luxembourg, until 2018-09-30
    "SE4": "10Y1001A1001A47J",        # Sweden, bidding zone SE4 (Nord Pool)
    "PL": "10YPL-AREA-----S",
}
SERIES = {
    "DE_price": {"documentType": "A44", "in_Domain": ZONES["DE_LU"], "out_Domain": ZONES["DE_LU"]},
    "DE_AT_LU_price": {"documentType": "A44", "in_Domain": ZONES["DE_AT_LU"], "out_Domain": ZONES["DE_AT_LU"]},
    "SE4_price": {"documentType": "A44", "in_Domain": ZONES["SE4"], "out_Domain": ZONES["SE4"]},
    "DE_loadfc": {"documentType": "A65", "processType": "A01", "outBiddingZone_Domain": ZONES["DE_LU"]},
    "DE_AT_LU_loadfc": {"documentType": "A65", "processType": "A01", "outBiddingZone_Domain": ZONES["DE_AT_LU"]},
    "SE4_loadfc": {"documentType": "A65", "processType": "A01", "outBiddingZone_Domain": ZONES["SE4"]},
    "PL_loadfc": {"documentType": "A65", "processType": "A01", "outBiddingZone_Domain": ZONES["PL"]},
}


def token() -> str:
    t = os.environ.get("ENTSOE_TOKEN", "")
    p = Path.home() / ".entsoe_token"
    if not t and p.exists():
        t = p.read_text().strip()
    if not t:
        raise SystemExit("ENTSO-E token missing: set ENTSOE_TOKEN or create ~/.entsoe_token")
    return t


def fetch_month(params: dict, month: pd.Period) -> bytes:
    start = month.start_time.tz_localize("UTC") - pd.Timedelta(hours=2)   # cover local midnight
    end = (month + 1).start_time.tz_localize("UTC")
    q = dict(params, securityToken=token(), periodStart=start.strftime("%Y%m%d%H%M"),
             periodEnd=end.strftime("%Y%m%d%H%M"))
    for attempt in range(6):
        try:
            r = requests.get(API, params=q, timeout=180)
            if r.status_code == 429 or r.status_code >= 500:
                raise RuntimeError(f"HTTP {r.status_code}")
            body = r.content
            if body[:2] == b"PK":                    # zipped multi-document answer
                z = zipfile.ZipFile(io.BytesIO(body))
                body = b"\n".join(z.read(n) for n in sorted(z.namelist()))
            return body
        except Exception as e:                       # noqa: BLE001 — network: retry with backoff
            wait = 20 * (attempt + 1)
            print(f"    retry {attempt + 1} after error {e!r} (sleep {wait}s)", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed after retries: {params} {month}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="+", default=list(SERIES))
    ap.add_argument("--start", default="2015-12")
    ap.add_argument("--end", default="2025-12")
    a = ap.parse_args()
    for s in a.series:
        d = RAW / s
        d.mkdir(parents=True, exist_ok=True)
        meta_f = d / "_sha256.json"
        meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
        n_new = 0
        for m in pd.period_range(a.start, a.end, freq="M"):
            f = d / f"{m.strftime('%Y%m')}.xml.gz"
            if f.exists():
                continue
            body = fetch_month(SERIES[s], m)
            f.write_bytes(gzip.compress(body))
            meta[f.name] = {"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body),
                            "params": {k: v for k, v in SERIES[s].items()},
                            "fetched_utc": pd.Timestamp.now("UTC").isoformat()}
            n_new += 1
            time.sleep(0.5)
        meta_f.write_text(json.dumps(meta, indent=1))
        print(f"{s}: {n_new} new monthly files", flush=True)


if __name__ == "__main__":
    main()
