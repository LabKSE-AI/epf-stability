"""Build data/inputs/PL.parquet — Polish day-ahead market (TGE Fixing I), hourly UTC grid.

Sources (see README, "Data"):
  * hourly master file of the project's data pipeline (timestamps in UTC, all sources merged by
    timestamp): TGE Fixing I price and volume (Instrat energy-data service, TGE web archive),
    generation by technology and system load (Instrat / ENTSO-E), day-ahead load forecast of the
    TSO PSE (ENTSO-E Transparency Platform, A65/A01);
  * fuel and carbon prices: EU ETS allowance (ICAP Allowance Price Explorer, daily), European gas
    and South-African coal benchmarks (World Bank "Pink Sheet", monthly), NBP exchange rates.

Information availability (no look-ahead):
  * a daily value of date d (EUA settlement, NBP mid-rate) is used from local date d+1 on and for at
    most 21 days (weekends, year-end gaps of the source); older values are not carried forward (missing instead);
  * a monthly average of month M is used from the 8th day of month M+1 (publication of the Pink
    Sheet early in the following month) until the 7th day of month M+2.
Missing values stay missing (NaN); nothing is interpolated or imputed.

Usage: python scripts/build_inputs_pl.py --master <path/master_hourly_utc.csv> --exo <dir with exogenous csv>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epfstab.paths import INPUTS  # noqa: E402

TZ = "Europe/Warsaw"
GEN = ["hard_coal", "coalderived", "lignite", "gas", "oil", "biomass", "wind_onshore", "solar",
       "hydro_pumped_storage", "hydro_runofriver_and_poundage", "hydro_water_reservoir", "other",
       "other_renewable", "electricity_load"]
MWH_PER_MMBTU = 0.29307107
END_UTC = pd.Timestamp("2025-12-31 22:00", tz="UTC")      # last hour of 2025 in Polish local time


def sha(f: Path) -> str:
    return hashlib.sha256(f.read_bytes()).hexdigest()


def daily_available(s: pd.Series, days: pd.DatetimeIndex, max_stale: int = 21) -> pd.Series:
    """Value known at the start of each local day: last observation strictly before that day,
    if it is at most `max_stale` days old."""
    s = s.dropna().sort_index()
    pos = np.searchsorted(s.index.values, days.values, side="left") - 1
    out = pd.Series(np.nan, index=days)
    ok = pos >= 0
    obs_day = pd.DatetimeIndex(s.index.values[pos[ok]])
    fresh = (days[ok] - obs_day).days <= max_stale
    vals = s.values[pos[ok]]
    out.iloc[np.flatnonzero(ok)[fresh]] = vals[fresh]
    return out


def monthly_available(s: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    """Monthly average of month M usable from day 8 of M+1 to day 7 of M+2."""
    out = pd.Series(np.nan, index=days)
    for m, v in s.dropna().items():
        start = (m + pd.offsets.MonthBegin(1)) + pd.Timedelta(days=7)
        end = (m + pd.offsets.MonthBegin(2)) + pd.Timedelta(days=6)
        out[(days >= start) & (days <= end)] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--exo", required=True)
    a = ap.parse_args()
    master, exo = Path(a.master), Path(a.exo)
    M = pd.read_csv(master, parse_dates=["start_utc"]).set_index("start_utc").sort_index()
    M = M[M.index <= END_UTC]
    idx = pd.date_range(M.index.min(), END_UTC, freq="h", tz="UTC")
    M = M.reindex(idx)
    loc = idx.tz_convert(TZ)
    df = pd.DataFrame(index=idx)
    df.index.name = "time"
    df["price"] = M["CenaI"].astype(float)
    df["load_fc"] = M["load_forecast_entsoe"].astype(float)
    df["origin"] = loc.hour == 0
    df["dow"] = loc.dayofweek
    lday = loc.normalize().tz_localize(None)
    df["lday"] = (lday - pd.Timestamp("1970-01-01")).days
    df["volume"] = M["VolumenI"].astype(float)
    for c in GEN:
        df[c] = M[c].astype(float)
    # fuels / carbon / FX known at the start of each local day
    days = pd.DatetimeIndex(sorted(set(lday)))
    co2 = pd.read_csv(exo / "co2_eua_eur.csv", parse_dates=["date"]).set_index("date")["co2_eua_eur"]
    fx = pd.read_csv(exo / "fx_nbp_daily.csv", parse_dates=["date"]).set_index("date")
    mon = pd.read_csv(exo / "exogenous_monthly.csv", parse_dates=["month"]).set_index("month")
    D = pd.DataFrame(index=days)
    D["co2_eua_eur"] = daily_available(co2, days)
    D["eurpln"] = daily_available(fx["eurpln"], days)
    D["usdpln"] = daily_available(fx["usdpln"], days)
    D["gas_eu_usd_mmbtu"] = monthly_available(mon["gas_eu_usd_mmbtu"], days)
    D["coal_za_usd_mt"] = monthly_available(mon["coal_za_usd_mt"], days)
    D["co2_eua_pln_t"] = D["co2_eua_eur"] * D["eurpln"]
    D["gas_eu_pln_mwh"] = D["gas_eu_usd_mmbtu"] * D["usdpln"] / MWH_PER_MMBTU
    D["coal_za_pln_t"] = D["coal_za_usd_mt"] * D["usdpln"]
    for c in ["co2_eua_eur", "gas_eu_usd_mmbtu", "coal_za_usd_mt", "co2_eua_pln_t", "gas_eu_pln_mwh", "coal_za_pln_t"]:
        df[c] = D[c].reindex(lday).to_numpy()
    INPUTS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(INPUTS / "PL.parquet")
    span = df.index[df.index >= pd.Timestamp("2015-12-31 23:00", tz="UTC")]
    meta = {"market": "PL", "timezone": TZ, "price": "TGE Fixing I, PLN/MWh", "load_fc": "PSE day-ahead total load forecast (ENTSO-E A65/A01), MW",
            "range_utc": [str(df.index.min()), str(df.index.max())],
            "missing_in_2016_2025": {c: int(df.loc[span, c].isna().sum()) for c in df.columns if c not in ("origin", "dow", "lday")},
            "availability_rules": {"daily": "value of date d used from d+1, at most 21 days old", "monthly": "month M used from day 8 of M+1 to day 7 of M+2"},
            "sources_sha256": {master.name: sha(master), **{f: sha(exo / f) for f in ["co2_eua_eur.csv", "fx_nbp_daily.csv", "exogenous_monthly.csv"]}},
            "built_by": "scripts/build_inputs_pl.py"}
    (INPUTS / "PL.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta["missing_in_2016_2025"], indent=0))


if __name__ == "__main__":
    main()
