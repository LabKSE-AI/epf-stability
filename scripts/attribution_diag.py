"""Quarterly attribution of the forecast error to fundamentals with diagnostics: number of quarters, R² and adjusted R², HAC (Newey–West) inference,
variance inflation factors, unit-root tests (ADF, KPSS) of every series, Engle–Granger cointegration
tests, and re-estimation in first differences and in log-differences.

Target: quarterly median RMSE48 (and sMAPE48) of a model's per-origin errors.
Drivers (quarterly means): gas and coal benchmarks, EU ETS allowance price, system load, share of
renewables without PV (wind, biomass, hydro — PV generation is published only from 04.2020), and the
price level. Sensitivity: renewable share including PV from 2020-Q2.

  python scripts/attribution_diag.py --market PL --model lear --master <master_hourly_utc.csv> --exo <dir>
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, coint, kpss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epfstab.paths import RESULTS  # noqa: E402

warnings.filterwarnings("ignore")
TZ = "Europe/Warsaw"


def quarterly_drivers(master: Path, exo: Path) -> pd.DataFrame:
    M = pd.read_csv(master, parse_dates=["start_utc"]).set_index("start_utc")
    M = M[(M.index >= "2015-12-31 23:00") & (M.index < "2025-12-31 23:00")]
    q = M.index.tz_convert(TZ).tz_localize(None).to_period("Q")
    res = M[["wind_onshore", "biomass", "hydro_runofriver_and_poundage", "hydro_water_reservoir"]]
    ok = res.notna().all(1) & M["Produkcja"].notna()
    share = (res[ok].sum(1) / M.loc[ok, "Produkcja"])
    pv_ok = ok & M["solar"].notna()
    share_pv = ((res[pv_ok].sum(1) + M.loc[pv_ok, "solar"]) / M.loc[pv_ok, "Produkcja"])
    D = pd.DataFrame({"price": M["CenaI"].groupby(q).mean(), "load": M["electricity_load"].groupby(q).mean(),
                      "res": share.groupby(q[ok.to_numpy()]).mean(), "res_pv": share_pv.groupby(q[pv_ok.to_numpy()]).mean()})
    mon = pd.read_csv(exo / "exogenous_monthly.csv", parse_dates=["month"]).set_index("month")
    co2 = pd.read_csv(exo / "co2_eua_eur.csv", parse_dates=["date"]).set_index("date")["co2_eua_eur"]
    D["gas"] = mon["gas_eu_usd_mmbtu"].groupby(mon.index.to_period("Q")).mean()
    D["coal"] = mon["coal_za_usd_mt"].groupby(mon.index.to_period("Q")).mean()
    D["co2"] = co2.groupby(co2.index.to_period("Q")).mean()
    return D


def ols(y, X, lags=2):
    Xz = (X - X.mean()) / X.std()
    m = sm.OLS(y, sm.add_constant(Xz)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    vif = {c: float(variance_inflation_factor(sm.add_constant(Xz).values, i + 1)) for i, c in enumerate(X.columns)}
    return {"N": int(m.nobs), "R2": float(m.rsquared), "adjR2": float(m.rsquared_adj),
            "coef_z": m.params.drop("const").round(3).to_dict(), "p_HAC": m.pvalues.drop("const").round(4).to_dict(),
            "F_p": float(m.f_pvalue), "VIF": {k: round(v, 1) for k, v in vif.items()}}


def unit_roots(s: pd.Series) -> dict:
    s = s.dropna()
    a = adfuller(s, autolag="AIC")
    k = kpss(s, regression="c", nlags="auto")
    da = adfuller(s.diff().dropna(), autolag="AIC")
    return {"ADF_level_p": round(a[1], 4), "KPSS_level_p": round(k[1], 4), "ADF_diff_p": round(da[1], 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="PL")
    ap.add_argument("--model", default="lear")
    ap.add_argument("--run", default="w730_c7")
    ap.add_argument("--master", default=None, help="PL only")
    ap.add_argument("--exo", required=True)
    a = ap.parse_args()
    E = pd.read_csv(RESULTS / a.market / a.run / f"per_origin_{a.model}.csv", parse_dates=["origin_utc"])
    q = E.origin_utc.dt.tz_convert(TZ).dt.tz_localize(None).dt.to_period("Q")
    tgt = pd.DataFrame({"rmse48": E.rmse48.groupby(q).median(), "smape48": E.smape48.groupby(q).median()})
    if a.market == "PL":
        D = quarterly_drivers(Path(a.master), Path(a.exo)).join(tgt, how="inner").dropna(subset=["rmse48", "gas", "coal", "co2", "load", "res"])
        full = ["gas", "coal", "co2", "load", "res"]
    else:      # other markets: EU-wide fuel and carbon prices only (load / RES shares not comparable before 10.2018)
        from epfstab.data import load_market
        mk = load_market(a.market)
        qq = mk.index.tz_convert(TZ).tz_localize(None).to_period("Q")
        mon = pd.read_csv(Path(a.exo) / "exogenous_monthly.csv", parse_dates=["month"]).set_index("month")
        co2 = pd.read_csv(Path(a.exo) / "co2_eua_eur.csv", parse_dates=["date"]).set_index("date")["co2_eua_eur"]
        D = pd.DataFrame({"price": pd.Series(mk.price, index=qq).groupby(level=0).mean(),
                          "gas": mon["gas_eu_usd_mmbtu"].groupby(mon.index.to_period("Q")).mean(),
                          "coal": mon["coal_za_usd_mt"].groupby(mon.index.to_period("Q")).mean(),
                          "co2": co2.groupby(co2.index.to_period("Q")).mean()}).join(tgt, how="inner").dropna()
        full = ["gas", "coal", "co2"]
    out = {"market": a.market, "quarters": [str(D.index.min()), str(D.index.max())], "N": int(len(D))}
    fund = ["gas", "coal", "co2"]
    for y in ("rmse48", "smape48"):
        r = {"levels_fundamentals": ols(D[y], D[fund]), "levels_full": ols(D[y], D[full]),
             "levels_price_only": ols(D[y], D[["price"]])}
        d1 = D[[y] + full].diff().dropna()
        r["diff_fundamentals"] = ols(d1[y], d1[fund])
        r["diff_full"] = ols(d1[y], d1[full])
        lg = np.log(D[[y] + [c for c in full if c != "res"]]).diff()
        if "res" in full:
            lg = lg.join(D[["res"]].diff())
        lg = lg.dropna()
        r["logdiff_full"] = ols(lg[y], lg[full])
        r["engle_granger_fundamentals_p"] = float(coint(D[y], D[fund], trend="c")[1])
        r["engle_granger_full_p"] = float(coint(D[y], D[full], trend="c")[1])
        if "res_pv" in D:
            sub = D.dropna(subset=["res_pv"])
            r["levels_full_res_with_pv_from_2020Q2"] = ols(sub[y], sub[["gas", "coal", "co2", "load", "res_pv"]])
        out[y] = r
    out["unit_roots"] = {c: unit_roots(D[c]) for c in ["rmse48", "smape48", "price"] + full}
    f = RESULTS / "tables" / f"attribution_{a.market}_{a.model}.json"
    f.write_text(json.dumps(out, indent=1, default=str))
    for y in ("rmse48", "smape48"):
        print(y, "N", out[y]["levels_full"]["N"], "| levels R² fund/full", round(out[y]["levels_fundamentals"]["R2"], 3), round(out[y]["levels_full"]["R2"], 3),
              "adj", round(out[y]["levels_full"]["adjR2"], 3), "| diff R² fund/full", round(out[y]["diff_fundamentals"]["R2"], 3), round(out[y]["diff_full"]["R2"], 3),
              "| EG p", round(out[y]["engle_granger_full_p"], 3))
    print("->", f)


if __name__ == "__main__":
    main()
