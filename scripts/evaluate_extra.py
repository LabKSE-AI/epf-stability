"""Additional analyses: benchmark dependence and negative prices.

(i)  skill against two naïve benchmarks (seasonal and similar-day) — see evaluate_main.py tables;
(ii) level-normalised absolute error: MAE48 divided by the median price of the 28 days before the
     origin (known at the origin), per model and regime;
(iii) did the naïve benchmark's own scaled error variance shift at the 2021-Q4 break? Brown–Forsythe
     (median-centred Levene) test and a HAC regression of the log scaled error on a post-break dummy;
(iv) counts of negative and near-zero prices per market and year.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epfstab import evaluate as ev  # noqa: E402
from epfstab.data import load_market  # noqa: E402
from epfstab.paths import RESULTS  # noqa: E402

OUT = RESULTS / "tables"
BREAK = pd.Timestamp("2021-10-01", tz="Europe/Warsaw")


def rolling_median_level(mk, origins):
    s = pd.Series(mk.price, index=mk.index)
    pos = mk.index.get_indexer(origins)
    return np.array([np.nanmedian(np.abs(s.iloc[max(0, o - 28 * 24 + 1):o + 1].to_numpy())) for o in pos])


def main():
    res = {}
    for market in ("PL", "DE", "SE4"):
        mk = load_market(market)
        loc = mk.index.tz_convert("Europe/Warsaw")
        span = (loc >= "2016-01-01") & (loc < "2026-01-01")
        p = pd.Series(mk.price[span], index=loc[span])
        res.setdefault("negative_prices", {})[market] = {
            "hours_negative": int((p < 0).sum()), "hours_le_zero": int((p <= 0).sum()),
            "hours_abs_lt_10": int((p.abs() < 10).sum()), "first_negative": str(p[p < 0].index.min()),
            "by_year_negative": {int(k): int(v) for k, v in (p < 0).groupby(p.index.year).sum().items()}}
        def finished(m):
            f = RESULTS / market / "w730_c7" / f"meta_{m}.json"
            return f.exists() and json.loads(f.read_text()).get("finished", False)
        models = [m for m in ("naive", "naive_sd", "lear", "gbt", "gbt_nofuel") if finished(m)]
        if "naive" not in models:
            print(market, "— no finished naive run yet, skipped")
            continue
        F = {m: ev.load_preds(market, m) for m in models}
        idx = None
        for d in F.values():
            idx = d.index if idx is None else idx.intersection(d.index)
        lvl = rolling_median_level(mk, idx)
        rg = ev.regimes(idx)
        rows = []
        for m, d in F.items():
            P, Y = ev.split(d.loc[idx])
            a = ev.mae(P, Y) / lvl
            row = {"model": m, "n": int(len(idx)), "scaled_MAE_median": float(np.median(a))}
            for r in ("pre", "crisis", "post"):
                row[f"scaled_MAE_{r}"] = float(np.median(a[rg == r]))
            rows.append(row)
        pd.DataFrame(rows).round(4).to_csv(OUT / f"scaled_errors_{market}.csv", index=False)
        # (iii) naive benchmark: scaled error variance before/after the break
        P, Y = ev.split(F["naive"].loc[idx])
        e = ev.mae(P, Y) / lvl
        post = idx.tz_convert("Europe/Warsaw") >= BREAK
        bf = stats.levene(e[~post], e[post], center="median")
        X = sm.add_constant(post.astype(float))
        hac = sm.OLS(np.log(e), X).fit(cov_type="HAC", cov_kwds={"maxlags": 8})
        res.setdefault("naive_scaled_error_shift", {})[market] = {
            "median_pre": float(np.median(e[~post])), "median_post": float(np.median(e[post])),
            "iqr_pre": float(np.subtract(*np.percentile(e[~post], [75, 25]))), "iqr_post": float(np.subtract(*np.percentile(e[post], [75, 25]))),
            "brown_forsythe_p": float(bf.pvalue), "log_scaled_error_post_dummy": float(hac.params[1]), "hac_p": float(hac.pvalues[1])}
        print(market, json.dumps(res["naive_scaled_error_shift"][market]), "| ujemne h:", res["negative_prices"][market]["hours_negative"])
    (OUT / "extra_r1_5.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
