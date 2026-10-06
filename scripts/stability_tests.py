"""Stability analyses of the paper (all on stored per-origin forecasts; CPU only).

1. Decomposition of error growth — two-way and three-way (level, volatility relative to level,
   error relative to volatility = model efficiency): per origin, log RMSE48 = log level + log (RMSE48 / level), where
   level = a price level known at the origin (median |price| over the previous 28 days; robustness: mean
   |price| and the standard deviation of the price over the same window). Differences of mean logs between
   regimes decompose the growth of the geometric-mean error exactly into a price-level part and a relative
   part; 95 % stationary-bootstrap CIs (blocks of 7 origins, resampled within each regime).
2. Fluctuation test (Giacomini & Rossi, 2010) of equal local predictive accuracy over time: rolling sums of
   the loss differential over m origins, standardised by the HAC long-run s.d.; critical values simulated from
   the limiting distribution sup |W(t) - W(t - mu)| / sqrt(mu), mu = m / P.
3. Conditional predictive ability test (Giacomini & White, 2006): instruments known at the origin —
   constant, the loss differential of the origin two days earlier (its target is realised), and price
   volatility (s.d. of the last 168 hourly prices); chi-square test with HAC covariance.
4. Multiple breaks in the mean of the monthly log error (raw, level-scaled and volatility-scaled) (least squares, global minimisation as
   in Bai & Perron, 2003; trimming 15 %; number of breaks by BIC; break-date CIs by moving-block bootstrap of
   residuals).

  python scripts/stability_tests.py --market PL --pairs lear:gbt lear:ens_inv30 lear:naive
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epfstab import evaluate as ev  # noqa: E402
from epfstab.data import load_market  # noqa: E402
from epfstab.paths import RESULTS  # noqa: E402

OUT = RESULTS / "tables"
RNG = np.random.default_rng(12345)


# ----------------------------------------------------------------------------- helpers
def levels(mk, idx):
    """Price-level measures known at each origin (previous 28 days incl. the origin hour)."""
    s = pd.Series(mk.price, index=mk.index)
    pos = mk.index.get_indexer(idx)
    w = 28 * 24
    med, mean, sd = [], [], []
    for o in pos:
        x = s.iloc[max(0, o - w + 1):o + 1].to_numpy()
        x = x[np.isfinite(x)]
        med.append(np.median(np.abs(x))); mean.append(np.mean(np.abs(x))); sd.append(np.std(x))
    return {"median_abs": np.array(med), "mean_abs": np.array(mean), "sd": np.array(sd)}


def lrv(x, q=None):
    x = np.asarray(x, float) - np.mean(x)
    T = len(x)
    q = q if q is not None else int(np.floor(4 * (T / 100) ** (2 / 9)))
    g = [np.dot(x[k:], x[:T - k]) / T for k in range(q + 1)]
    return g[0] + 2 * sum((1 - k / (q + 1)) * g[k] for k in range(1, q + 1))


def stat_boot_idx(n, block=7):
    """Indices of one stationary-bootstrap resample (Politis & Romano, 1994)."""
    out = np.empty(n, int)
    i = RNG.integers(n)
    for k in range(n):
        out[k] = i
        i = RNG.integers(n) if RNG.random() < 1 / block else (i + 1) % n
    return out


# ----------------------------------------------------------------------------- 1. decomposition
def decomposition(r, lvl, rg, reps=1000):
    rows = []
    lr, ll = np.log(r), np.log(lvl)
    base = rg == "pre"
    for reg in ("crisis", "post"):
        sel = rg == reg
        def comp(ib, ir):
            d_err = lr[sel][ir].mean() - lr[base][ib].mean()
            d_lvl = ll[sel][ir].mean() - ll[base][ib].mean()
            return d_err, d_lvl, d_err - d_lvl
        e, l, s = comp(np.arange(base.sum()), np.arange(sel.sum()))
        B = np.array([comp(stat_boot_idx(base.sum()), stat_boot_idx(sel.sum())) for _ in range(reps)])
        share = B[:, 1] / B[:, 0]
        rows.append({"regime_vs_pre": reg, "error_ratio": np.exp(e), "error_ratio_ci": np.exp(np.percentile(B[:, 0], [2.5, 97.5])).tolist(),
                     "level_ratio": np.exp(l), "level_ratio_ci": np.exp(np.percentile(B[:, 1], [2.5, 97.5])).tolist(),
                     "relative_error_ratio": np.exp(s), "relative_error_ratio_ci": np.exp(np.percentile(B[:, 2], [2.5, 97.5])).tolist(),
                     "level_share_of_log_growth_%": 100 * l / e, "level_share_ci_%": (100 * np.percentile(share, [2.5, 97.5])).tolist()})
    return rows


def decomposition3(r, lvl, vol, rg, reps=1000):
    """log error = log level + log(volatility / level) + log(error / volatility): exact split of the change
    of mean logs between a regime and the pre-crisis regime; shares of the log growth with bootstrap CIs."""
    rows = []
    A = np.column_stack([np.log(lvl), np.log(vol / lvl), np.log(r / vol)])
    base = rg == "pre"
    for reg in ("crisis", "post"):
        sel = rg == reg
        def comp(ib, ir):
            return A[sel][ir].mean(0) - A[base][ib].mean(0)
        c = comp(np.arange(base.sum()), np.arange(sel.sum()))
        B = np.array([comp(stat_boot_idx(base.sum()), stat_boot_idx(sel.sum())) for _ in range(reps)])
        tot, btot = c.sum(), B.sum(1)
        rows.append({"regime_vs_pre": reg, "error_ratio": float(np.exp(tot)),
                     **{f"{n}_ratio": float(np.exp(c[i])) for i, n in enumerate(("level", "rel_volatility", "efficiency"))},
                     **{f"{n}_ratio_ci": np.exp(np.percentile(B[:, i], [2.5, 97.5])).tolist() for i, n in enumerate(("level", "rel_volatility", "efficiency"))},
                     **{f"{n}_share_%": float(100 * c[i] / tot) for i, n in enumerate(("level", "rel_volatility", "efficiency"))},
                     **{f"{n}_share_ci_%": (100 * np.percentile(B[:, i] / btot, [2.5, 97.5])).tolist() for i, n in enumerate(("level", "rel_volatility", "efficiency"))}})
    return rows


# ----------------------------------------------------------------------------- 2. fluctuation test
def gr_critical(mu, alpha=0.05, reps=20000, steps=2000):
    """Simulated critical value of sup_{t in [mu,1]} |W(t) - W(t-mu)| / sqrt(mu) (two-sided)."""
    k = int(round(mu * steps))
    mx = np.empty(reps)
    for r in range(reps):
        w = np.concatenate([[0], np.cumsum(RNG.standard_normal(steps)) / np.sqrt(steps)])
        mx[r] = np.max(np.abs(w[k:] - w[:-k])) / np.sqrt(mu)
    return float(np.quantile(mx, 1 - alpha))


def fluctuation(d, m):
    P = len(d)
    sig = np.sqrt(lrv(d))
    c = np.concatenate([[0], np.cumsum(d)])
    F = (c[m:] - c[:-m]) / (sig * np.sqrt(m))
    return F


# ----------------------------------------------------------------------------- 3. conditional predictive ability
def gw_test(d, H, tau=2):
    """GW statistic for E[d_t | h_t] = 0 with instruments H (n x q), HAC covariance (lag tau-1 + NW bandwidth)."""
    ok = np.isfinite(d) & np.isfinite(H).all(1)
    Z = H[ok] * d[ok][:, None]
    T, q = Z.shape
    zb = Z.mean(0)
    Zc = Z - zb
    L = max(tau - 1, int(np.floor(4 * (T / 100) ** (2 / 9))))
    S = Zc.T @ Zc / T
    for k in range(1, L + 1):
        G = Zc[k:].T @ Zc[:-k] / T
        S += (1 - k / (L + 1)) * (G + G.T)
    stat_ = float(T * zb @ np.linalg.solve(S, zb))
    return {"n": int(T), "q": int(q), "stat": stat_, "p": float(stats.chi2.sf(stat_, q))}


# ----------------------------------------------------------------------------- 4. multiple breaks (LS, BIC)
def ls_breaks(y, max_breaks=4, trim=0.15):
    from ruptures import Dynp
    n = len(y)
    h = max(2, int(np.ceil(trim * n)))
    best = None
    res = {}
    for k in range(0, max_breaks + 1):
        if k == 0:
            bk = [n]
        else:
            try:
                bk = Dynp(model="l2", min_size=h, jump=1).fit(y.reshape(-1, 1)).predict(n_bkps=k)
            except Exception:
                break
        segs = np.split(y, bk[:-1])
        rss = sum(((s - s.mean()) ** 2).sum() for s in segs)
        bic = n * np.log(rss / n) + (2 * k + 1) * np.log(n)        # k break dates + (k+1) means
        res[k] = {"breaks": bk[:-1], "bic": bic}
        if best is None or bic < res[best]["bic"]:
            best = k
    return best, res


def break_ci(y, k, bks, reps=500, block=6):
    """Moving-block bootstrap of residuals around the fitted segment means; CI of each break index."""
    from ruptures import Dynp
    n = len(y)
    h = max(2, int(np.ceil(0.15 * n)))
    segs = np.split(np.arange(n), bks)
    fit = np.concatenate([np.full(len(s), y[s].mean()) for s in segs])
    e = y - fit
    draws = []
    for _ in range(reps):
        starts = RNG.integers(0, n - block + 1, size=int(np.ceil(n / block)))
        eb = np.concatenate([e[s:s + block] for s in starts])[:n]
        yb = fit + eb
        draws.append(Dynp(model="l2", min_size=h, jump=1).fit(yb.reshape(-1, 1)).predict(n_bkps=k)[:-1])
    D = np.array(draws)
    return [np.percentile(D[:, j], [2.5, 97.5]).tolist() for j in range(k)]


def load(market, models):
    F = {m: ev.load_preds(market, m) for m in models if not m.startswith("ens_")}
    idx = None
    for d in F.values():
        idx = d.index if idx is None else idx.intersection(d.index)
    P = {m: ev.split(d.loc[idx])[0] for m, d in F.items()}
    Y = ev.split(F[next(iter(F))].loc[idx])[1]
    g = "gbt" if "gbt" in P else ("gbt_nofuel" if "gbt_nofuel" in P else None)
    if g and "lear" in P:
        P["ens_avg"], _ = ev.combine(P["lear"], P[g], Y, "avg")
        P["ens_inv30"], _ = ev.combine(P["lear"], P[g], Y, "inv", 30)
    return idx, P, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="PL")
    ap.add_argument("--models", nargs="+", default=["naive", "lear", "gbt"])
    ap.add_argument("--pairs", nargs="+", default=["lear:gbt", "lear:ens_inv30", "ens_avg:ens_inv30", "lear:naive"])
    ap.add_argument("--block", choices=["all", "d1"], default="all", help="all 48 hours or the next delivery day (forecast hours 24..47)")
    ap.add_argument("--m", type=int, default=365, help="fluctuation-test window [origins]")
    ap.add_argument("--tag", default="", help="suffix of the output files (e.g. _allmodels for the decomposition of every model)")
    ap.add_argument("--scaled", action="store_true", help="tests on level-scaled losses (MAE48 / median |price| of the previous 28 days)")
    a = ap.parse_args()
    mk = load_market(a.market)
    idx, P, Y = load(a.market, a.models)
    sl = slice(0, 48) if a.block == "all" else slice(23, 47)
    P = {k: v[:, sl] for k, v in P.items()}
    Y = Y[:, sl]
    rg = ev.regimes(idx)
    L = levels(mk, idx)
    out = {"market": a.market, "block": a.block, "scaled_losses": bool(a.scaled), "n_origins": int(len(idx)), "first": str(idx.min()), "last": str(idx.max())}
    # 1. decomposition
    out["decomposition"] = {}
    for m in [x for x in P if x != "ens_avg"]:
        r = ev.rmse(P[m], Y)
        out["decomposition"][m] = {lv: decomposition(r, L[lv], rg) for lv in ("median_abs", "mean_abs", "sd")}
        out.setdefault("decomposition3", {})[m] = decomposition3(r, L["median_abs"], L["sd"], rg)
    # 2./3. pairwise tests on MAE48 (d = loss(first) - loss(second); negative = first better)
    vol = pd.Series(mk.price, index=mk.index).rolling(168, min_periods=168).std().reindex(idx).to_numpy()
    out["fluctuation"], out["gw"] = {}, {}
    P_n = len(idx)
    mu = a.m / P_n
    cv = gr_critical(mu)
    paths = {}
    for pr in a.pairs:
        x, y = pr.split(":")
        if x not in P or y not in P:
            continue
        d = ev.mae(P[x], Y) - ev.mae(P[y], Y)
        if a.scaled:
            d = d / L["median_abs"]
        F = fluctuation(d, a.m)
        paths[pr] = F
        out["fluctuation"][pr] = {"m": a.m, "mu": mu, "crit_5%": cv, "max_abs": float(np.max(np.abs(F))),
                                  "reject": bool(np.max(np.abs(F)) > cv), "share_outside_%": float(100 * np.mean(np.abs(F) > cv))}
        dl = np.r_[np.nan, np.nan, d[:-2]]
        H = np.column_stack([np.ones(len(d)), dl, vol])
        out["gw"][pr] = {"instruments": "const, d(t-2), price s.d. 168 h", **gw_test(d, H)}
        for reg in ("pre", "crisis", "post"):
            s = rg == reg
            out["gw"][f"{pr}|{reg}"] = gw_test(d[s], H[s][:, [0, 1]])
    pd.DataFrame({k: pd.Series(v, index=idx[a.m - 1:][:len(v)]) for k, v in paths.items()}).to_csv(OUT / f"fluctuation_{a.market}_{a.block}{'_scaled' if a.scaled else ''}{a.tag}.csv")
    # 4. breaks in monthly log scaled error
    out["breaks"] = {}
    q = idx.tz_convert("Europe/Warsaw").tz_localize(None).to_period("M")
    for m, kind in [(x, k) for x in ("naive", "lear") if x in P for k in ("raw", "level_scaled", "vol_scaled")]:
        e = ev.rmse(P[m], Y)
        e = e if kind == "raw" else e / (L["median_abs"] if kind == "level_scaled" else L["sd"])
        s = pd.Series(np.log(e), index=q).groupby(level=0).mean()
        y = s.to_numpy()
        k, res = ls_breaks(y)
        bks = res[k]["breaks"]
        seg_means = [float(v.mean()) for v in np.split(y, bks)]
        out["breaks"][f"{m}|{kind}"] = {"n_months": int(len(y)), "n_breaks_bic": int(k), "break_months": [str(s.index[b]) for b in bks],
                            "segment_mean_log_scaled_error": seg_means, "shifts": [seg_means[i + 1] - seg_means[i] for i in range(len(bks))],
                            "one_break_month": str(s.index[res[1]["breaks"][0]]) if 1 in res else None,
                            "bic": {int(kk): float(v["bic"]) for kk, v in res.items()},
                            "ci95_months": [[str(s.index[int(np.floor(lo))]), str(s.index[min(len(s) - 1, int(np.ceil(hi)))])] for lo, hi in (break_ci(y, k, bks) if k else [])]}
    f = OUT / f"stability_{a.market}_{a.block}{'_scaled' if a.scaled else ''}{a.tag}.json"
    f.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("fluctuation", "breaks")}, indent=1, default=float)[:3000])
    for m, rows in out.get("decomposition3", {}).items():
        print(m, "3-way", [(x["regime_vs_pre"], round(x["error_ratio"], 2), round(x["level_ratio"], 2), round(x["rel_volatility_ratio"], 2), round(x["efficiency_ratio"], 2),
                            round(x["level_share_%"], 1), round(x["rel_volatility_share_%"], 1), round(x["efficiency_share_%"], 1)) for x in rows])
    for m, d in out["decomposition"].items():
        r = d["median_abs"]
        print(m, [(x["regime_vs_pre"], round(x["error_ratio"], 2), round(x["level_ratio"], 2), round(x["relative_error_ratio"], 2), round(x["level_share_of_log_growth_%"], 1)) for x in r])
    print("->", f)


if __name__ == "__main__":
    main()
