"""Probabilistic evaluation: predictive quantiles built on top of the
point forecasts, recalibrated every 7 days on the last C origins whose 48-hour targets are already
realised (origins at least 2 days before the forecast origin — no look-ahead).

  QRA   quantile regression averaging on a single forecast (Nowotarski & Weron, 2015): for each
        level tau, y = a + b * yhat, estimated on the calibration origins pooled over the 48 hours;
  EMP   empirical error quantiles: yhat + tau-quantile of past errors of the same horizon.
Quantile levels 0.05, 0.10, …, 0.95; crossing quantiles are sorted (monotone rearrangement).
Scores: pinball loss averaged over levels and hours, empirical coverage and mean width of the central
50 % and 90 % intervals, per regime; Kupiec-type binomial test of the 90 % coverage.

  python scripts/probabilistic.py --market PL --models lear gbt ens_inv30 --calib 182
"""
import argparse
import warnings
warnings.filterwarnings("ignore", message="Maximum number of iterations")
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epfstab import evaluate as ev  # noqa: E402
from epfstab.paths import RESULTS  # noqa: E402

TAUS = np.round(np.arange(0.05, 0.96, 0.05), 2)
LAG = 2


def qr_fit(x, y, taus):
    """Linear quantile regression y = a + b x for several taus (statsmodels QuantReg)."""
    import statsmodels.api as sm
    X = sm.add_constant(x)
    out = []
    for t in taus:
        try:
            out.append(sm.QuantReg(y, X).fit(q=t, max_iter=5000).params)
        except Exception:      # noqa: BLE001 — degenerate window: fall back to empirical shift
            out.append(np.array([np.quantile(y - x, t), 1.0]))
    return np.array(out)                                   # (n_tau, 2)


def quantiles(P, Y, method, C, cadence=7):
    n, H = P.shape
    Q = np.full((n, H, len(TAUS)), np.nan)
    coef = None
    for t in range(n):
        lo, hi = t - LAG - C + 1, t - LAG + 1
        if lo < 0:
            continue
        if method == "emp":
            E = Y[lo:hi] - P[lo:hi]
            Q[t] = P[t][:, None] + np.quantile(E, TAUS, axis=0).T
        else:
            if coef is None or t % cadence == 0:          # one regression per level, pooled over the 48 hours
                coef = qr_fit(P[lo:hi].ravel(), Y[lo:hi].ravel(), TAUS)                       # (n_tau, 2)
            Q[t] = coef[None, :, 0] + coef[None, :, 1] * P[t][:, None]
        Q[t] = np.sort(Q[t], axis=1)
    return Q


def pinball(Q, Y):
    d = Y[..., None] - Q
    return np.mean(np.maximum(TAUS * d, (TAUS - 1) * d), axis=(1, 2))      # per origin


def interval(Q, Y, lo, hi):
    il, ih = int(np.argmin(np.abs(TAUS - lo))), int(np.argmin(np.abs(TAUS - hi)))
    cover = (Y >= Q[:, :, il]) & (Y <= Q[:, :, ih])
    return cover.mean(axis=1), (Q[:, :, ih] - Q[:, :, il]).mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="PL")
    ap.add_argument("--models", nargs="+", default=["lear", "gbt", "ens_inv30"])
    ap.add_argument("--calib", type=int, default=182, help="calibration origins")
    a = ap.parse_args()
    base = {}
    for m in a.models:
        if m.startswith("ens_"):
            continue
        base[m] = ev.load_preds(a.market, m)
    idx = None
    for d in base.values():
        idx = d.index if idx is None else idx.intersection(d.index)
    P = {m: ev.split(d.loc[idx])[0] for m, d in base.items()}
    Y = ev.split(base[next(iter(base))].loc[idx])[1]
    if "ens_inv30" in a.models:
        g = "gbt" if "gbt" in P else "gbt_nofuel"
        P["ens_inv30"], _ = ev.combine(P["lear"], P[g], Y, "inv", 30)
    rg = ev.regimes(idx)
    rows = []
    for m in a.models:
        for meth in ("qra", "emp"):
            Q = quantiles(P[m], Y, meth, a.calib)
            ok = np.isfinite(Q).all(axis=(1, 2))
            pb = pinball(Q[ok], Y[ok])
            c50, w50 = interval(Q[ok], Y[ok], 0.25, 0.75)
            c90, w90 = interval(Q[ok], Y[ok], 0.05, 0.95)
            r = rg[ok]
            for reg in ("all", "pre", "crisis", "post"):
                s = np.ones(len(r), bool) if reg == "all" else r == reg
                k = int(np.round(c90[s].mean() * s.sum() * Y.shape[1]))
                N = int(s.sum() * Y.shape[1])
                rows.append({"model": m, "method": meth, "regime": reg, "n_origins": int(s.sum()),
                             "pinball": pb[s].mean(), "cov50_%": 100 * c50[s].mean(), "width50": w50[s].mean(),
                             "cov90_%": 100 * c90[s].mean(), "width90": w90[s].mean(),
                             "kupiec90_p_naive": float(stats.binomtest(k, N, 0.9).pvalue)})
            print(f"{m:10} {meth}: pinball {pb.mean():.2f}  cov90 {100 * c90.mean():.1f}%  width90 {w90.mean():.1f}", flush=True)
    out = RESULTS / "tables" / f"prob_{a.market}.csv"
    pd.DataFrame(rows).round(4).to_csv(out, index=False)
    print("->", out, "(note: the binomial p-value treats hours as independent — indicative only)")


if __name__ == "__main__":
    main()
