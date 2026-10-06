"""Economic value of the forecasts: day-ahead arbitrage of a battery storage
system scheduled with the forecasts.

At each origin (00:00 of day D) the forecast for the next delivery day D+1 (forecast hours 24…47,
i.e. a genuine day-ahead decision taken before the D+1 auction) is used to choose one charging block
of k consecutive hours and one later discharging block of k hours that maximise the forecast spread;
the schedule is executed only if the forecast profit is positive. Realised profit uses actual prices:
  profit = eta * sum(discharge prices) - sum(charge prices)        [per MW of power, k MWh capacity]
with round-trip efficiency eta. Reported: total and per-regime profit, share of the perfect-foresight
profit, and the profit of every model relative to LEAR.

  python scripts/economic.py --market PL --models naive lear gbt ens_inv30 --k 2 --eta 0.85
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epfstab import evaluate as ev  # noqa: E402
from epfstab.paths import RESULTS  # noqa: E402


def schedule(p: np.ndarray, k: int, eta: float):
    """Best (charge start i, discharge start j>=i+k) on a 24-h price vector p."""
    c = np.convolve(p, np.ones(k), "valid")             # block sums, start 0..24-k
    best, arg = -np.inf, None
    for i in range(len(c)):
        j0 = i + k
        if j0 >= len(c):
            break
        j = j0 + int(np.argmax(c[j0:]))
        v = eta * c[j] - c[i]
        if v > best:
            best, arg = v, (i, j)
    return best, arg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="PL")
    ap.add_argument("--models", nargs="+", default=["naive", "lear", "gbt", "ens_inv30"])
    ap.add_argument("--k", type=int, default=2, help="hours of charging = hours of discharging (capacity k MWh per MW)")
    ap.add_argument("--eta", type=float, default=0.85, help="round-trip efficiency")
    a = ap.parse_args()
    base = {m: ev.load_preds(a.market, m) for m in a.models if not m.startswith("ens_")}
    idx = None
    for d in base.values():
        idx = d.index if idx is None else idx.intersection(d.index)
    P = {m: ev.split(d.loc[idx])[0] for m, d in base.items()}
    Y = ev.split(base[next(iter(base))].loc[idx])[1]
    if "ens_inv30" in a.models:
        g = "gbt" if "gbt" in P else "gbt_nofuel"
        P["ens_inv30"], _ = ev.combine(P["lear"], P[g], Y, "inv", 30)
    D1 = slice(23, 47)          # forecast hours 24..47 = delivery day D+1 (rows o+24 … o+47)
    prof = {}
    for m, Pm in list(P.items()) + [("perfect foresight", Y)]:
        out = np.zeros(len(idx))
        for t in range(len(idx)):
            v, arg = schedule(Pm[t, D1], a.k, a.eta)
            if arg is not None and v > 0:
                i, j = arg
                y = Y[t, D1]
                out[t] = a.eta * y[j:j + a.k].sum() - y[i:i + a.k].sum()
        prof[m] = out
    rg = ev.regimes(idx)
    rows = []
    for m, v in prof.items():
        r = {"model": m, "total": v.sum(), "per_day_mean": v.mean(),
             "share_of_perfect_%": 100 * v.sum() / prof["perfect foresight"].sum()}
        for reg in ("pre", "crisis", "post"):
            r[f"total_{reg}"] = v[rg == reg].sum()
        if "lear" in prof and m != "lear":
            est, lo, hi = ev.boot_ci(-v, -prof["lear"], reps=1000)     # gain in mean profit vs LEAR, %
            r.update({"vs_LEAR_%": -est, "ci95_lo": -hi, "ci95_hi": -lo})
        rows.append(r)
    T = pd.DataFrame(rows).set_index("model").round(2)
    cur = "PLN" if a.market == "PL" else "EUR"
    out = RESULTS / "tables" / f"economic_{a.market}_k{a.k}.csv"
    T.to_csv(out)
    print(f"{a.market}: {len(idx)} days {idx.min().date()}..{idx.max().date()}, profit in {cur} per MW")
    print(T.to_string())
    print("->", out)


if __name__ == "__main__":
    main()
