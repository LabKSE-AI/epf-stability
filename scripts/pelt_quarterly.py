"""PELT change points on the quarterly error series: per-quarter median of the per-origin RMSE48, log10-transformed and z-scored, kernel (rbf)
PELT with minimum segment of 2 quarters, penalty sweep. Signals: LEAR alone and the pool of the naïve benchmark,
LEAR and GBT; raw, level-scaled (by the median |price| of the previous 28 days) and volatility-scaled
(by the standard deviation of the price over the previous 28 days) errors.
Outputs: results/tables/pelt_<M>.csv, pelt.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epfstab import evaluate as ev  # noqa: E402
from epfstab.data import load_market  # noqa: E402
from epfstab.paths import RESULTS  # noqa: E402

PENS = (1, 2, 3, 5, 8, 10)
OUT = RESULTS / "tables"


def scales(mk, idx):
    s = pd.Series(mk.price, index=mk.index)
    lvl = s.abs().rolling(672, min_periods=1).median().reindex(idx).to_numpy()
    vol = s.rolling(672, min_periods=2).std(ddof=0).reindex(idx).to_numpy()
    return lvl, vol


def detect(q: pd.Series, pen):
    x = np.log10(q.to_numpy(float))
    z = (x - x.mean()) / x.std()
    b = rpt.Pelt(model="rbf", min_size=2, jump=1).fit(z.reshape(-1, 1)).predict(pen=pen)
    return [str(q.index[i]) for i in b[:-1]]


def main():
    summary, rows = {}, []
    for M in ("PL", "DE", "SE4"):
        g = "gbt" if (RESULTS / M / "w730_c7" / "meta_gbt.json").exists() else "gbt_nofuel"
        F = {m: ev.load_preds(M, m) for m in ("naive", "lear", g)}
        idx = None
        for d in F.values():
            idx = d.index if idx is None else idx.intersection(d.index)
        Y = ev.split(F["lear"].loc[idx])[1]
        R = {m: ev.rmse(ev.split(d.loc[idx])[0], Y) for m, d in F.items()}
        lvl, vol = scales(load_market(M), idx)
        q = idx.tz_convert("Europe/Warsaw").tz_localize(None).to_period("Q")
        for kind, den in (("raw", 1.0), ("level_scaled", lvl), ("vol_scaled", vol)):
            sig = {"lear": pd.Series(R["lear"] / den, index=q).groupby(level=0).median(),
                   "pooled": pd.concat([pd.Series(R[m] / den, index=q) for m in R]).groupby(level=0).median()}
            for name, s in sig.items():
                for pen in PENS:
                    cps = detect(s, pen)
                    rows.append({"market": M, "signal": name, "errors": kind, "pen": pen, "n_cp": len(cps), "cp_quarters": ";".join(cps)})
        T = pd.DataFrame([r for r in rows if r["market"] == M])
        T.to_csv(OUT / f"pelt_{M}.csv", index=False)
        summary[M] = {"n_origins": int(len(idx)), "n_quarters": int(q.nunique()), "models": list(F)}
        print(f"== {M} ({len(idx)} origins, {q.nunique()} quarters)")
        print(T.pivot_table(index=["signal", "errors"], columns="pen", values="cp_quarters", aggfunc="first").to_string())
    (OUT / "pelt.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
