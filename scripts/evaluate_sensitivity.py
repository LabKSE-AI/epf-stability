"""Sensitivity analyses of the paper (can be run on partial results; missing runs are skipped).

1. Training-window sweep (Fig. 4): median RMSE48 and mean MAE48 of each model and window on the origins
   common to all windows (the longest window determines the first common origin).
2. Fundamentals as daily inputs (Table 4, Fig. 3): LEAR with fuel and carbon prices (native currency, PLN,
   daily carbon price only) against LEAR without them at the same window, on pairwise common origins;
   GBT with and without PLN fuel inputs at 730 days. Gain in mean MAE48 (positive = fundamentals help),
   95% stationary-bootstrap CI, Diebold–Mariano test (HAC + HLN).
3. Recalibration cadence (Fig. 6): LEAR (730 days) recalibrated every 1, 7, 14, 30 days, against daily.
Outputs (results/tables/): window_sweep_<M>.csv, fuel_inputs_<M>.csv, cadence_<M>.csv, sensitivity.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epfstab import evaluate as ev  # noqa: E402
from epfstab.paths import RESULTS  # noqa: E402

OUT = RESULTS / "tables"
WINDOWS = (90, 120, 180, 365, 548, 730, 1095, 1460)
SWEEP_MODELS = ("naive", "lear", "lear_fuel", "lear_fuelpln", "gbt", "gbt_nofuel", "dnn")
_cache = {}


def finished(market, run, name):
    f = RESULTS / market / run / f"meta_{name}.json"
    return f.exists() and json.loads(f.read_text()).get("finished", False)


def family(market, run, base):
    """Seed members of a model family in a run (e.g. dnn -> dnn_s1..s3); single models -> [base]."""
    if finished(market, run, base):
        return [base]
    d = RESULTS / market / run
    return sorted(f.stem[5:] for f in d.glob(f"meta_{base}_s*.json")
                  if re.fullmatch(rf"{base}_s\d+", f.stem[5:]) and finished(market, run, f.stem[5:]))


def preds(market, run, base):
    """Forecast of a family (seed ensemble = average of the seeds' forecast vectors) as a DataFrame."""
    key = (market, run, base)
    if key not in _cache:
        mem = family(market, run, base)
        if not mem:
            _cache[key] = None
        else:
            dfs = [ev.load_preds(market, m, run) for m in mem]
            idx = dfs[0].index
            for d in dfs[1:]:
                idx = idx.intersection(d.index)
            P = np.mean([ev.split(d.loc[idx])[0] for d in dfs], axis=0)
            Y = ev.split(dfs[0].loc[idx])[1]
            _cache[key] = (idx, P, Y, len(mem))
    return _cache[key]


def compare(a, b):
    """Gain of forecast a over b in mean MAE48 on their common origins (positive = a better)."""
    idx = a[0].intersection(b[0])
    ia, ib = a[0].get_indexer(idx), b[0].get_indexer(idx)
    la, lb = ev.mae(a[1][ia], a[2][ia]), ev.mae(b[1][ib], b[2][ib])
    est, lo, hi = ev.boot_ci(la, lb, reps=1000)
    dm = ev.dm_test(la, lb)
    reg = ev.regimes(idx)
    by = {r: float(100 * (1 - la[reg == r].mean() / lb[reg == r].mean())) for r in ("pre", "crisis", "post") if (reg == r).any()}
    return {"n": len(idx), "gain_mean_MAE_%": est, "ci95_lo": lo, "ci95_hi": hi, "dm": dm["dm"], "p": dm["p"],
            **{f"gain_{r}_%": v for r, v in by.items()}}


def window_sweep(market):
    runs = {(w, m): preds(market, f"w{w}_c7", m) for w in WINDOWS for m in SWEEP_MODELS}
    runs = {k: v for k, v in runs.items() if v is not None}
    if not runs:
        return None, {}
    idx = None
    for v in runs.values():
        idx = v[0] if idx is None else idx.intersection(v[0])
    rows = []
    reg = ev.regimes(idx)
    for (w, m), (i, P, Y, ns) in sorted(runs.items()):
        j = i.get_indexer(idx)
        r, a = ev.rmse(P[j], Y[j]), ev.mae(P[j], Y[j])
        rows.append({"window": w, "model": m, "seeds": ns, "n_common": len(idx), "median_RMSE48": np.median(r),
                     "mean_MAE48": a.mean(), **{f"median_RMSE48_{g}": np.median(r[reg == g]) for g in ("pre", "crisis", "post") if (reg == g).any()}})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / f"window_sweep_{market}.csv", index=False, float_format="%.4f")
    info = {"n_common": len(idx), "first": str(idx.min()), "last": str(idx.max())} if len(idx) else {}
    return t, info


def fuel_inputs(market):
    rows = []
    for w in WINDOWS:
        base = preds(market, f"w{w}_c7", "lear")
        if base is None:
            continue
        for v in ("lear_fuel", "lear_fuelpln", "lear_co2", "lear_co2pln"):
            x = preds(market, f"w{w}_c7", v)
            if x is not None:
                rows.append({"window": w, "model": "LEAR", "variant": v.split("_", 1)[1], **compare(x, base)})
    g0 = preds(market, "w730_c7", "gbt_nofuel")
    for v in ("gbt", "gbt_co2pln"):
        x = preds(market, "w730_c7", v)
        if g0 is not None and x is not None:
            rows.append({"window": 730, "model": "GBT", "variant": "fuelpln" if v == "gbt" else "co2pln", **compare(x, g0)})
    t = pd.DataFrame(rows)
    if len(t):
        t.to_csv(OUT / f"fuel_inputs_{market}.csv", index=False, float_format="%.4f")
    return t


def cadence(market):
    runs = {c: preds(market, f"w730_c{c}", "lear") for c in (1, 7, 14, 30)}
    runs = {c: v for c, v in runs.items() if v is not None}
    if 1 not in runs:
        return pd.DataFrame()
    idx = None
    for v in runs.values():
        idx = v[0] if idx is None else idx.intersection(v[0])
    rows = []
    for c, (i, P, Y, _) in sorted(runs.items()):
        j = i.get_indexer(idx)
        cmp = compare((idx, P[j], Y[j]), (idx, runs[1][1][runs[1][0].get_indexer(idx)], runs[1][2][runs[1][0].get_indexer(idx)])) if c != 1 else {}
        days = np.asarray((i - i[0]).days)            # recalibration every c days from the first origin
        rows.append({"cadence_days": c, "n_common": len(idx), "median_RMSE48": np.median(ev.rmse(P[j], Y[j])),
                     "mean_MAE48": ev.mae(P[j], Y[j]).mean(), "refits": len(np.unique(days // c)),
                     **{k: v for k, v in cmp.items() if k != "n"}})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / f"cadence_{market}.csv", index=False, float_format="%.4f")
    return t


def window_pairs(market):
    """Four-year versus two-year window for every family run with both (pairwise common origins)."""
    rows = []
    for base in ("lear", "gbt", "gbt_nofuel", "dnn", "dnn_ens"):
        a, b = preds(market, "w1460_c7", base), preds(market, "w730_c7", base)
        if a is not None and b is not None:
            idx = a[0].intersection(b[0])
            ia, ib = a[0].get_indexer(idx), b[0].get_indexer(idx)
            ra, rb = ev.rmse(a[1][ia], a[2][ia]), ev.rmse(b[1][ib], b[2][ib])
            reg = ev.regimes(idx)
            rows.append({"model": base, "seeds": a[3], "first": str(idx.min()), "median_RMSE48_1460": np.median(ra),
                         "median_RMSE48_730": np.median(rb),
                         **{f"median_RMSE48_{g}_1460": np.median(ra[reg == g]) for g in ("pre", "crisis", "post") if (reg == g).any()},
                         **{f"median_RMSE48_{g}_730": np.median(rb[reg == g]) for g in ("pre", "crisis", "post") if (reg == g).any()},
                         **compare(a, b)})
    t = pd.DataFrame(rows)
    if len(t):
        t.to_csv(OUT / f"window_pairs_{market}.csv", index=False, float_format="%.4f")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", nargs="+", default=["PL", "DE", "SE4"])
    a = ap.parse_args()
    summary = {}
    for m in a.markets:
        t, info = window_sweep(m)
        f = fuel_inputs(m)
        c = cadence(m)
        wp = window_pairs(m)
        summary[m] = {"window_sweep": info, "fuel_rows": len(f), "cadence_rows": len(c)}
        print(f"== {m}: window sweep {info}")
        if t is not None:
            print(t.pivot(index="window", columns="model", values="median_RMSE48").round(2).to_string())
        if len(f):
            print(f[["window", "model", "variant", "n", "gain_mean_MAE_%", "ci95_lo", "ci95_hi", "p"]].round(3).to_string(index=False))
        if len(c):
            print(c.round(3).to_string(index=False))
        if len(wp):
            print(wp[["model", "seeds", "n", "median_RMSE48_1460", "median_RMSE48_730", "gain_mean_MAE_%", "ci95_lo", "ci95_hi", "p"]].round(3).to_string(index=False))
    (OUT / "sensitivity.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
