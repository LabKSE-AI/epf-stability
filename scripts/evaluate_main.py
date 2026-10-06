"""Main comparison tables of the paper (can be run on partial results).

For every market: all available models at the 730-day window on the set of origins common to all
of them. Models with several random seeds enter as the seed ensemble (average of the seeds'
48-hour forecast vectors); the spread of single seeds is reported separately.
Outputs (results/tables/): models_<M>.csv, combination_<M>.csv, dm_vs_lear_<M>.csv, mcs_<M>.csv,
seeds_<M>.csv, summary.json
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epfstab import evaluate as ev  # noqa: E402
from epfstab.paths import RESULTS  # noqa: E402

RUN = "w730_c7"
OUT = RESULTS / "tables"
OUT.mkdir(parents=True, exist_ok=True)
LABEL = {"naive": "seasonal naïve", "naive_sd": "similar-day naïve", "lear": "LEAR", "gbt": "GBT (PLN fuels)",
         "gbt_nofuel": "GBT", "lstm_untuned": "LSTM", "bilstm_untuned": "BiLSTM", "gru_untuned": "GRU",
         "lstm_tuned": "LSTM (tuned)", "dnn": "DNN", "dnn_ens": "DNN-Ens", "nbeatsx": "N-BEATSx", "tft": "TFT"}


def available(market):
    d = RESULTS / market / RUN
    fam = {}
    for f in d.glob("meta_*.json"):
        m = json.loads(f.read_text())
        if not m.get("finished"):
            continue
        name = f.stem[5:]
        if name.endswith(("_avail", "_fuel", "_fuelpln", "_co2", "_co2pln")):
            continue
        base = re.sub(r"_s\d+$", "", name)
        fam.setdefault(base, []).append(name)
    return fam


def family_forecast(market, members):
    dfs = [ev.load_preds(market, m, RUN) for m in members]
    idx = dfs[0].index
    for d in dfs[1:]:
        idx = idx.intersection(d.index)
    Ps = [ev.split(d.loc[idx])[0] for d in dfs]
    _, Y = ev.split(dfs[0].loc[idx])
    return idx, np.mean(Ps, axis=0), Y, Ps


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", choices=["all", "d1"], default="all",
                    help="all 48 forecast hours, or only the next delivery day (forecast hours 24..47, rows o+24..o+47)")
    a = ap.parse_args()
    sl = slice(0, 48) if a.block == "all" else slice(23, 47)
    sfx = "" if a.block == "all" else "_d1"
    summary = {}
    for market in ("PL", "DE", "SE4"):
        fam = available(market)
        if "naive" not in fam or "lear" not in fam:
            continue
        F = {}
        for base, members in sorted(fam.items()):
            F[base] = family_forecast(market, sorted(members))
        common = None
        for idx, *_ in F.values():
            common = idx if common is None else common.intersection(idx)
        rg = ev.regimes(common)
        P = {k: pd.DataFrame(v[1], index=v[0]).loc[common].to_numpy()[:, sl] for k, v in F.items()}
        Y = pd.DataFrame(F["lear"][2], index=F["lear"][0]).loc[common].to_numpy()[:, sl]
        if "gbt" in P or "gbt_nofuel" in P:
            g = "gbt" if "gbt" in P else "gbt_nofuel"
            P["ens_avg"], _ = ev.combine(P["lear"], P[g], Y, "avg")
            P["ens_inv30"], w = ev.combine(P["lear"], P[g], Y, "inv", 30)
            for W in (7, 14):
                P[f"ens_inv{W}"], _ = ev.combine(P["lear"], P[g], Y, "inv", W)
            P["oracle"] = np.where((ev.rmse(P["lear"], Y) <= ev.rmse(P[g], Y))[:, None], P["lear"], P[g])
        R = {k: ev.rmse(v, Y) for k, v in P.items()}
        A = {k: ev.mae(v, Y) for k, v in P.items()}
        rows = []
        for k in P:
            r, a = R[k], A[k]
            rows.append({"model": LABEL.get(k, k), "key": k, "seeds": len(fam.get(k, [k])) if k in fam else 0,
                         "median_RMSE48": np.median(r), "pre": np.median(r[rg == "pre"]), "crisis": np.median(r[rg == "crisis"]),
                         "post": np.median(r[rg == "post"]), "mean_MAE48": a.mean(),
                         "skill_rRMSE_%": (1 - np.median(r / R["naive"])) * 100,
                         "rMAE_median": np.median(a / A["naive"]), "beats_naive_%": (r < R["naive"]).mean() * 100,
                         "vs_LEAR_median_%": (1 - np.median(r) / np.median(R["lear"])) * 100})
        T = pd.DataFrame(rows).set_index("key")
        T.round(3).to_csv(OUT / f"models_{market}{sfx}.csv")
        # DM tests vs LEAR (MAE48 per origin), Holm over all comparisons
        dm = {k: ev.dm_test(A[k], A["lear"]) for k in P if k not in ("lear", "oracle")}
        ph = ev.holm({k: v["p"] for k, v in dm.items()})
        D = pd.DataFrame({k: dict(v, p_holm=ph[k]) for k, v in dm.items()}).T
        D.round(5).to_csv(OUT / f"dm_vs_lear_{market}{sfx}.csv")
        # combination: adaptive (inv-MSE W30) vs equal weights and vs LEAR, per regime
        if "ens_inv30" in P:
            C = []
            for reg in ("all", "pre", "crisis", "post"):
                m = np.ones(len(rg), bool) if reg == "all" else rg == reg
                for a, b in (("ens_inv30", "ens_avg"), ("ens_inv30", "lear"), ("ens_avg", "lear")):
                    t = ev.dm_test(A[a][m], A[b][m])
                    est, lo, hi = ev.boot_ci(A[a][m], A[b][m], reps=1000)
                    C.append({"regime": reg, "model": a, "vs": b, "n": t["n"], "gain_mean_MAE_%": est,
                              "ci95_lo": lo, "ci95_hi": hi, "dm": t["dm"], "p": t["p"]})
            pd.DataFrame(C).round(4).to_csv(OUT / f"combination_{market}{sfx}.csv", index=False)
        # MCS on MAE48 (models with seeds as seed ensembles; oracle excluded)
        L = pd.DataFrame({k: A[k] for k in P if k not in ("oracle",)}, index=common)
        try:
            ev.mcs(L).round(4).to_csv(OUT / f"mcs_{market}{sfx}.csv")
        except Exception as e:      # noqa: BLE001
            print("MCS failed:", e)
        # seed spread
        S = []
        for base, members in fam.items():
            if len(members) > 1:
                med = [np.median(ev.rmse(Pi[np.isin(F[base][0], common)][:, sl], Y)) for Pi in F[base][3]]
                S.append({"model": base, "n_seeds": len(members), "median_RMSE48_mean": np.mean(med),
                          "median_RMSE48_sd": np.std(med, ddof=1), "seed_ensemble": np.median(R[base])})
        if S:
            pd.DataFrame(S).round(3).to_csv(OUT / f"seeds_{market}{sfx}.csv", index=False)
        summary[market] = {"n_common": int(len(common)), "first": str(common.min()), "last": str(common.max()),
                           "models": sorted(P), "lear_gbt_error_corr": float(np.corrcoef(R["lear"], R.get("gbt", R.get("gbt_nofuel", R["lear"])))[0, 1])}
        print(f"\n=== {market}: {len(common)} common origins {common.min().date()}..{common.max().date()}")
        print(T[["model", "seeds", "median_RMSE48", "pre", "crisis", "post", "skill_rRMSE_%", "rMAE_median", "vs_LEAR_median_%"]].round(2).to_string())
    (OUT / f"summary{sfx}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
