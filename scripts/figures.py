"""Figures of the paper. All styling and texts come from configs/figures.yaml (language: --lang).

  python scripts/figures.py --lang en            # all figures that have data
  python scripts/figures.py --lang pl --only level
Outputs: results/figures/<lang>/<name>.{png,pdf}
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epfstab import evaluate as ev  # noqa: E402
from epfstab.data import load_market  # noqa: E402
from epfstab.paths import RESULTS  # noqa: E402

CFG = yaml.safe_load((ROOT / "configs" / "figures.yaml").read_text())
CM = 1 / 2.54


def T(key, lang):
    return CFG["text"][key][lang]


def style():
    plt.rcParams.update({"font.family": CFG["font"]["family"], "font.size": CFG["font"]["size"],
                         "axes.edgecolor": CFG["ink"]["secondary"], "axes.labelcolor": CFG["ink"]["primary"],
                         "xtick.color": CFG["ink"]["secondary"], "ytick.color": CFG["ink"]["secondary"],
                         "axes.grid": True, "grid.color": CFG["ink"]["grid"], "grid.linewidth": 0.5,
                         "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
                         "legend.frameon": False, "axes.axisbelow": True, "figure.facecolor": CFG["surface"], "axes.facecolor": CFG["surface"]})


def save(fig, name, lang):
    out = RESULTS / "figures" / lang
    out.mkdir(parents=True, exist_ok=True)
    for f in CFG["formats"]:
        fig.savefig(out / f"{name}.{f}", dpi=CFG["dpi"], bbox_inches="tight")
    plt.close(fig)
    print("->", out / name)


def finished(market, model, run="w730_c7"):
    f = RESULTS / market / run / f"meta_{model}.json"
    return f.exists() and json.loads(f.read_text()).get("finished", False)


def mstyle(key):
    m = CFG["models"].get(key, {"color": "#000000", "marker": "o", "ls": "-"})
    return dict(color=m["color"], marker=m["marker"] or None, ls=m["ls"])


def mlabel(key, lang):
    return CFG["models"].get(key, {}).get(lang, key)


def quarter_index(idx):
    return idx.tz_convert("Europe/Warsaw").tz_localize(None).to_period("Q")


def fig_level(market, lang):
    """Quarterly raw error, level-scaled error and skill of LEAR and the seasonal naïve (one axis per panel)."""
    if not (finished(market, "lear") and finished(market, "naive")):
        return
    mk = load_market(market)
    L, N = ev.load_preds(market, "lear"), ev.load_preds(market, "naive")
    idx = L.index.intersection(N.index)
    PL, Y = ev.split(L.loc[idx]); PN, _ = ev.split(N.loc[idx])
    s = pd.Series(np.abs(mk.price), index=mk.index)
    pos = mk.index.get_indexer(idx)
    lvl = np.array([np.nanmedian(s.iloc[max(0, o - 671):o + 1].to_numpy()) for o in pos])
    q = quarter_index(idx)
    D = pd.DataFrame({"r_l": ev.rmse(PL, Y), "r_n": ev.rmse(PN, Y), "s_l": ev.mae(PL, Y) / lvl, "s_n": ev.mae(PN, Y) / lvl}, index=q)
    G = D.groupby(level=0).median()
    skill = (1 - (D["r_l"] / D["r_n"]).groupby(level=0).median()) * 100
    x = G.index.to_timestamp()
    cur = T("currency_PL" if market == "PL" else "currency_EUR", lang)
    fig, axs = plt.subplots(3, 1, figsize=np.array(CFG["size_cm"]["tall"]) * CM, sharex=True)
    for ax, cols, title in ((axs[0], ("r_l", "r_n"), f"(a) {T('rmse48', lang)} [{cur}]"), (axs[1], ("s_l", "s_n"), f"(b) {T('scaled_mae', lang)}")):
        for c, key in zip(cols, ("lear", "naive")):
            ax.plot(x, G[c], label=mlabel(key, lang), markersize=3, **mstyle(key))
        ax.set_title(title, loc="left", fontsize=CFG["font"]["size"], color=CFG["ink"]["primary"])
    axs[0].legend(loc="upper left")
    axs[2].plot(x, skill, markersize=3, **mstyle("lear"))
    axs[2].axhline(0, color=CFG["ink"]["secondary"], lw=0.8)
    axs[2].set_title(f"(c) {T('skill', lang)}", loc="left", fontsize=CFG["font"]["size"], color=CFG["ink"]["primary"])
    for ax in axs:
        ax.axvline(pd.Timestamp("2021-10-01"), color=CFG["ink"]["secondary"], ls="--", lw=0.8)
    axs[2].text(pd.Timestamp("2021-11-15"), 2, T("break", lang), color=CFG["ink"]["secondary"], fontsize=7, va="bottom")
    axs[2].set_xlabel(T("quarter", lang))
    fig.suptitle(CFG["markets"][market][lang], x=0.01, ha="left", fontsize=CFG["font"]["size"] + 1)
    fig.tight_layout()
    save(fig, f"level_vs_predictability_{market}", lang)


def fig_skill(lang):
    """Skill vs seasonal naïve by model, one panel per market (common origins within each market)."""
    tabs = {m: RESULTS / "tables" / f"models_{m}.csv" for m in ("PL", "DE", "SE4")}
    tabs = {m: pd.read_csv(f, index_col=0) for m, f in tabs.items() if f.exists()}
    if not tabs:
        return
    fig, axs = plt.subplots(1, len(tabs), figsize=np.array(CFG["size_cm"]["double"]) * CM, sharex=True, squeeze=False)
    for ax, (m, t) in zip(axs[0], tabs.items()):
        t = t[~t.index.isin(["naive", "oracle"])].sort_values("skill_rRMSE_%")
        ax.barh([mlabel(k, lang) for k in t.index], t["skill_rRMSE_%"], color=[mstyle(k)["color"] for k in t.index], height=0.7)
        for i, v in enumerate(t["skill_rRMSE_%"]):
            ax.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=6.5, color=CFG["ink"]["primary"])
        ax.set_title(CFG["markets"][m][lang], fontsize=CFG["font"]["size"])
        ax.set_xlabel(T("skill", lang))
        ax.grid(axis="y", visible=False)
    save(fig, "skill_by_market", lang)


def fig_decomposition(lang, block="all"):
    """Three-way decomposition of error growth (stacked shares of the log growth), per market and regime."""
    rows = []
    for m in ("PL", "DE", "SE4"):
        f = RESULTS / "tables" / f"stability_{m}_{block}_scaled.json"     # latest run (raw + scaled losses)
        if not f.exists():
            f = RESULTS / "tables" / f"stability_{m}_{block}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        for r in d.get("decomposition3", {}).get("lear", []):
            rows.append((m, r))
    if not rows:
        return
    parts = [("level", "#8a8986", {"en": "price level", "pl": "poziom cen"}),
             ("rel_volatility", "#2a78d6", {"en": "volatility / level", "pl": "zmienność / poziom"}),
             ("efficiency", "#eb6834", {"en": "error / volatility (model)", "pl": "błąd / zmienność (model)"})]
    fig, ax = plt.subplots(figsize=np.array(CFG["size_cm"]["double"]) * CM)
    labels = [f"{CFG['markets'][m][lang].split(' (')[0]}\n{CFG['regimes'][r['regime_vs_pre']][lang]}" for m, r in rows]
    x = np.arange(len(rows))
    for key, col, lab in parts:
        v = np.array([np.log(r[f"{key}_ratio"]) for _, r in rows])
        pos = np.where(v > 0, v, 0)
        neg = np.where(v < 0, v, 0)
        base_p = np.array([sum(max(0, np.log(r[f"{k}_ratio"])) for k, _, _ in parts[:parts.index((key, col, lab))]) for _, r in rows])
        base_n = np.array([sum(min(0, np.log(r[f"{k}_ratio"])) for k, _, _ in parts[:parts.index((key, col, lab))]) for _, r in rows])
        ax.bar(x, pos, bottom=base_p, color=col, width=0.6, label=lab[lang], edgecolor=CFG["surface"], linewidth=1)
        ax.bar(x, neg, bottom=base_n, color=col, width=0.6, edgecolor=CFG["surface"], linewidth=1)
    for i, (_, r) in enumerate(rows):
        ax.text(i, np.log(r["error_ratio"]) + 0.05, f"×{r['error_ratio']:.1f}", ha="center", fontsize=7, color=CFG["ink"]["primary"])
    ax.axhline(0, color=CFG["ink"]["secondary"], lw=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel({"en": "log of error growth vs pre-crisis", "pl": "log wzrostu błędu wzgl. okresu przed kryzysem"}[lang])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3)
    ax.grid(axis="x", visible=False)
    save(fig, f"error_growth_decomposition_{block}", lang)


def fig_fluctuation(market, lang, block="all", scaled=""):
    f = RESULTS / "tables" / f"fluctuation_{market}_{block}{scaled}.csv"
    j = RESULTS / "tables" / f"stability_{market}_{block}{scaled}.json"
    if not (f.exists() and j.exists()):
        return
    F = pd.read_csv(f, index_col=0, parse_dates=True)
    cv = json.loads(j.read_text())["fluctuation"]
    fig, ax = plt.subplots(figsize=np.array(CFG["size_cm"]["double"]) * CM)
    for col in F.columns:
        a, b = col.split(":")
        ax.plot(F.index, F[col], label=f"{mlabel(a, lang)} vs {mlabel(b, lang)}", color=mstyle(b)["color"], ls=mstyle(b)["ls"], lw=1.2)
    c = next(iter(cv.values()))["crit_5%"]
    for v in (c, -c):
        ax.axhline(v, color=CFG["ink"]["secondary"], ls="--", lw=0.8)
    ax.axhline(0, color=CFG["ink"]["secondary"], lw=0.6)
    ax.axvline(pd.Timestamp("2021-10-01", tz=F.index.tz) if F.index.tz else pd.Timestamp("2021-10-01"), color=CFG["ink"]["secondary"], ls=":", lw=0.8)
    ax.set_ylabel({"en": "fluctuation statistic (negative = first model better)", "pl": "statystyka fluktuacji (ujemna = lepszy pierwszy model)"}[lang])
    ax.legend(loc="lower left", fontsize=7)
    ax.set_title(CFG["markets"][market][lang], loc="left", fontsize=CFG["font"]["size"])
    save(fig, f"fluctuation_{market}_{block}{scaled}", lang)


def fig_window_sweep(market, lang):
    """Median RMSE48 versus training-window length on the origins common to all windows."""
    f = RESULTS / "tables" / f"window_sweep_{market}.csv"
    if not f.exists():
        return
    t = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=np.array(CFG["size_cm"]["single"]) * CM)
    keys = ("naive", "lear", "gbt", "dnn") if (t.model == "gbt").any() else ("naive", "lear", "gbt_nofuel", "dnn")
    for k in keys:                       # one GBT variant per market: with fuel inputs in PL, without elsewhere
        s = t[t.model == k].sort_values("window")
        if s.empty:
            continue
        st = mstyle(k)
        ax.plot(s.window, s.median_RMSE48, color=st["color"], ls=st["ls"], marker=st["marker"], ms=4, label=mlabel(k, lang))
    ax.set_xscale("log")
    w = sorted(t.window.unique())
    ax.set_xticks(w, [str(x) for x in w])
    ax.minorticks_off()
    ax.set_xlabel(T("window", lang))
    unit = T("currency_PL" if market == "PL" else "currency_EUR", lang)
    ax.set_ylabel(f"{T('rmse48', lang)} [{unit}]")
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=4)
    save(fig, f"window_sweep_{market}", lang)


def fig_fuel_inputs(market, lang):
    """Gain from fuel and carbon prices as daily inputs of LEAR, by window, with 95% bootstrap CI."""
    f = RESULTS / "tables" / f"fuel_inputs_{market}.csv"
    if not f.exists():
        return
    t = pd.read_csv(f)
    t = t[t.model == "LEAR"]
    fig, ax = plt.subplots(figsize=np.array(CFG["size_cm"]["single"]) * CM)
    vs = [v for v in CFG["fuel_variants"] if v in set(t.variant)]
    off = {v: 1 + 0.035 * (i - (len(vs) - 1) / 2) for i, v in enumerate(vs)}     # small horizontal dodge (log axis)
    for v in vs:
        s = t[t.variant == v].sort_values("window")
        c = CFG["fuel_variants"][v]
        x = s.window * off[v]
        y = s["gain_mean_MAE_%"]
        ax.errorbar(x, y, yerr=[y - s.ci95_lo, s.ci95_hi - y], color=c["color"], marker=c["marker"], ms=4, lw=1.2 if len(s) > 1 else 0,
                    elinewidth=0.8, capsize=2, label=c[lang])
    ax.axhline(0, color=CFG["ink"]["secondary"], lw=0.8)
    ax.set_xscale("log")
    w = sorted(t.window.unique())
    ax.set_xticks(w, [str(x) for x in w])
    ax.minorticks_off()
    ax.set_xlabel(T("window", lang))
    ax.set_ylabel(T("fuel_gain", lang))
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    save(fig, f"fuel_inputs_vs_window", lang) if market == "PL" else save(fig, f"fuel_inputs_vs_window_{market}", lang)


def fig_cadence(market, lang):
    """LEAR (730 days): change of mean MAE48 relative to daily recalibration, with 95% CI and refit counts."""
    f = RESULTS / "tables" / f"cadence_{market}.csv"
    if not f.exists():
        return
    t = pd.read_csv(f).sort_values("cadence_days").reset_index(drop=True)
    if len(t) < 2:
        return
    fig, ax = plt.subplots(figsize=np.array(CFG["size_cm"]["single"]) * CM)
    x = np.arange(len(t))                           # cadences as equally spaced categories
    y = t["gain_mean_MAE_%"].fillna(0.0).to_numpy()
    lo = (y - t.ci95_lo.fillna(0.0)).to_numpy()
    hi = (t.ci95_hi.fillna(0.0) - y).to_numpy()
    col = mstyle("lear")["color"]
    ax.errorbar(x, y, yerr=[lo, hi], color=col, marker="o", ms=4, lw=1.2, elinewidth=0.8, capsize=2)
    ax.axhline(0, color=CFG["ink"]["secondary"], lw=0.8)
    ax.set_xticks(x, [f"{int(r.cadence_days)}\n({int(r.refits)} {T('refits', lang)})" for _, r in t.iterrows()], fontsize=6.5)
    ax.set_xlabel(T("cadence", lang))
    ax.set_ylabel(T("gain_vs_daily", lang))
    save(fig, "retraining_cadence" if market == "PL" else f"retraining_cadence_{market}", lang)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default=CFG["lang"], choices=["en", "pl"])
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    style()
    if a.only in (None, "level"):
        for m in ("PL", "DE", "SE4"):
            fig_level(m, a.lang)
    if a.only in (None, "skill"):
        fig_skill(a.lang)
    if a.only in (None, "decomposition"):
        fig_decomposition(a.lang)
    if a.only in (None, "sensitivity"):
        fig_window_sweep("PL", a.lang)
        fig_fuel_inputs("PL", a.lang)
        fig_cadence("PL", a.lang)
    if a.only in (None, "fluctuation"):
        for m in ("PL", "DE", "SE4"):
            for sc in ("", "_scaled"):
                fig_fluctuation(m, a.lang, scaled=sc)


if __name__ == "__main__":
    main()
