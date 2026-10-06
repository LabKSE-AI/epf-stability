"""Evaluation utilities: loading forecasts, per-origin losses, regimes, forecast combination,
Diebold–Mariano tests with HAC variance, Holm correction, stationary-bootstrap confidence
intervals and the Model Confidence Set.

Regimes (quarters of the origin date, local time): pre-crisis up to 2021-Q3, crisis 2021-Q4 … 2022-Q4,
post-crisis from 2023-Q1 (2021-Q4 = change point found by PELT in the paper).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .paths import RESULTS

CS, CE = pd.Period("2021Q4"), pd.Period("2022Q4")
LAG = 2   # combination weights use errors of origins at least 2 days old (their 48-h targets are realised)


def load_preds(market: str, model: str, run: str = "w730_c7") -> pd.DataFrame:
    """Forecasts of one model: DataFrame indexed by origin (UTC) with columns p0..p47, y0..y47.
    A family name without its own forecasts (e.g. 'dnn') returns the seed ensemble of its finished seeds
    ('dnn_s1', 'dnn_s2', …): the average of the seeds' forecast vectors on their common origins."""
    d = RESULTS / market / run / f"preds_{model}"
    if not d.exists():
        import json
        import re
        seeds = sorted(f.stem[5:] for f in (RESULTS / market / run).glob(f"meta_{model}_s*.json")
                       if re.fullmatch(rf"{model}_s\d+", f.stem[5:]) and json.loads(f.read_text()).get("finished"))
        if not seeds:
            raise FileNotFoundError(d)
        dfs = [load_preds(market, m, run) for m in seeds]
        idx = dfs[0].index
        for x in dfs[1:]:
            idx = idx.intersection(x.index)
        pc = [f"p{i}" for i in range(48)]
        out = dfs[0].loc[idx].copy()
        out[pc] = np.mean([x.loc[idx, pc].to_numpy() for x in dfs], axis=0)
        return out
    parts = []
    for f in sorted(d.glob("block_*.npz")):
        z = np.load(f)
        idx = pd.DatetimeIndex(z["origins"])
        parts.append(pd.DataFrame(np.hstack([z["y_pred"], z["y_true"]]), index=idx,
                                  columns=[f"p{i}" for i in range(48)] + [f"y{i}" for i in range(48)]))
    if not parts:
        raise FileNotFoundError(d)
    df = pd.concat(parts).sort_index()
    return df[~df.index.duplicated(keep="last")]


def split(df: pd.DataFrame):
    return df[[f"p{i}" for i in range(48)]].to_numpy(), df[[f"y{i}" for i in range(48)]].to_numpy()


def rmse(P, Y):
    return np.sqrt(np.mean((P - Y) ** 2, axis=1))


def mae(P, Y):
    return np.mean(np.abs(P - Y), axis=1)


def regimes(idx: pd.DatetimeIndex, tz: str = "Europe/Warsaw") -> np.ndarray:
    q = idx.tz_convert(tz).tz_localize(None).to_period("Q") if idx.tz is not None else idx.to_period("Q")
    return np.where(q < CS, "pre", np.where(q <= CE, "crisis", "post"))


def combine(P1, P2, Y, scheme: str = "inv", W: int = 30):
    """Combination of two forecast vectors per origin. 'avg' = 0.5/0.5; 'inv' = weights inversely
    proportional to the mean squared error of each model over the last W origins whose targets are
    realised (origins t-LAG-W+1 … t-LAG), as in the paper. Returns combined forecasts and weights of model 1."""
    if scheme == "avg":
        return 0.5 * P1 + 0.5 * P2, np.full(len(P1), 0.5)
    m1, m2 = rmse(P1, Y) ** 2, rmse(P2, Y) ** 2
    w = np.full(len(P1), 0.5)
    for t in range(len(P1)):
        lo, hi = max(0, t - LAG - W + 1), t - LAG + 1
        if hi > 0:
            a, b = m1[lo:hi].mean() + 1e-6, m2[lo:hi].mean() + 1e-6
            w[t] = (1 / a) / (1 / a + 1 / b)
    return w[:, None] * P1 + (1 - w[:, None]) * P2, w


def dm_test(l1: np.ndarray, l2: np.ndarray, h: int = 2) -> dict:
    """Diebold–Mariano test of equal mean loss (two-sided) on per-origin losses.
    Variance: Newey–West (Bartlett) with bandwidth max(h-1, floor(4 (T/100)^(2/9))), because daily
    origins with 48-hour targets overlap; Harvey–Leybourne–Newbold small-sample correction; t(T-1)."""
    d = np.asarray(l1, float) - np.asarray(l2, float)
    d = d[np.isfinite(d)]
    T = len(d)
    q = max(h - 1, int(np.floor(4 * (T / 100) ** (2 / 9))))
    dc = d - d.mean()
    g = [np.dot(dc[k:], dc[:T - k]) / T for k in range(q + 1)]
    lrv = g[0] + 2 * sum((1 - k / (q + 1)) * g[k] for k in range(1, q + 1))
    dm = d.mean() / np.sqrt(lrv / T)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=T - 1)
    return {"n": T, "mean_diff": float(d.mean()), "dm": float(stat), "p": float(p), "bandwidth": q}


def holm(p: dict) -> dict:
    keys = sorted(p, key=p.get)
    m, out, run = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * p[k]))
        out[k] = run
    return out


def boot_ci(l1: np.ndarray, l2: np.ndarray, stat: str = "rel_mean", reps: int = 2000, block: int = 7, seed: int = 12345):
    """Stationary bootstrap (mean block length `block` origins) 95 % CI of the relative difference of
    mean losses, 100 * (1 - mean(l1) / mean(l2)) (positive = model 1 better), or of medians."""
    from arch.bootstrap import StationaryBootstrap
    x = np.column_stack([l1, l2])
    bs = StationaryBootstrap(block, x, seed=seed)
    f = (lambda a: 100 * (1 - a[:, 0].mean() / a[:, 1].mean())) if stat == "rel_mean" else \
        (lambda a: 100 * (1 - np.median(a[:, 0]) / np.median(a[:, 1])))
    draws = np.array([f(d[0][0]) for d in bs.bootstrap(reps)])
    return float(f(x)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def mcs(losses: pd.DataFrame, size: float = 0.10, reps: int = 2000, block: int = 7, seed: int = 12345) -> pd.Series:
    """Model Confidence Set (Hansen, Lunde & Nason, 2011) on per-origin losses (columns = models).
    Returns the MCS p-value of every model; models with p >= size are in the set."""
    from arch.bootstrap import MCS
    m = MCS(losses, size=size, reps=reps, block_size=block, method="R", seed=seed)
    m.compute()
    return m.pvalues["Pvalue"].reindex(losses.columns)
