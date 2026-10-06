"""Direct multi-horizon models on tabular features: seasonal naïve, LEAR, gradient-boosted trees.

Specification follows the paper:
  LEAR  LASSO-estimated autoregressive model (after Lago et al., 2021), one LASSO per horizon
        h = 1..48, penalty chosen by AIC (LassoLarsIC), z-scored features:
          * 96 price lags p(o-k), k in 0..71 and 144..167 (days -1, -2, -3, -7)
          * day-ahead load forecast at the delivery hour and 24 h / 168 h before it
          * day-of-week dummies of the delivery hour
          * optionally fuel/CO2 prices at the origin
        trained on pseudo-origins every 2 hours of the training window.
  GBT   histogram gradient-boosted trees (scikit-learn), one model per horizon, on the LEAR
        design plus five volatility/spike features of the past prices; pseudo-origins every 3 hours.
  naive seasonal naïve: ŷ(o+h) = p(o+h-168).

Features are held in single precision (float32), as in the original code (this matters for the
binning of the gradient-boosted trees).

Load-forecast availability: with loadfc="full" the forecast for the
delivery hour is used at every horizon, as in the paper. With loadfc="available" only load
forecasts already published at a 00:00 origin are used, i.e. for hours of the origin day
(rows o..o+23): for h = 24..47 the term at lag 0 is dropped, for h = 48 the lags 0 and 24 are
replaced by lag 48.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..data import Market
from ..protocol import H

P_LAGS = list(range(0, 72)) + list(range(144, 168))
MAXLAG = max(P_LAGS)          # 167, as in the paper's code (defines the first pseudo-origin of a window)
FL_LAGS = (0, 24, 168)


def lag_matrix(p: np.ndarray, lags: list[int]) -> np.ndarray:
    P = np.full((len(p), len(lags)), np.nan, np.float32)
    for j, k in enumerate(lags):
        P[k:, j] = p[: len(p) - k].astype(np.float32)
    return P


def fl_lags_for(h: int, mode: str) -> tuple[int, ...]:
    if mode == "full":
        return FL_LAGS
    if mode != "available":
        raise ValueError(mode)
    if h <= 23:
        return FL_LAGS
    if h <= 47:
        return (24, 168)
    return (48, 168)


def spike_features(p: np.ndarray) -> np.ndarray:
    """Volatility/spike features of past and current prices (NaN where the window is incomplete)."""
    s = pd.Series(p)
    mean168 = s.rolling(168).mean()
    return np.column_stack([s.rolling(24).std(), s.rolling(168).std(),
                            s.rolling(24).max() - s.rolling(24).min(),
                            s - mean168, s / (mean168 + 1e-6)]).astype(np.float32)


class Naive:
    name = "naive"
    maxlag = 168
    needs_fit = False

    def __init__(self, mk: Market, **_):
        self.p = mk.price

    def inputs_ok(self, o: int) -> bool:
        return bool(np.isfinite(self.p[o + 1 - 168:o + 1 + H - 168]).all())

    def predict(self, O: np.ndarray) -> np.ndarray:
        return np.vstack([self.p[o + 1 - 168:o + 1 + H - 168] for o in O])


def _fit_lear(X, y):
    from sklearn.linear_model import LassoLarsIC
    import warnings
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    X, y = X[ok], y[ok]
    mu, sd = X.mean(0), X.std(0) + 1e-8
    m = LassoLarsIC(criterion="aic", max_iter=100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit((X - mu) / sd, y)
    return (m, mu, sd, int(ok.sum()))


def _fit_gbt(X, y, hp, seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    m = HistGradientBoostingRegressor(random_state=seed, early_stopping=False, **hp)
    m.fit(X[ok], y[ok])
    return (m, None, None, int(ok.sum()))


class _Direct:
    """Shared logic of the per-horizon models (LEAR, GBT)."""
    needs_fit = True
    stride = 2

    def __init__(self, mk: Market, fuels: list[str] | None = None, loadfc: str = "full", jobs: int = 8, **_):
        self.p = mk.price.astype(np.float32).astype(np.float64)   # single precision, as in the original code
        self.fl = mk.load_fc.astype(np.float32)
        self.fuels = mk.cols(fuels or [])
        self.dow1h = np.eye(7, dtype=np.float32)[mk.dow]
        self.P = lag_matrix(self.p, P_LAGS)
        self.loadfc = loadfc
        self.jobs = jobs
        self.models: dict = {}
        self.n_train: list[int] = []

    def design(self, O: np.ndarray, h: int) -> np.ndarray:
        parts = [self.P[O], self.fuels[O]] + [self.fl[O + h - lag][:, None] for lag in fl_lags_for(h, self.loadfc)] \
            + [self.dow1h[O + h]]
        return np.column_stack(parts)

    def inputs_ok(self, o: int) -> bool:
        O = np.array([o])
        return all(np.isfinite(self.design(O, h)).all() for h in range(1, H + 1))

    def fit(self, O: np.ndarray) -> None:
        # module-level fit functions: only (X, y) are sent to the workers, not the whole model object
        res = Parallel(n_jobs=self.jobs)(delayed(self._fit_fn)(self.design(O, h), self.p[O + h].astype(np.float64), *self._fit_args)
                                         for h in range(1, H + 1))
        self.models = {h: r for h, r in zip(range(1, H + 1), res)}
        self.n_train = [r[3] for r in res]

    def predict(self, O: np.ndarray) -> np.ndarray:
        cols = []
        for h in range(1, H + 1):
            m, mu, sd, _ = self.models[h]
            X = self.design(O, h)
            cols.append(m.predict(X if mu is None else (X - mu) / sd))
        return np.column_stack(cols)


class LEAR(_Direct):
    name = "lear"
    maxlag = MAXLAG
    stride = 2

    _fit_fn = staticmethod(_fit_lear)
    _fit_args = ()


class GBT(_Direct):
    name = "gbt"
    maxlag = MAXLAG
    stride = 3
    DEFAULT_HP = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=0.1)

    def __init__(self, mk: Market, hp: dict | None = None, seed: int = 12345, **kw):
        super().__init__(mk, **kw)
        self.hp = dict(self.DEFAULT_HP, **(hp or {}))
        self.seed = seed
        self.EX = spike_features(self.p.astype(np.float32))
        os.environ.setdefault("OMP_NUM_THREADS", "1")

    def design(self, O: np.ndarray, h: int) -> np.ndarray:
        return np.column_stack([super().design(O, h), self.EX[O]])

    _fit_fn = staticmethod(_fit_gbt)

    @property
    def _fit_args(self):
        return (self.hp, self.seed)


class NaiveSimilarDay:
    """Similar-day naïve benchmark common in EPF (e.g. Weron, 2014; Lago et al., 2021): for delivery
    days that are Mondays, Saturdays or Sundays the price one week earlier, otherwise the price one
    day earlier (same hour)."""
    name = "naive_sd"
    maxlag = 168
    needs_fit = False

    def __init__(self, mk: Market, **_):
        self.p = mk.price
        self.dow = mk.dow

    def _lags(self, o):
        r = np.arange(o + 1, o + 1 + H)
        return r - np.where(np.isin(self.dow[r], (0, 5, 6)), 168, 24)

    def inputs_ok(self, o: int) -> bool:
        return bool(np.isfinite(self.p[self._lags(o)]).all())

    def predict(self, O: np.ndarray) -> np.ndarray:
        return np.vstack([self.p[self._lags(o)] for o in O])


MODELS = {"naive": Naive, "naive_sd": NaiveSimilarDay, "lear": LEAR, "gbt": GBT}
