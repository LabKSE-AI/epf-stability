"""Rolling walk-forward protocol.

At every daily origin o (row of the 00:00 local hour) the model forecasts the next H = 48 hourly
prices, rows o+1 … o+48, using information up to and including row o. Models are recalibrated
every `cadence` days on a training window of `train_days` days that ends `EMBARGO` hours before
the origin. Training samples are "pseudo-origins" o' inside the window whose 48-hour targets are
all realised before the window end, so no target of the evaluated origin (nor any later value)
can enter training.
"""
from __future__ import annotations

import numpy as np

H = 48
EMBARGO = 48
WEEK = 168


def origin_positions(origin_flag: np.ndarray, first: int, n: int) -> np.ndarray:
    pos = np.arange(n)
    return pos[(pos >= first) & (pos <= n - H - 1) & origin_flag]


def train_rows(o: int, train_days: int, maxlag: int, stride: int) -> np.ndarray:
    """Pseudo-origins used to (re)calibrate at origin o (identical rule to the paper's code)."""
    te = o - EMBARGO
    ts = te - train_days * 24
    return np.arange(ts + maxlag + 1, te - H, stride)


def retrain_flags(origins: np.ndarray, cadence_days: int) -> np.ndarray:
    """Retrain at the first origin and whenever `cadence_days` have passed since the last
    retraining (on the list of evaluated origins). Deterministic, so runs can be resumed."""
    flags = np.zeros(len(origins), bool)
    last = -10 ** 12
    for k, o in enumerate(origins):
        if o - last >= cadence_days * 24:
            flags[k] = True
            last = o
    return flags


def target_ok(price: np.ndarray, o: int) -> bool:
    return bool(np.isfinite(price[o + 1:o + 1 + H]).all())
