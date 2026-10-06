"""Market input frames for the walk-forward engine.

Each market is one file data/inputs/<MARKET>.parquet with a regular hourly grid and the columns

  price     target day-ahead price (PL: TGE Fixing I in PLN/MWh; DE, SE4: SDAC price in EUR/MWh)
  load_fc   day-ahead total-load forecast of the transmission system operator for that hour (MW)
  origin    True at the forecast-origin hour (00:00 local time)
  dow       day of week of the hour in local time (0 = Monday)
  lday      local calendar day of the hour (integer day number) — used to decide which
            day-ahead forecasts are already published at the origin
  ...       optional further numeric inputs (traded volume, generation by technology, fuel prices)

Missing values are kept as NaN. The engine never fills them: an origin whose inputs or 48-hour
target contain a missing value is skipped, and training rows with missing values are dropped.
A JSON side-car data/inputs/<MARKET>.json records units, sources and the build script.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .paths import INPUTS


@dataclass
class Market:
    name: str
    index: pd.DatetimeIndex
    price: np.ndarray
    load_fc: np.ndarray
    origin: np.ndarray
    dow: np.ndarray
    lday: np.ndarray
    extra: pd.DataFrame
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.index)

    def cols(self, names: list[str]) -> np.ndarray:
        """Extra input columns as float32 array (NaN kept)."""
        if not names:
            return np.zeros((self.n, 0), np.float32)
        missing = [c for c in names if c not in self.extra]
        if missing:
            raise KeyError(f"{self.name}: input columns not available: {missing}")
        return self.extra[names].to_numpy(np.float32)


def load_market(name: str) -> Market:
    f = INPUTS / f"{name}.parquet"
    df = pd.read_parquet(f)
    meta_f = INPUTS / f"{name}.json"
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
    step = pd.Series(df.index[1:] - df.index[:-1]).unique()
    if len(step) != 1 or step[0] != pd.Timedelta(hours=1):
        raise ValueError(f"{name}: index is not a regular hourly grid")
    base = ["price", "load_fc", "origin", "dow", "lday"]
    return Market(name=name, index=df.index,
                  price=df["price"].to_numpy(np.float64),
                  load_fc=df["load_fc"].to_numpy(np.float64),
                  origin=df["origin"].to_numpy(bool),
                  dow=df["dow"].to_numpy(np.int64),
                  lday=df["lday"].to_numpy(np.int64),
                  extra=df.drop(columns=base), meta=meta)
