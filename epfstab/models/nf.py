"""N-BEATSx (Olivares et al., 2023) and the Temporal Fusion Transformer (Lim et al., 2021) from the
`neuralforecast` library (Nixtla), wrapped into the same rolling walk-forward protocol.

Target: the next 48 hourly prices after the origin; input window `input_size` hours of past prices.
Exogenous inputs known in advance (futr_exog): day-ahead load forecast and calendar (sin/cos of the
hour of day and of the day of week). Scaling: "robust" temporal normalisation inside each window.
Loss MAE. Training uses only data up to EMBARGO hours before the origin; the training series is cut at
missing values into contiguous segments, so no window containing a missing value is used.
"""
from __future__ import annotations

import logging
import os
import warnings

import numpy as np
import pandas as pd

from ..data import Market
from ..protocol import EMBARGO, H

os.environ.setdefault("NIXTLA_ID_AS_COL", "1")
warnings.filterwarnings("ignore")
for _n in ("pytorch_lightning", "lightning", "lightning.pytorch", "lightning_fabric", "neuralforecast"):
    logging.getLogger(_n).setLevel(logging.ERROR)

FUTR = ["load_fc", "h_sin", "h_cos", "d_sin", "d_cos"]
NBEATSX_HP = dict(input_size=168, max_steps=1000, learning_rate=1e-3, batch_size=32, windows_batch_size=512,
                  mlp_units=[[512, 512], [512, 512], [512, 512]], n_blocks=[1, 1, 1],
                  stack_types=["identity", "identity", "identity"], dropout_prob_theta=0.1)
TFT_HP = dict(input_size=168, max_steps=500, learning_rate=1e-3, batch_size=32, windows_batch_size=256,
              hidden_size=64, n_head=4, dropout=0.1)          # lighter than N-BEATSx: TFT is ~4x slower per step


def _trainer_kw():
    return dict(accelerator="gpu" if __import__("torch").cuda.is_available() else "cpu", devices=1,
                enable_progress_bar=False, logger=False, enable_model_summary=False, enable_checkpointing=False)


class NF:
    needs_fit = True
    stride = 1

    def __init__(self, mk: Market, model: str = "nbeatsx", hp: dict | None = None, seed: int = 1,
                 train_days: int = 730, val_days: int = 14, name: str | None = None, **_):
        self.kind = model
        self.hp = dict(NBEATSX_HP if model == "nbeatsx" else TFT_HP, **(hp or {}))
        self.L = int(self.hp["input_size"])
        self.maxlag = 167 if self.L <= 168 else self.L - 1      # same first origin as LEAR when possible
        self.name = name or model
        self.seed = seed
        self.train_days = train_days
        self.val = val_days * 24
        loc = mk.index.tz_convert("Europe/Warsaw") if mk.index.tz is not None else mk.index
        h, d = loc.hour.to_numpy(), loc.dayofweek.to_numpy()
        self.df = pd.DataFrame({"ds": mk.index.tz_localize(None) if mk.index.tz is not None else mk.index,
                                "y": mk.price, "load_fc": mk.load_fc,
                                "h_sin": np.sin(2 * np.pi * h / 24), "h_cos": np.cos(2 * np.pi * h / 24),
                                "d_sin": np.sin(2 * np.pi * d / 7), "d_cos": np.cos(2 * np.pi * d / 7)})
        self.rowok = np.isfinite(self.df[["y"] + FUTR].to_numpy()).all(1)
        self.exok = np.isfinite(self.df[FUTR].to_numpy()).all(1)

    def inputs_ok(self, o: int) -> bool:
        return bool(self.rowok[o - self.L + 1:o + 1].all() and self.exok[o + 1:o + 1 + H].all())

    def _model(self):
        from neuralforecast.losses.pytorch import MAE
        kw = {k: v for k, v in self.hp.items()}
        common = dict(h=H, futr_exog_list=FUTR, scaler_type="robust", loss=MAE(), random_seed=self.seed,
                      early_stop_patience_steps=5, val_check_steps=100, **_trainer_kw())
        if self.kind == "nbeatsx":
            from neuralforecast.models import NBEATSx
            return NBEATSx(**common, **kw)
        from neuralforecast.models import TFT
        return TFT(**common, **kw)

    def fit_rows(self, ts: int, te: int) -> None:
        from neuralforecast import NeuralForecast
        ok = self.rowok[ts:te]
        seg = np.cumsum(np.r_[1, np.diff(ok.astype(int)) != 0])
        parts = []
        for sid in np.unique(seg[ok]):
            rows = np.flatnonzero((seg == sid) & ok) + ts
            if len(rows) >= self.L + H + self.val + 1:
                d = self.df.iloc[rows].copy()
                d.insert(0, "unique_id", f"s{sid}")
                parts.append(d)
        train = pd.concat(parts, ignore_index=True)
        self.nf = NeuralForecast(models=[self._model()], freq="h")
        self.nf.fit(df=train, val_size=self.val)
        self.n_train = int(len(train))

    def fit_at(self, o: int) -> None:
        te = o - EMBARGO
        self.fit_rows(te - self.train_days * 24, te)

    def predict(self, O: np.ndarray) -> np.ndarray:
        hist, fut = [], []
        for o in O:
            a = self.df.iloc[o - self.L + 1:o + 1].copy(); a.insert(0, "unique_id", f"o{o}")
            b = self.df.iloc[o + 1:o + 1 + H][["ds"] + FUTR].copy(); b.insert(0, "unique_id", f"o{o}")
            hist.append(a); fut.append(b)
        fc = self.nf.predict(df=pd.concat(hist, ignore_index=True), futr_df=pd.concat(fut, ignore_index=True))
        col = [c for c in fc.columns if c not in ("unique_id", "ds")][0]
        fc = fc.sort_values(["unique_id", "ds"])
        out = np.vstack([fc.loc[fc.unique_id == f"o{o}", col].to_numpy() for o in O])
        return out
