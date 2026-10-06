"""Recurrent sequence-to-vector networks (LSTM, BiLSTM, GRU) as specified in the paper.

Input: the last L hours of [exogenous inputs + price + calendar (sin/cos of hour and day of week)],
standardised with the mean and standard deviation of the training window; output: the next 48
hourly prices in one shot (direct multi-output head). Loss MSE, Adam, gradient clipping 1.0,
fixed number of epochs (no early stopping), retrained every `cadence` days on all hourly
pseudo-origins of the training window. Windows containing a missing value are not used.

Each (model, seed) pair has its own random generator, so seed ensembles are reproducible and
independent of the order in which models are run.
"""
from __future__ import annotations

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn

from ..data import Market
from ..protocol import EMBARGO, H

UNTUNED = dict(L=336, hidden=96, fc=128, dropout=0.15, lr=0.0008, batch_size=64, epochs=10, num_layers=1)
TUNED_LSTM = dict(L=168, hidden=224, fc=64, dropout=0.4, lr=0.0006447727024796179, batch_size=32, epochs=20, num_layers=1)
# exogenous inputs of the Polish network models (generation by technology and load from the
# master file, Fixing I volume). Series whose history starts later than the study period
# (solar before 04.2020, "other", "other_renewable") are not used.
PL_RNN_INPUTS = ["volume", "hard_coal", "coalderived", "lignite", "gas", "oil", "biomass", "wind_onshore",
                 "hydro_pumped_storage", "hydro_runofriver_and_poundage", "hydro_water_reservoir", "electricity_load"]


class RNNForecaster(nn.Module):
    def __init__(self, n_feat, hidden, fc, dropout, kind, num_layers=1):
        super().__init__()
        bidir = kind == "bilstm"
        cls = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn = cls(n_feat, hidden, num_layers=num_layers, batch_first=True, bidirectional=bidir,
                       dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden * (2 if bidir else 1), fc), nn.ReLU(), nn.Dropout(dropout), nn.Linear(fc, H))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class RNN:
    needs_fit = True
    stride = 1

    def __init__(self, mk: Market, kind: str = "lstm", hp: dict | None = None, inputs: list[str] | None = None,
                 seed: int = 1, train_days: int = 730, device: str | None = None, **_):
        self.kind = kind
        self.hp = dict(hp or UNTUNED)
        self.L = int(self.hp["L"])
        self.maxlag = self.L - 1
        self.name = kind
        self.seed = seed
        self.train_days = train_days
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        idx = mk.index.tz_convert("Europe/Warsaw") if mk.index.tz is not None else mk.index
        h, dow = idx.hour.to_numpy(), idx.dayofweek.to_numpy()
        cal = np.column_stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                               np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)])
        ex = mk.cols(inputs or [])
        self.X = np.column_stack([ex, mk.price[:, None], cal]).astype(np.float32)
        self.y = mk.price.astype(np.float32)
        self.n_feat = self.X.shape[1]
        rowok = np.isfinite(self.X).all(1)
        # win_ok[o] — the L-hour input window ending at o is complete
        c = np.concatenate([[0], np.cumsum(~rowok)])
        self.win_ok = np.zeros(len(self.y), bool)
        self.win_ok[self.L - 1:] = (c[self.L:] - c[:-self.L]) == 0
        yok = np.isfinite(self.y).astype(int)
        cy = np.concatenate([[0], np.cumsum(1 - yok)])
        self.tgt_ok = np.zeros(len(self.y), bool)
        self.tgt_ok[: len(self.y) - H] = (cy[1 + H: len(self.y) + 1] - cy[1: len(self.y) - H + 1]) == 0
        self.gen = torch.Generator(device="cpu").manual_seed(seed * 1000 + {"lstm": 1, "bilstm": 2, "gru": 3}[kind])

    def inputs_ok(self, o: int) -> bool:
        return bool(self.win_ok[o])

    def fit_at(self, o: int) -> None:
        te = o - EMBARGO
        ts = te - self.train_days * 24
        tr = self.X[ts:te]
        self.muX = np.nanmean(tr, 0)
        self.sdX = np.nanstd(tr, 0) + 1e-6
        self.muY = float(np.nanmean(self.y[ts:te]))
        self.sdY = float(np.nanstd(self.y[ts:te]) + 1e-6)
        O = np.arange(ts + self.L - 1, te - H)
        O = O[self.win_ok[O] & self.tgt_ok[O]]
        Xs = np.nan_to_num((self.X - self.muX) / self.sdX).astype(np.float32)   # NaN rows never enter a used window
        Ys = np.nan_to_num((self.y - self.muY) / self.sdY).astype(np.float32)
        vx = sliding_window_view(Xs, self.L, axis=0)
        Xw = torch.from_numpy(vx[O - self.L + 1].transpose(0, 2, 1).copy()).to(self.device)
        Yw = torch.from_numpy(sliding_window_view(Ys, H)[O + 1].copy()).to(self.device)
        torch.manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=self.gen)))
        hp = self.hp
        self.net = RNNForecaster(self.n_feat, hp["hidden"], hp["fc"], hp["dropout"], self.kind, hp.get("num_layers", 1)).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=hp["lr"])
        lossf = nn.MSELoss()
        self.net.train()
        n, bs = len(Xw), int(hp["batch_size"])
        for _ in range(int(hp["epochs"])):
            perm = torch.randperm(n, generator=self.gen).to(self.device)
            for i in range(0, n, bs):
                b = perm[i:i + bs]
                opt.zero_grad()
                loss = lossf(self.net(Xw[b]), Yw[b])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
        self.net.eval()
        self.n_train = int(n)

    def predict(self, O: np.ndarray) -> np.ndarray:
        Xs = ((self.X - self.muX) / self.sdX).astype(np.float32)
        xin = np.stack([Xs[o - self.L + 1:o + 1] for o in O])
        with torch.no_grad():
            pr = self.net(torch.from_numpy(xin).to(self.device)).cpu().numpy()
        return pr * self.sdY + self.muY
