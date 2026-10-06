"""DNN of the open-access EPF benchmark (Lago, Marcjasz, De Schutter & Weron, 2021), re-implemented in
PyTorch and adapted to the 48-hour target of this study.

One sample per daily origin o (00:00 local time). Inputs (feature groups, each can be switched off by
the hyper-parameter search, as in the original DNN, except the D-1 prices which are always used):
  price lags at the origin: hours of days D-1 (lags 0..23), D-2 (24..47), D-3 (48..71), D-7 (144..167)
  day-ahead load forecast for the 48 target hours and for the same hours one week earlier
  day-of-week one-hot of the origin day
Output: the 48 hourly prices. Two hidden layers; activation, sizes, dropout, batch normalisation, L1
penalty on the first layer, learning rate and input/output scaling are hyper-parameters (TPE search
in scripts/tune_nn.py on 2016–2017 data, before the first evaluated origin). Adam, MAE loss, early
stopping on the chronologically last 10 % of the training days (patience 20 epochs, at most 1000).
DNN-Ens = average of the four best distinct configurations (Lago et al. combine four DNNs).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..data import Market
from ..protocol import EMBARGO, H

GROUPS = {"d1": list(range(0, 24)), "d2": list(range(24, 48)), "d3": list(range(48, 72)), "d7": list(range(144, 168))}
ACT = {"relu": nn.ReLU, "softplus": nn.Softplus, "tanh": nn.Tanh, "selu": nn.SELU, "leakyrelu": nn.LeakyReLU,
       "prelu": nn.PReLU, "sigmoid": nn.Sigmoid}
DEFAULT_HP = dict(n1=256, n2=128, act="relu", dropout=0.1, batchnorm=False, l1=1e-4, lr=1e-3, scaler="std",
                  use_d2=True, use_d3=True, use_d7=True, use_flag=True, use_dow=True)


def asinh_scale(x, med, mad):
    return np.arcsinh((x - med) / mad)


def asinh_inv(z, med, mad):
    return np.sinh(z) * mad + med


class DNNNet(nn.Module):
    def __init__(self, n_in, hp):
        super().__init__()
        layers = []
        for a, b in ((n_in, hp["n1"]), (hp["n1"], hp["n2"])):
            layers.append(nn.Linear(a, b))
            if hp["batchnorm"]:
                layers.append(nn.BatchNorm1d(b))
            layers.append(ACT[hp["act"]]())
            if hp["dropout"] > 0:
                layers.append(nn.Dropout(hp["dropout"]))
        layers.append(nn.Linear(hp["n2"], H))
        self.f = nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x)


class DNN:
    needs_fit = True
    maxlag = 167                         # same first origin as LEAR (price lags 0..167)
    stride = 24

    def __init__(self, mk: Market, hp: dict | None = None, seed: int = 1, train_days: int = 730,
                 device: str | None = None, name: str = "dnn", **_):
        self.hp = dict(DEFAULT_HP, **(hp or {}))
        self.name = name
        self.seed = seed
        self.train_days = train_days
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        p = mk.price
        lags = GROUPS["d1"] + [k for g in ("d2", "d3", "d7") if self.hp[f"use_{g}"] for k in GROUPS[g]]
        n = len(p)
        P = np.full((n, len(lags)), np.nan)
        for j, k in enumerate(lags):
            P[k:, j] = p[:n - k]
        fl = mk.load_fc
        FL = np.full((n, 2 * H if self.hp["use_flag"] else H), np.nan)
        for h in range(1, H + 1):
            FL[: n - h, h - 1] = fl[h:]
            if self.hp["use_flag"]:
                ok = np.arange(n) + h - 168 >= 0
                FL[ok, H + h - 1] = fl[(np.arange(n) + h - 168)[ok]]
        parts = [P, FL]
        if self.hp["use_dow"]:
            parts.append(np.eye(7)[mk.dow])
        self.X = np.column_stack(parts).astype(np.float64)
        self.Y = np.full((n, H), np.nan)
        for h in range(1, H + 1):
            self.Y[: n - h, h - 1] = p[h:]
        self.origin = mk.origin
        self.rng = np.random.default_rng(seed)

    def inputs_ok(self, o: int) -> bool:
        return bool(np.isfinite(self.X[o]).all())

    def _scale_fit(self, X, Y):
        if self.hp["scaler"] == "std":
            self.mx, self.sx = X.mean(0), X.std(0) + 1e-6
            self.my, self.sy = Y.mean(), Y.std() + 1e-6
        else:                                           # "invariant": asinh of median/MAD-standardised values
            self.mx = np.median(X, 0)
            self.sx = np.median(np.abs(X - self.mx), 0) / 0.6745 + 1e-6
            self.my = np.median(Y)
            self.sy = np.median(np.abs(Y - self.my)) / 0.6745 + 1e-6

    def _tx(self, X):
        return (X - self.mx) / self.sx if self.hp["scaler"] == "std" else asinh_scale(X, self.mx, self.sx)

    def _ty(self, Y):
        return (Y - self.my) / self.sy if self.hp["scaler"] == "std" else asinh_scale(Y, self.my, self.sy)

    def _ty_inv(self, Z):
        return Z * self.sy + self.my if self.hp["scaler"] == "std" else asinh_inv(Z, self.my, self.sy)

    def fit_rows(self, O: np.ndarray) -> None:
        O = O[np.isfinite(self.X[O]).all(1) & np.isfinite(self.Y[O]).all(1)]
        X, Y = self.X[O], self.Y[O]
        self._scale_fit(X, Y)
        Xs = torch.tensor(self._tx(X), dtype=torch.float32, device=self.device)
        Ys = torch.tensor(self._ty(Y), dtype=torch.float32, device=self.device)
        nv = max(7, int(0.1 * len(O)))
        Xt, Yt, Xv, Yv = Xs[:-nv], Ys[:-nv], Xs[-nv:], Ys[-nv:]
        torch.manual_seed(int(self.rng.integers(0, 2 ** 31 - 1)))
        net = DNNNet(X.shape[1], self.hp).to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=self.hp["lr"])
        first = net.f[0]
        best, best_state, bad = np.inf, None, 0
        bs = 32
        for _ in range(1000):
            net.train()
            perm = torch.randperm(len(Xt), device=self.device)
            for i in range(0, len(Xt), bs):
                b = perm[i:i + bs]
                if len(b) < 2 and self.hp["batchnorm"]:
                    continue
                opt.zero_grad()
                loss = (net(Xt[b]) - Yt[b]).abs().mean() + self.hp["l1"] * first.weight.abs().sum() / first.weight.numel()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float((net(Xv) - Yv).abs().mean())
            if v < best - 1e-6:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 20:
                    break
        net.load_state_dict(best_state)
        net.eval()
        self.net = net
        self.n_train = int(len(O))
        self.val_mae = best

    def fit_at(self, o: int) -> None:
        te = o - EMBARGO
        ts = te - self.train_days * 24
        O = np.arange(ts + self.maxlag, te - H)
        self.fit_rows(O[self.origin[O]])

    def predict(self, O: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z = self.net(torch.tensor(self._tx(self.X[O]), dtype=torch.float32, device=self.device)).cpu().numpy()
        return self._ty_inv(z)


class DNNEnsemble:
    """Average of several DNNs with different hyper-parameter configurations."""
    needs_fit = True
    maxlag = DNN.maxlag
    stride = 24

    def __init__(self, mk: Market, hps: list[dict], seed: int = 1, name: str = "dnn_ens", **kw):
        self.members = [DNN(mk, hp=hp, seed=seed * 100 + i, **kw) for i, hp in enumerate(hps)]
        self.name = name

    def inputs_ok(self, o):
        return all(m.inputs_ok(o) for m in self.members)

    def fit_at(self, o):
        for m in self.members:
            m.fit_at(o)

    def predict(self, O):
        return np.mean([m.predict(O) for m in self.members], axis=0)
