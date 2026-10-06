"""Hyper-parameter search (Optuna TPE) for the additional deep-learning benchmarks, using only data
BEFORE the first evaluated origin: calibration on all data up to 2017-06-30 (minus the embargo),
validation = daily origins 2017-07-01 … 2017-12-29 forecast by the calibrated model (no
recalibration), objective = mean MAE48. The evaluated period starts in January 2018.

  python scripts/tune_nn.py --market PL --model dnn --trials 100
  python scripts/tune_nn.py --market PL --model nbeatsx --trials 15
  python scripts/tune_nn.py --market PL --model tft --trials 15
Output: configs/tuned_<model>_<market>.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")   # before numpy/sklearn: avoids CPU oversubscription; results unchanged
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less GPU memory fragmentation
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
torch.set_num_threads(int(os.environ.get("EPF_TORCH_THREADS", "2")))   # do not starve the CPU queues
from epfstab.data import load_market  # noqa: E402
from epfstab.models.dnn import DNN  # noqa: E402
from epfstab.models.nf import NF  # noqa: E402
from epfstab.protocol import EMBARGO, H, target_ok  # noqa: E402

CAL_END = "2017-07-01"
VAL_END = "2017-12-30"


def space(trial, model):
    if model == "dnn":
        return dict(n1=trial.suggest_int("n1", 64, 512, step=32), n2=trial.suggest_int("n2", 32, 256, step=16),
                    act=trial.suggest_categorical("act", ["relu", "softplus", "tanh", "selu", "leakyrelu", "prelu", "sigmoid"]),
                    dropout=trial.suggest_float("dropout", 0.0, 0.5), batchnorm=trial.suggest_categorical("batchnorm", [False, True]),
                    l1=trial.suggest_float("l1", 1e-6, 1e-2, log=True), lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                    scaler=trial.suggest_categorical("scaler", ["std", "invariant"]),
                    use_d2=trial.suggest_categorical("use_d2", [True, False]), use_d3=trial.suggest_categorical("use_d3", [True, False]),
                    use_d7=trial.suggest_categorical("use_d7", [True, False]), use_flag=trial.suggest_categorical("use_flag", [True, False]),
                    use_dow=trial.suggest_categorical("use_dow", [True, False]))
    if model == "nbeatsx":
        w = trial.suggest_categorical("width", [256, 512, 1024])
        nb = trial.suggest_int("n_blocks", 1, 3)
        return dict(input_size=trial.suggest_categorical("input_size", [72, 168, 336]),
                    learning_rate=trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
                    mlp_units=[[w, w]] * 3, n_blocks=[nb] * 3, stack_types=["identity"] * 3,
                    dropout_prob_theta=trial.suggest_float("dropout_prob_theta", 0.0, 0.3),
                    batch_size=trial.suggest_categorical("batch_size", [32, 64]))
    return dict(input_size=trial.suggest_categorical("input_size", [72, 168, 336]),
                hidden_size=trial.suggest_categorical("hidden_size", [32, 64, 128]),
                n_head=trial.suggest_categorical("n_head", [2, 4]), dropout=trial.suggest_float("dropout", 0.0, 0.3),
                learning_rate=trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True)
    ap.add_argument("--model", required=True, choices=["dnn", "nbeatsx", "tft"])
    ap.add_argument("--trials", type=int, default=20)
    a = ap.parse_args()
    mk = load_market(a.market)
    t = mk.index
    tz = t.tz
    cal_o = int(np.searchsorted(t, pd.Timestamp(CAL_END, tz="Europe/Warsaw").tz_convert(tz) if tz else pd.Timestamp(CAL_END)))
    vend = int(np.searchsorted(t, pd.Timestamp(VAL_END, tz="Europe/Warsaw").tz_convert(tz) if tz else pd.Timestamp(VAL_END)))
    te = cal_o - EMBARGO
    val = np.array([o for o in range(cal_o, vend) if mk.origin[o] and target_ok(mk.price, o)])
    log = []
    t0 = time.time()

    def objective(trial):
        hp = space(trial, a.model)
        if a.model == "dnn":
            m = DNN(mk, hp=hp, seed=1)
            O = np.arange(167, te - H)
            m.fit_rows(O[mk.origin[O]])
        else:
            m = NF(mk, model=a.model, hp=hp, seed=1)
            m.fit_rows(m.L, te)
        O = np.array([o for o in val if m.inputs_ok(o)])
        P = m.predict(O)
        Y = np.vstack([mk.price[o + 1:o + 1 + H] for o in O])
        mae = float(np.mean(np.abs(P - Y)))
        log.append({"trial": trial.number, "mae48": mae, "n_val": int(len(O)), "params": trial.params, "sec": time.time() - t0})
        print(f"trial {trial.number}: MAE48 {mae:.2f} ({len(O)} origins) {trial.params} [{time.time() - t0:.0f}s]", flush=True)
        return mae

    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=12345))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st.optimize(objective, n_trials=a.trials, catch=(RuntimeError,))      # e.g. CUDA out of memory: trial skipped
    done = sorted([tr for tr in st.trials if tr.value is not None], key=lambda tr: tr.value)
    failed = [tr.number for tr in st.trials if tr.value is None]
    cfg = {"market": a.market, "model": a.model, "calibration_end": CAL_END, "validation": [CAL_END, VAL_END],
           "objective": "mean MAE48 on validation origins", "n_trials": a.trials,
           "best_value": done[0].value, "failed_trials": failed, "log": log}
    if a.model == "dnn":
        cfg["top"] = [space(optuna.trial.FixedTrial(tr.params), "dnn") for tr in done[:4]]
        cfg["top_values"] = [tr.value for tr in done[:4]]
    else:
        cfg["best"] = space(optuna.trial.FixedTrial(done[0].params), a.model)
    (ROOT / "configs" / f"tuned_{a.model}_{a.market}.json").write_text(json.dumps(cfg, indent=1))
    print("best", done[0].value, done[0].params)


if __name__ == "__main__":
    main()
