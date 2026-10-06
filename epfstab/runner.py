"""Walk-forward runner for direct (per-horizon) models, resumable, with a heartbeat file.

Outputs in results/<market>/<run>/:
  per_origin_<model>.csv   one row per evaluated origin: origin_utc, rmse48, mae48, smape48, retrain
  preds_<model>/           48-hour forecast and actual vectors, one compressed file per retraining block
  meta_<model>.json        configuration, numbers of evaluated and skipped origins, timings
  heartbeat_<model>.json   progress, updated after every block (used by scripts/watchdog.py)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_market
from .metrics import metrics48
from .paths import RESULTS
from .protocol import EMBARGO, H, origin_positions, retrain_flags, target_ok, train_rows


def run_direct(market: str, model_cls, train_days: int, cadence: int, run: str, model_kw: dict,
               origin_start: str | None = None, origin_end: str | None = None, smoke: bool = False) -> Path:
    mk = load_market(market)
    model = model_cls(mk, **model_kw)
    name = model.name + model_kw.get("suffix", "")
    out = RESULTS / market / run
    (out / f"preds_{name}").mkdir(parents=True, exist_ok=True)
    first = model.maxlag + train_days * 24 + EMBARGO
    cand = origin_positions(mk.origin, first, mk.n)
    t = mk.index
    if origin_start:
        cand = cand[np.asarray(t[cand] >= pd.Timestamp(origin_start, tz=t.tz))]
    if origin_end:
        cand = cand[np.asarray(t[cand] <= pd.Timestamp(origin_end, tz=t.tz))]
    skipped = {"target_missing": 0, "inputs_missing": 0}
    valid = []
    for o in cand:
        if not target_ok(mk.price, o):
            skipped["target_missing"] += 1
        elif not model.inputs_ok(o):
            skipped["inputs_missing"] += 1
        else:
            valid.append(o)
    valid = np.array(valid, int)
    if smoke:
        valid = valid[: 3 * cadence]
    flags = retrain_flags(valid, cadence)
    po = out / f"per_origin_{name}.csv"
    done = set(pd.read_csv(po)["origin_utc"]) if po.exists() else set()
    if not po.exists():
        po.write_text("origin_utc,rmse48,mae48,smape48,retrain\n")
    hb = out / f"heartbeat_{name}.json"
    t0 = time.time()
    n_done = len(done)
    blocks = np.flatnonzero(flags).tolist() + [len(valid)]
    meta = {"market": market, "model": name, "train_days": train_days, "cadence_days": cadence,
            "embargo_hours": EMBARGO, "H": H, "model_kw": {k: v for k, v in model_kw.items() if k not in ("jobs", "device")},
            "n_candidates": int(len(cand)), "skipped": skipped, "n_valid": int(len(valid)),
            "first_origin": str(t[valid[0]]) if len(valid) else None,
            "last_origin": str(t[valid[-1]]) if len(valid) else None, "pid": os.getpid()}
    (out / f"meta_{name}.json").write_text(json.dumps(meta, indent=1))
    print(f"[cfg] {market} {name} W={train_days}d cadence={cadence}d valid={len(valid)} skipped={skipped} "
          f"resume_done={len(done)} -> {out}", flush=True)
    for b in range(len(blocks) - 1):
        k0, k1 = blocks[b], blocks[b + 1]
        O = valid[k0:k1]
        todo = np.array([o for o in O if str(t[o]) not in done], int)
        if len(todo) == 0:
            continue
        tb = time.time()
        if model.needs_fit:
            if hasattr(model, "fit_at"):
                model.fit_at(O[0])
            else:
                model.fit(train_rows(O[0], train_days, model.maxlag, model.stride))
        YP = model.predict(todo)
        rows, YT = [], []
        for i, o in enumerate(todo):
            yt = mk.price[o + 1:o + 1 + H]
            m = metrics48(yt, YP[i])
            rows.append(f"{t[o]},{m['rmse48']:.4f},{m['mae48']:.4f},{m['smape48']:.4f},{int(o == O[0])}\n")
            YT.append(yt)
        with open(po, "a") as fh:
            fh.writelines(rows)
        np.savez_compressed(out / f"preds_{name}" / f"block_{k0:05d}.npz",
                            origins=np.array([str(t[o]) for o in todo]), y_pred=YP, y_true=np.vstack(YT))
        n_done += len(todo)
        el = time.time() - t0
        rate = el / max(1, n_done - len(done))
        hb.write_text(json.dumps({"market": market, "model": name, "run": run, "pid": os.getpid(),
                                  "updated": time.time(), "done": n_done, "total": int(len(valid)),
                                  "last_origin": str(t[todo[-1]]), "sec_per_origin": rate,
                                  "eta_min": rate * (len(valid) - n_done) / 60,
                                  "last_block_sec": time.time() - tb}))
    meta.update({"finished": True, "runtime_sec": time.time() - t0})
    (out / f"meta_{name}.json").write_text(json.dumps(meta, indent=1))
    print(f"[done] {market} {name}: {n_done}/{len(valid)} origins in {time.time() - t0:.0f}s", flush=True)
    return out
