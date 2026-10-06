"""Run the additional deep-learning benchmarks requested in the review in the walk-forward protocol:
DNN and DNN ensemble of the open-access benchmark (Lago et al., 2021), N-BEATSx and the Temporal
Fusion Transformer (neuralforecast). Hyper-parameters are read from configs/tuned_<model>_<market>.json
(produced by scripts/tune_nn.py on data before the first evaluated origin) when available.

  python scripts/run_deep.py --market PL --model dnn --seed 1
  python scripts/run_deep.py --market PL --model dnn_ens --seed 1
  python scripts/run_deep.py --market PL --model nbeatsx --seed 1
  python scripts/run_deep.py --market PL --model tft --seed 1
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")   # before numpy/sklearn: avoids CPU oversubscription; results unchanged
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less GPU memory fragmentation
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
torch.set_num_threads(int(os.environ.get("EPF_TORCH_THREADS", "2")))   # do not starve the CPU queues

from epfstab.models.dnn import DNN, DNNEnsemble  # noqa: E402
from epfstab.models.nf import NF  # noqa: E402
from epfstab.runner import run_direct  # noqa: E402


def tuned(model: str, market: str):
    for m in (market, "PL"):
        f = ROOT / "configs" / f"tuned_{model}_{m}.json"
        if f.exists():
            return json.loads(f.read_text()), f.name
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True)
    ap.add_argument("--model", required=True, choices=["dnn", "dnn_ens", "nbeatsx", "tft"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--window", type=int, default=730)
    ap.add_argument("--cadence", type=int, default=7)
    ap.add_argument("--run", default=None)
    ap.add_argument("--origin-start", default=None)
    ap.add_argument("--origin-end", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg, src = tuned("dnn" if a.model.startswith("dnn") else a.model, a.market)
    suffix = f"_s{a.seed}"
    kw = {"seed": a.seed, "train_days": a.window, "suffix": suffix, "hp_source": src}
    if a.model == "dnn":
        cls, kw["hp"] = DNN, (cfg["top"][0] if cfg else None)
    elif a.model == "dnn_ens":
        cls, kw["hps"] = DNNEnsemble, (cfg["top"][:4] if cfg else [None] * 4)
    else:
        cls, kw["model"], kw["hp"] = NF, a.model, (cfg["best"] if cfg else None)
    run = a.run or f"w{a.window}_c{a.cadence}"
    run_direct(a.market, cls, a.window, a.cadence, run, kw, a.origin_start, a.origin_end, a.smoke)


if __name__ == "__main__":
    main()
