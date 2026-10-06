"""Run a neural sequence model (GPU) in the rolling walk-forward protocol.

  python scripts/run_nn.py --market PL --model lstm --config untuned --seed 1      # paper: LSTM/BiLSTM/GRU with fuels
  python scripts/run_nn.py --market PL --model lstm --config tuned --seed 1        # paper: tuned LSTM, no fuels
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")   # before numpy/sklearn: avoids CPU oversubscription; results unchanged
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # less GPU memory fragmentation
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402
torch.set_num_threads(int(os.environ.get("EPF_TORCH_THREADS", "2")))   # do not starve the CPU queues

from epfstab.models.rnn import PL_RNN_INPUTS, RNN, TUNED_LSTM, UNTUNED  # noqa: E402
from epfstab.runner import run_direct  # noqa: E402

FUELS_NAT = ["co2_eua_eur", "gas_eu_usd_mmbtu", "coal_za_usd_mt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True)
    ap.add_argument("--model", required=True, choices=["lstm", "bilstm", "gru"])
    ap.add_argument("--config", choices=["untuned", "tuned"], default="untuned")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--window", type=int, default=730)
    ap.add_argument("--cadence", type=int, default=7)
    ap.add_argument("--no-fuels", action="store_true", help="untuned trio: drop fuel inputs")
    ap.add_argument("--run", default=None)
    ap.add_argument("--origin-start", default=None)
    ap.add_argument("--origin-end", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    inputs = list(PL_RNN_INPUTS) if a.market.startswith("PL") else []
    if a.config == "untuned" and not a.no_fuels and a.market.startswith("PL"):
        inputs += FUELS_NAT
    hp = UNTUNED if a.config == "untuned" else TUNED_LSTM
    suffix = f"_{a.config}_s{a.seed}"
    kw = {"kind": a.model, "hp": hp, "inputs": inputs, "seed": a.seed, "train_days": a.window, "suffix": suffix}
    run = a.run or f"w{a.window}_c{a.cadence}"
    run_direct(a.market, RNN, a.window, a.cadence, run, kw, a.origin_start, a.origin_end, a.smoke)


if __name__ == "__main__":
    main()
