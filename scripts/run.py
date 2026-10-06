"""Run one model on one market in the rolling walk-forward protocol.

Examples:
  python scripts/run.py --market PL --model lear --window 730
  python scripts/run.py --market PL --model gbt --window 730 --fuels co2_eua_pln_t gas_eu_pln_mwh coal_za_pln_t --jobs 12
  python scripts/run.py --market DE --model lear --window 730 --loadfc available --suffix _avail
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")   # before numpy/sklearn: 4x faster parallel tree fitting; results unchanged
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epfstab.models.tabular import MODELS  # noqa: E402
from epfstab.runner import run_direct  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--window", type=int, default=730, help="training window [days]")
    ap.add_argument("--cadence", type=int, default=7, help="recalibration every N days")
    ap.add_argument("--fuels", nargs="*", default=[], help="fuel/CO2 input columns (at the origin)")
    ap.add_argument("--loadfc", choices=["full", "available"], default="full")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--suffix", default="", help="appended to the model name in output files")
    ap.add_argument("--run", default=None, help="run directory name (default w<window>_c<cadence>)")
    ap.add_argument("--origin-start", default=None)
    ap.add_argument("--origin-end", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    kw = {"fuels": a.fuels, "loadfc": a.loadfc, "jobs": a.jobs, "suffix": a.suffix}
    if a.model == "gbt":
        kw["seed"] = a.seed
    run = a.run or f"w{a.window}_c{a.cadence}"
    run_direct(a.market, MODELS[a.model], a.window, a.cadence, run, kw, a.origin_start, a.origin_end, a.smoke)


if __name__ == "__main__":
    main()
