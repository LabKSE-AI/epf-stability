"""Progress and stall check of all runs (reads results/*/*/heartbeat_*.json).

A run is reported as STALLED when its heartbeat has not been updated for longer than
max(--min-stall minutes, 5 x the duration of its last block) while its process is still alive,
and as DIED when the process is gone before all origins were done.

  python scripts/watchdog.py            # table of all runs
  python scripts/watchdog.py --alerts   # print only problems and newly finished runs (for loops)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epfstab.paths import RESULTS  # noqa: E402


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alerts", action="store_true")
    ap.add_argument("--min-stall", type=float, default=20.0, help="minutes")
    ap.add_argument("--state", default=str(RESULTS / ".watchdog_state.json"))
    a = ap.parse_args()
    state_f = Path(a.state)
    state = json.loads(state_f.read_text()) if state_f.exists() else {}
    now = time.time()
    rows = []
    for hb in sorted(RESULTS.glob("*/*/heartbeat_*.json")):
        try:
            h = json.loads(hb.read_text())
        except Exception:
            continue
        key = f"{h['market']}/{h['run']}/{h['model']}"
        age = (now - h["updated"]) / 60
        finished = h["done"] >= h["total"]
        is_alive = alive(h["pid"])
        limit = max(a.min_stall, 5 * h.get("last_block_sec", 0) / 60)
        status = "done" if finished else ("STALLED" if (is_alive and age > limit) else ("DIED" if not is_alive else "running"))
        rows.append((key, status, h["done"], h["total"], age, h.get("eta_min", 0), h.get("last_origin", "")))
        if a.alerts:
            prev = state.get(key)
            if status in ("STALLED", "DIED") or (status == "done" and prev != "done"):
                print(f"[{status}] {key}: {h['done']}/{h['total']} last update {age:.0f} min ago, last origin {h.get('last_origin', '')}", flush=True)
        state[key] = status
    state_f.write_text(json.dumps(state))
    if not a.alerts:
        print(f"{'run':45} {'status':8} {'done':>11} {'upd[min]':>8} {'ETA[min]':>8}  last origin")
        for k, s, d, t, age, eta, lo in rows:
            print(f"{k:45} {s:8} {d:>5}/{t:<5} {age:8.1f} {eta:8.0f}  {lo}")


if __name__ == "__main__":
    main()
