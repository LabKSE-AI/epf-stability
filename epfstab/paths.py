"""Locations of data and results. Data live outside version control (see README, "Data");
set EPFSTAB_DATA to use another directory."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("EPFSTAB_DATA", ROOT / "data"))
INPUTS = DATA / "inputs"
RAW = DATA / "raw"
RESULTS = Path(os.environ.get("EPFSTAB_RESULTS", ROOT / "results"))
