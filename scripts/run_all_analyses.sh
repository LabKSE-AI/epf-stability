#!/usr/bin/env bash
# Final analysis pipeline of the paper — run after all forecasting runs have finished.
#   EXO=<exogenous dir> MASTER=<PL hourly master> bash scripts/run_all_analyses.sh
set -eu
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
EXO=${EXO:?set EXO to the directory with the fuel, carbon and exchange-rate files}
MASTER=${MASTER:-}
echo "=== ANALYSES START $(date)"
$PY scripts/evaluate_main.py
$PY scripts/evaluate_main.py --block d1
for M in PL DE SE4; do
  G=gbt_nofuel; [ "$M" = PL ] && G=gbt
  # main stability analyses on the common origins of the naive benchmark, LEAR and GBT
  $PY scripts/stability_tests.py --market $M --models naive lear $G --pairs lear:$G lear:ens_inv30 ens_avg:ens_inv30 lear:naive --scaled
  # decomposition of error growth for every model family available in this market
  ALL=""
  for f in naive lear $G dnn dnn_ens nbeatsx tft lstm_untuned bilstm_untuned gru_untuned lstm_tuned; do
    # a single model with forecasts, or a family with at least one finished seed
    if [ -d results/$M/w730_c7/preds_$f ] || grep -lq '"finished": true' results/$M/w730_c7/meta_${f}_s[0-9]*.json 2>/dev/null; then ALL="$ALL $f"; fi
  done
  $PY scripts/stability_tests.py --market $M --models $ALL --pairs lear:$G --scaled --tag _allmodels
  if [ "$M" = PL ]; then
    $PY scripts/attribution_diag.py --market $M --exo "$EXO" --master "${MASTER:?set MASTER for PL}"
  else
    $PY scripts/attribution_diag.py --market $M --exo "$EXO"
  fi
  $PY scripts/probabilistic.py --market $M --models naive lear $G ens_inv30
  $PY scripts/economic.py --market $M --models naive lear $G ens_inv30
done
$PY scripts/pelt_quarterly.py
$PY scripts/evaluate_sensitivity.py
$PY scripts/evaluate_extra.py
for L in en pl; do $PY scripts/figures.py --lang $L; done
echo "=== ANALYSES DONE $(date)"
