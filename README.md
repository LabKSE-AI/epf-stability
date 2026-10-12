# epf-stability — temporal (in)stability of day-ahead electricity-price forecasting models

Code accompanying the article *“Temporal (in)stability of day-ahead electricity-price-forecasting models:
detection, attribution and forecast combination on the Polish market, with validation on two European markets”*
(D. Wotzka, P. Kurtasz, O. Zmarzły). Maintained by **LabKSE-AI**, the research group of the authors.

The repository contains **code only**. No data are distributed: the data sources require registration,
are licensed, or prohibit redistribution (see [Data](#data)). The scripts download the freely accessible
series with your own credentials and document how every input was assembled.

## What is here

| path | content |
|---|---|
| `epfstab/protocol.py` | leakage-free rolling walk-forward protocol (daily origins at 00:00 local time, 48-hour horizon, 48-hour embargo, recalibration every 7 days) |
| `epfstab/models/tabular.py` | seasonal naïve, LEAR (LASSO-estimated autoregression), gradient-boosted trees |
| `epfstab/models/rnn.py` | LSTM, BiLSTM, GRU (sequence-to-vector, direct 48-hour output) |
| `epfstab/models/dnn.py` | DNN and DNN ensemble of the open-access benchmark of Lago et al. (2021), re-implemented in PyTorch |
| `epfstab/models/nf.py` | N-BEATSx and Temporal Fusion Transformer (via `neuralforecast`) |
| `epfstab/runner.py` | resumable walk-forward runner (per-origin errors, 48-hour forecast vectors, heartbeat) |
| `epfstab/evaluate.py` | combinations, Diebold–Mariano tests with HAC variance, Holm correction, stationary-bootstrap CIs, Model Confidence Set |
| `scripts/` | data download and input building, runs, hyper-parameter search, analyses and figures of the paper, progress monitor (`watchdog.py`), full analysis pipeline (`run_all_analyses.sh`) |

## Missing data — no imputation

Missing values are never interpolated or imputed. An origin whose inputs or 48-hour target contain a
missing value is skipped (the number of skipped origins is recorded in `meta_<model>.json`), and
training samples containing missing values are dropped. Daily exogenous series (EU ETS allowance
price, NBP exchange rates) enter with the value of the last day published *before* the origin day
(at most 21 days old); a monthly average of month *M* (World Bank) is used only from the 8th day of
month *M+1*, after its publication.

## Data

| series | source | access |
|---|---|---|
| TGE Fixing I price and volume (PL), 2016–2025 | Towarowa Giełda Energii — https://tge.pl/energia-elektryczna-rdn (last ~2 months; the full official archive is sold by TGE); historical series via Instrat energy data, https://energy-api.instrat.pl/api | TGE: redistribution prohibited; Instrat: see its terms |
| generation by technology, system load (PL) | Instrat energy data (https://energy-api.instrat.pl/api), originally ENTSO-E / PSE | see Instrat terms |
| day-ahead load forecasts (PL, DE-AT-LU, DE-LU, SE4), day-ahead prices (DE, SE4) | ENTSO-E Transparency Platform, https://transparency.entsoe.eu (documents A65/A01, A44) | free account and security token required |
| EU ETS allowance price (daily) | ICAP Allowance Price Explorer, https://allowancepriceexplorer.icapcarbonaction.com | free |
| European gas and South-African coal benchmarks (monthly) | World Bank Commodity Price Data (“Pink Sheet”), https://www.worldbank.org/en/research/commodity-markets | free |
| EUR/PLN, USD/PLN mid-rates | National Bank of Poland Web API, https://api.nbp.pl | free |

Inputs are expected in `data/inputs/<MARKET>.parquet` (schema in `epfstab/data.py`); set `EPFSTAB_DATA`
to keep data elsewhere.

* **DE and SE4**: `scripts/fetch_entsoe.py` (needs `ENTSOE_TOKEN` or `~/.entsoe_token`) downloads the
  load forecasts; prices are the hourly ENTSO-E A44 day-ahead prices (zone DE-AT-LU until 30.09.2018,
  DE-LU afterwards). `scripts/build_inputs_entsoe.py` builds the inputs.
* **PL**: `scripts/build_inputs_pl.py` builds the inputs from an hourly master file with UTC timestamps
  (columns `start_utc, CenaI, VolumenI, load_forecast_entsoe` and generation by technology) and the
  exogenous series. The master file was assembled by merging the sources above **by UTC timestamp**;
  see `docs/DATA_PL.md` for the exact sources per period and the documented corrections of the raw
  series.

## Running

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python scripts/run.py --market PL --model lear --window 730               # one model, one market
python scripts/run_nn.py --market PL --model lstm --config tuned --seed 1  # recurrent networks (GPU)
python scripts/tune_nn.py --market PL --model dnn --trials 200             # hyper-parameters on 2016–2017 only
python scripts/run_deep.py --market PL --model nbeatsx --seed 1            # DNN, DNN-Ens, N-BEATSx, TFT
python scripts/watchdog.py                                                  # progress / stall check
```
Outputs: `results/<MARKET>/w<window>_c<cadence>/` (per-origin errors, forecast vectors, metadata).

Sensitivity runs of the paper (training windows `W` ∈ {90, 120, 180, 365, 548, 730, 1095, 1460} days):

```bash
python scripts/run.py --market PL --model lear --window W --fuels co2_eua_eur gas_eu_usd_mmbtu coal_za_usd_mt --suffix _fuel     # gas, coal, CO2 in their original currencies
python scripts/run.py --market PL --model lear --window W --fuels co2_eua_pln_t gas_eu_pln_mwh coal_za_pln_t --suffix _fuelpln  # the same in PLN
python scripts/run.py --market PL --model lear --window W --fuels co2_eua_eur --suffix _co2        # daily carbon price as the only fundamental input (EUR)
python scripts/run.py --market PL --model lear --window W --fuels co2_eua_pln_t --suffix _co2pln   # the same in PLN
python scripts/run_deep.py --market PL --model dnn --seed S --window W                              # DNN at every window (seeds S = 1, 2, 3)
python scripts/run.py --market PL --model lear --window 730 --cadence C                            # recalibration every C = 1, 14, 30 days
```

### Analyses of the paper (after the runs)

```bash
python scripts/evaluate_main.py [--block d1]       # model tables, DM tests (HAC + Holm), combinations, MCS, seed spread
python scripts/stability_tests.py --market PL --scaled   # error-growth decomposition, fluctuation and CPA tests, LS breaks
python scripts/pelt_quarterly.py                   # PELT on the quarterly error (penalty sweep)
python scripts/attribution_diag.py --market PL --exo <fuels file> --master <PL master>   # quarterly attribution, unit-root and cointegration tests
python scripts/evaluate_sensitivity.py             # training window, fundamentals as inputs, recalibration cadence
python scripts/evaluate_extra.py                   # second benchmark, negative prices, naïve scaled-error shift
python scripts/probabilistic.py --market PL        # QRA quantiles: pinball loss, coverage, width
python scripts/economic.py --market PL             # storage arbitrage (1 MW / 2 MWh, efficiency 0.85)
python scripts/figures.py --lang en                # figures (styles and texts in configs/figures.yaml)
```
Tables are written to `results/tables/`, figures to `results/figures/<lang>/`.

## Reproducibility of the neural networks

Neural networks are trained on GPUs (PyTorch/cuDNN), which is not bitwise reproducible across hardware: repeating a seed on
another machine changes single-seed forecasts about as much as changing the seed. The paper therefore reports seed ensembles
(average of the seeds' forecasts) and the seed-to-seed spread. In the runs of the paper, seeds 2, 4 and 5 of the untuned
GRU and seed 5 of the untuned LSTM and BiLSTM were computed on the CPU, and TFT seed 3 on a second GPU.

All steps above are chained in `scripts/run_all_analyses.sh` (set `EXO` to the directory with the fuel, carbon and
exchange-rate files and `MASTER` to the Polish hourly master file).

## How the code was written

The methods, models and analyses were designed by the authors, who specified every component, reviewed the code and validated
the results. The implementation was written with the assistance of AI coding assistants (Claude models by Anthropic) working
under the authors' direction.

## How to cite

Please cite the article and this software; citation metadata are in `CITATION.cff`. Version 1.0.0 is archived in Zenodo:
https://doi.org/10.5281/zenodo.23321571 (all versions: https://doi.org/10.5281/zenodo.23321570).

## Licence

Code: MIT licence (see `LICENSE`). The licence does not cover any data.
