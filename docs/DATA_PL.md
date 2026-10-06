# Polish market inputs — how the hourly master file was assembled

All series were merged **by UTC timestamp** (never by row position). Missing values remain missing.

## TGE Fixing I price (`CenaI`) and volume (`VolumenI`), 2016-01-01 … 2025-12-31
| period | source |
|---|---|
| 2016-01-01 … 2025-09-30 | Instrat energy data (export of the TGE Fixing I series) |
| 2025-10-01 … 2025-12-31 | TGE website (https://tge.pl/energia-elektryczna-rdn), pages saved daily while they were public |

Documented corrections of the Instrat series (each verified against an independent source):
1. timestamps until 2020-05-07 were one hour early — shifted by +1 h (after the shift 99.9 % of hours
   agree with the SDAC coupling price; the ENTSO-E A44 price in PLN for 2017–11.2019 agrees with TGE
   Fixing II in 99.7 % of hours);
2. days 2020-05-08 and 2020-05-09 (copies of 2020-05-22) and 2022-11-14 (copy of 2022-11-15) — removed
   (missing);
3. the repeated 02:00 hour of the autumn clock change appears once — assigned to CEST or CET by
   comparison with Fixing II and SDAC; the other 02:00 hour stays missing.

Hours without a price in any source remain missing (2016–2025: 91 hours in UTC).

## Generation by technology and system load
Instrat energy data (https://energy-api.instrat.pl/api; originally ENTSO-E A75 / PSE), 15-minute data
averaged to full hours only when complete. Generation 2020-05-10 … 2020-06-09: timestamps two hours late
— shifted by −2 h (verified against ENTSO-E). Series whose history starts later than 2016 (solar before
04.2020, "other", "other renewable") are not used as model inputs.

## Day-ahead load forecast
Official day-ahead total-load forecast of PSE from the ENTSO-E Transparency Platform (A65/A01), complete
for 2016–2025. (The load-forecast series distributed by Instrat differs from the official D-1 forecast
in about 06.2024–2025 — it appears to be a later revision — and is therefore not used.)

## Fuel and carbon prices, exchange rates
EU ETS allowance (ICAP Allowance Price Explorer, daily), European natural gas and South-African coal
benchmarks (World Bank "Pink Sheet", monthly), NBP mid-rates (daily); PLN conversions with the latest
published rate. Availability rules: see README ("Missing data — no imputation").
