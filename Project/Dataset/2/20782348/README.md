# Soil Moisture & Climate Dataset (2016–2025)
**Berambadi / Bechanahalli field station, southern India — depth-corrected release**

## Overview
This dataset provides **15-minute interval environmental measurements** collected between **January 2016 and June 2025** at the Berambadi/Bechanahalli field station in southern India.

It contains:
- Soil moisture (SM)
- Electrical conductivity (EC, 2016–2021) / Relative dielectric permittivity (RDP, 2021–2025)
- Soil temperature (Temp) at multiple depths (5 cm, 15 cm, 50 cm)
- Precipitation

The dataset is distributed as **yearly CSV files** (`2016_V1.csv`, `2017_V1.csv`, …, `2025_V1.csv`), accompanied by a rainfall **event catalogue** (`events_metadata.csv`) and a **depth-assignment log** (`swap_log.csv`).

---

## Data Quality Control

This dataset has undergone the following quality-control procedures:
1. **Outlier removal**: Physically impossible values removed (e.g., T < 0 °C or T > 60 °C).
2. **Sensor validation**: Sensors verified for realistic diurnal variation during the dry season (Jan–Mar).
3. **Completeness check**: Data gaps identified and preserved as NaN.
4. **Duplicate removal**: Duplicated timestamps removed (96 leap-day duplicate records in 2016); every annual file now has unique, chronologically sorted timestamps.
5. **Depth-assignment validation**: The 5 cm / 50 cm depth labels of the soil-moisture and electrical-conductivity / dielectric-permittivity channels were validated against physical expectations (surface should be more dynamic, reach higher peak moisture, and breach saturation more often than the deep layer). Where the channels were found to be transposed, the 5 cm and 50 cm columns were exchanged. **Soil temperature was unaffected and is left unchanged.** Per-year decisions and the diagnostics behind them are recorded in `swap_log.csv`.
6. **Error code handling**: Logger error codes (`E36`, `E6`) replaced with NaN.

**No interpolation or gap-filling was performed.** Missing values remain as NaN.

---

## Variables & Units

| Column        | Description                                           | Unit                     | Availability |
|---------------|-------------------------------------------------------|--------------------------|-------|
| timestamp     | Date & time of measurement (IST, UTC+05:30)           | YYYY-MM-DD HH:MM:SS      | 15-min intervals |
| Temp_5cm      | Soil temperature at 5 cm depth                        | °C                       | 2016–2025 |
| SM_5cm        | Soil moisture at 5 cm depth                           | % (vol. water content)   | 2016–2025 |
| EC_5cm        | Electrical conductivity at 5 cm depth                 | dS/m (deciSiemens/m)     | 2016–2021 |
| RDP_5cm       | Relative dielectric permittivity at 5 cm              | dimensionless            | 2022–2025 |
| Temp_15cm     | Soil temperature at 15 cm depth                       | °C                       | 2021–2025 |
| SM_15cm       | Soil moisture at 15 cm depth                          | %                        | 2021–2025 |
| EC_15cm       | Electrical conductivity at 15 cm depth                | dS/m                     | Column present in 2021 only; no valid data |
| RDP_15cm      | Relative dielectric permittivity at 15 cm depth       | dimensionless            | 2021–2025 |
| Temp_50cm     | Soil temperature at 50 cm depth                       | °C                       | 2016–2025 |
| SM_50cm       | Soil moisture at 50 cm depth                          | %                        | 2016–2025 |
| EC_50cm       | Electrical conductivity at 50 cm depth                | dS/m                     | 2016–2021 |
| RDP_50cm      | Relative dielectric permittivity at 50 cm depth       | dimensionless            | 2021–2025 |
| Precipitation | Rainfall                                              | mm per 15 min            | 2016–2025 |

> Note: there is **no `RDP_5cm` column in 2021** — during the 2021 EC→RDP transition the 5 cm location logged electrical conductivity (`EC_5cm`) but not relative dielectric permittivity. `RDP_5cm` is therefore available from 2022 onward.

---

## Yearly Data Summary

| Year | Rows   | Coverage        | Columns Present | Notes |
|------|--------|-----------------|-----------------|-------|
| 2016 | 35,041 | Jan 1 – Dec 31  | SM, EC, Temp, Precipitation | Full year (96 duplicate timestamps removed) |
| 2017 | 35,027 | Jan 1 – Dec 31  | SM, EC, Temp, Precipitation | Full year |
| 2018 | 33,309 | Jan 1 – Dec 31  | SM, EC, Temp, Precipitation | Full year |
| 2019 | 22,620 | Jan 1 – Dec 31  | SM, EC, Temp, Precipitation | Partial coverage |
| 2020 | 7,733  | Oct 12 – Dec 31 | SM, EC, Temp, Precipitation | Only ~3 months |
| 2021 | 32,392 | Jan 1 – Dec 31  | SM, EC, RDP (5 cm: EC only), Temp, Precipitation | EC→RDP transition year |
| 2022 | 33,064 | Jan 1 – Dec 31  | SM, RDP, Temp, Precipitation | Full year |
| 2023 | 34,480 | Jan 1 – Dec 31  | SM, RDP, Temp, Precipitation | Full year |
| 2024 | 35,080 | Jan 1 – Dec 31  | SM, RDP, Temp, Precipitation | Full year |
| 2025 | 17,316 | Jan 1 – Jun 30  | SM, RDP, Temp, Precipitation | Partial year (Jan–Jun) |

---

## Data Completeness (% Non-Missing Values)

A dash (—) indicates the column is not present that year.

| Year | SM_5cm | EC_5cm | RDP_5cm | Temp_5cm | SM_15cm | EC_15cm | RDP_15cm | Temp_15cm | SM_50cm | EC_50cm | RDP_50cm | Temp_50cm | Precip |
|------|--------|--------|---------|----------|---------|---------|----------|-----------|---------|---------|----------|-----------|--------|
| 2016 | 100 | 100 | — | 100 | — | — | — | — | 100 | 100 | — | 100 | 100 |
| 2017 | 100 | 100 | — | 100 | — | — | — | — | 100 | 100 | — | 100 | 100 |
| 2018 | 100 | 100 | — | 76  | — | — | — | — | 76  | 76  | — | 100 | 100 |
| 2019 | 98  | 79  | — | 100 | — | — | — | — | 100 | 100 | — | 79  | 100 |
| 2020 | 100 | 100 | — | 100 | — | — | — | — | 100 | 100 | — | 100 | 100 |
| 2021 | 100 | 51  | — | 55  | 45 | 0 | 45 | 45 | 33  | 32  | 45 | 100 | 100 |
| 2022 | 100 | —   | 100 | 100 | 100 | — | 100 | 100 | 100 | —   | 100 | 100 | 100 |
| 2023 | 100 | —   | 100 | 100 | 100 | — | 100 | 100 | 100 | —   | 100 | 100 | 92  |
| 2024 | 100 | —   | 100 | 100 | 100 | — | 100 | 100 | 100 | —   | 100 | 100 | 100 |
| 2025 | 100 | —   | 100 | 100 | 100 | — | 100 | 100 | 100 | —   | 100 | 100 | 100 |

---

## Schema Changes

**2016–2020:** EC-based sensors
- Columns: `timestamp`, `Temp_5cm`, `SM_5cm`, `EC_5cm`, `Temp_50cm`, `SM_50cm`, `EC_50cm`, `Precipitation`

**2021:** Transition year (contains both EC and RDP; 15 cm depth introduced)
- Columns: `timestamp`, `Temp_5cm`, `SM_5cm`, `EC_5cm`, `Temp_15cm`, `SM_15cm`, `EC_15cm`, `RDP_15cm`, `Temp_50cm`, `SM_50cm`, `EC_50cm`, `RDP_50cm`, `Precipitation`
- Note: no `RDP_5cm` this year; `EC_15cm` is present as a column but contains no valid data.

**2022–2025:** RDP-based sensors with 15 cm depth
- Columns: `timestamp`, `Temp_5cm`, `SM_5cm`, `RDP_5cm`, `Temp_15cm`, `SM_15cm`, `RDP_15cm`, `Temp_50cm`, `SM_50cm`, `RDP_50cm`, `Precipitation`

---

## Accompanying Files

- **`swap_log.csv`** — per-year depth-assignment decisions (SWAP/keep), the diagnostic votes behind each decision, and the number of duplicate timestamps removed.
- **`events_metadata.csv`** — rainfall event catalogue (944 events) derived from the precipitation record using a 6-hour dry-gap separation and a 0.5 mm minimum-depth threshold. Each event carries onset time, total depth, duration, and antecedent precipitation index values (`API_24h`, `API_48h`, `API_72h`).

---

## Data Handling Notes

1. **Missing values** — Logger error codes (`E36`, `E6`) have been replaced with `NaN`. Users should verify NaN handling:
   ```python
   df = df.replace("NaN", np.nan)
   ```
2. **Negative values** — Negative **RDP** and **EC** values in 2019 and 2021 have been corrected.
3. **File encoding** — CSV files are UTF-8 encoded; values are aligned at 15-minute intervals.
4. **Time zone** — All timestamps are Indian Standard Time (IST, UTC+05:30).

---

## Usage Recommendations

### Loading Data
```python
import pandas as pd
import numpy as np

df = pd.read_csv('2017_V1.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.replace("NaN", np.nan)   # handle any remaining string NaNs
```

### Handling Schema Changes
For analyses spanning 2016–2025:
- Use **EC columns** for 2016–2021.
- Use **RDP columns** for 2022–2025 (and `RDP_15cm`/`RDP_50cm` in 2021).
- 2021 is the transition year (EC at 5/50 cm; RDP at 15/50 cm only).

### Working with Multiple Years
```python
import pandas as pd

dfs = []
for year in range(2016, 2026):
    try:
        df = pd.read_csv(f'{year}_V1.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        dfs.append(df)
    except FileNotFoundError:
        print(f"Warning: {year}_V1.csv not found")

all_data = pd.concat(dfs, ignore_index=True)
```

---

## File Format

- **Format**: CSV (Comma-Separated Values)
- **Encoding**: UTF-8
- **Naming**: `YYYY_V1.csv` (e.g., `2017_V1.csv`)
- **Time resolution**: 15 minutes
- **Missing values**: Represented as empty cells or `NaN`

---

## Citation

> Rao, S., Upadhya, A., Goswami, S., Shaju, A. P., Kandala, R., Upadhyaya, D.,
> Gupta, V., Ruiz, L., and Muddu, S. (2026).
> High-frequency soil hydrothermal observations from a semi-arid monsoon
> catchment in southern India (2016–2025) [Data set]. Zenodo.
> https://doi.org/10.5281/zenodo.18409640

---

## Contact

Prof. Sekhar Muddu (sekhar.muddu@gmail.com)
**Field Site:** Berambadi/Bechanahalli, southern India

---

## Acknowledgments

The authors acknowledge funding support from the Rejuvenating Watersheds for Agricultural Resilience through Innovative Development (REWARD) project (Project Number: CP/OTHR-22-0037.13) of the Watershed Development Department, Government of Karnataka, and the Department of Civil Engineering, Indian Institute of Science. We thank the field teams who assisted with data collection in the Berambadi watershed.

---

**Last Updated:** June 2026
