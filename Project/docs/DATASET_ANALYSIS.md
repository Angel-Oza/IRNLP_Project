# Dataset Analysis & Strategy Specification (Revised)

**Project Title:** Explainable Multi-Source Soil Health Intelligence System Using Machine Learning, RAG and Gated Fusion  
**Author:** Angel Oza  
**Status:** Validated Architecture & Integration Strategy  

---

## 1. Candidate Dataset Inspection & Assessment

To satisfy the requirements of multi-source intelligence (real-time IoT telemetry, chemical soil test profiling, and crop agronomic requirements), three public datasets are integrated into the architecture. Rather than artificially forcing disparate data into an unscientific single table, each dataset serves a dedicated, scientifically defensible role.

---

### Dataset 1: Soil Testing & Nutrient Certificate Dataset (IVRI KVK Bareilly)

- **Source / Origin:** Krishi Vigyan Kendra (KVK), Indian Veterinary Research Institute (IVRI), Indian Council of Agricultural Research (ICAR), Bareilly, India.
- **Physical Verification:** Accompanied by scanned official laboratory soil testing certificates (`soilTestDataset-WAIJfM.pdf`).
- **Format:** CSV (`AgriNet_Dataset-G7TD2K.csv`) & Scanned Laboratory Certificates.
- **Number of Records:** 257 certified soil samples.
- **Geographic Context:** Indo-Gangetic Alluvial Plains (Bareilly district, Uttar Pradesh, India).
- **Features / Measured Variables:**
  - `pH` (dimensionless, 1:2.5 soil-water suspension, Range: 6.76 – 7.45)
  - `EC` (Electrical Conductivity in dS/m, Range: 0.08 – 0.27 dS/m)
  - `OC` (Organic Carbon in %, Range: 0.11 – 0.41%)
  - `N` (Available Nitrogen in kg/ha, Range: 61 – 179 kg/ha)
  - `P` (Available Phosphorus in kg/ha, Range: 2.11 – 18.71 kg/ha)
  - `K` (Available Potassium in kg/ha, Range: 76 – 269 kg/ha)
  - `S` (Available Sulphur in ppm, Range: 11.78 – 15.77 ppm)
  - `Zn` (Available Zinc in ppm, Range: 0.64 – 1.43 ppm)
  - `B` (Available Boron in ppm, Range: 0.43 – 1.25 ppm)
  - `Fe` (Available Iron in ppm, Range: 6.37 – 9.33 ppm)
  - `Mn` (Available Manganese in ppm, Range: 1.73 – 2.71 ppm)
  - `Cu` (Available Copper in ppm, Range: 0.18 – 0.89 ppm)
  - `Area in Acre`, `Bigha` (Farmland plot area)
- **Data Quality & Integrity:** 100% complete records, zero missing values.
- **Intended Architectural Role:** Ground truth chemical calibration for the Rule-Based **Soil Health Index (SHI)**, nutrient deficiency rule engine, and agronomic thresholds.

---

### Dataset 2: 10-Year In-Situ Environmental & Soil Sensor Telemetry (Berambadi / Bechanahalli Field Station)

- **Source / Origin:** Long-term hydrological/agricultural field observatory at Berambadi / Bechanahalli, Southern India (AMBHAS Research Basin).
- **Time Horizon:** January 1, 2016 to June 30, 2025 (10 full years).
- **Temporal Resolution:** 15-minute continuous sampling interval (>300,000 temporal records).
- **Format:** Chronologically sorted yearly CSV files (`2016_V1.csv` through `2025_V1.csv`) with metadata logs (`events_metadata.csv`, `swap_log.csv`).
- **Features / Measured Variables:**
  - `timestamp` (IST, YYYY-MM-DD HH:MM:SS)
  - `Temp_5cm`, `Temp_15cm`, `Temp_50cm` (Soil Temperature at 5cm, 15cm, 50cm in °C)
  - `SM_5cm`, `SM_15cm`, `SM_50cm` (Soil Moisture / Volumetric Water Content in %)
  - `EC_5cm`, `EC_50cm` (Bulk Soil Electrical Conductivity in dS/m)
  - `RDP_5cm`, `RDP_15cm`, `RDP_50cm` (Relative Dielectric Permittivity)
  - `Precipitation` (Rainfall per 15-min interval in mm)
- **Quality Control Applied in Source:** Strict outlier bounds (T in [0, 60]°C), seasonal dry/monsoon verification, logged depth swaps. Hardware missing records preserved as `NaN`.
- **Intended Architectural Role:** Powers the **Software Sensor Stream Simulator** (`iot/simulator.py`), the **Sensor Noise & Drift Injection Engine** (`iot/noise_simulator.py`), and the **Dynamic Sensor Reliability Weighting** calculation ($r \in [0, 1]$).

---

### Dataset 3: Crop Suitability & Soil Requirement Benchmark Dataset

- **Source / Origin:** Precision Agriculture Benchmark (derived from ICAR and FAO crop-soil ecological growth envelopes).
- **Format:** CSV (`Crop_recommendation.csv`).
- **Number of Records:** 2,200 observations across 22 distinct crop categories (100 samples per class).
- **Features:**
  - `N` (Available Nitrogen level / index, Range: 0 – 140)
  - `P` (Available Phosphorus level / index, Range: 5 – 145)
  - `K` (Available Potassium level / index, Range: 5 – 205)
  - `temperature` (°C, Range: 8.83 – 43.68)
  - `humidity` (%, Range: 14.26 – 99.98)
  - `ph` (pH value, Range: 3.50 – 9.94)
  - `rainfall` (Precipitation in mm, Range: 20.21 – 298.56)
  - `label` (Crop suitability ground truth class)
- **Intended Architectural Role:** Benchmark dataset for **Machine Learning Classifiers** (Logistic Regression, Random Forest, XGBoost, MLP), training the **Gated Fusion Network**, and calculating **SHAP Feature Attribution**.

---

## 2. Information Flow Across Logically Separated Datasets

```
+-----------------------------------------------------------------------------------------------+
| LOGICALLY SEPARATED PIPELINE WORKFLOW                                                          |
|                                                                                               |
|  [Dataset 2: 10-Yr Berambadi Telemetry]                                                       |
|       |                                                                                       |
|       v                                                                                       |
|  Software Sensor Stream Simulator (Replays 15-min SM, Temp, EC, Rain)                         |
|       |                                                                                       |
|       +---> Injected Noise / Drift Simulation (0%, 5%, 10%, 20%)                              |
|       |                                                                                       |
|       v                                                                                       |
|  Telemetry Stream + Sensor Reliability Score r in [0, 1]                                      |
|       |                                                                                       |
|  [Dataset 1: IVRI Lab Soil Chemistry]                                                         |
|       |                                                                                       |
|       v                                                                                       |
|  Soil Chemistry Profile (N, P, K, pH, EC, OC, Micronutrients)                                 |
|       |                                                                                       |
|       +------------------------------------+------------------------------------+             |
|       |                                    |                                    |             |
|       v                                    v                                    v             |
|  Soil Health Index (SHI)              RAG Contextual Retrieval             Dataset C Features |
|  (ICAR/FAO Rule-Based Index 0-100)    (Agronomic Science Corpus)           (ML & Gated Fusion)|
|       |                                    |                                    |             |
|       v                                    v                                    v             |
|  Dashboard Health KPI & Deficiency    Retrieved Passages & Embeddings      PyTorch Gated Net  |
|  (Separate from Supervised ML)        z_text in R^384                      Prediction & SHAP  |
+-----------------------------------------------------------------------------------------------+
```
