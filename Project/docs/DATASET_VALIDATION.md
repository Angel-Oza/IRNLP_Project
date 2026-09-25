# Dataset Validation & Statistical Integrity Report

**Project Title:** Explainable Multi-Source Soil Health Intelligence System Using Machine Learning, RAG and Gated Fusion  
**Author:** Angel Oza  
**Document Purpose:** Deep empirical validation, distributional analysis, unit verification, and integrity auditing of Datasets A, B, and C.

---

## 1. Deep Inspection of Dataset C (`Crop_recommendation.csv`)

### 1.1 Structural Properties & Origin Analysis
- **Total Records:** 2,200 observations.
- **Classes:** 22 crops, exactly 100 observations per class (100% class balance).
- **Exact Duplicate Rows:** 0 (zero identical rows).
- **Minimum Inter-Sample Euclidean Distance:** 1.5236 (no degenerate collapsed clusters).

### 1.2 Feature Analysis, Physical Meaning & Units

| Feature Name | Verified Agricultural Meaning | Unit | Minimum | Median | Maximum | Mean ± Std | Distribution Type |
|---|---|---|---|---|---|---|---|
| `N` | Available Nitrogen ratio | Index / kg/ha equivalent | 0.0 | 37.0 | 140.0 | 50.55 ± 36.92 | Multi-modal (bimodal high/low nitrogen crops) |
| `P` | Available Phosphorus ratio | Index / kg/ha equivalent | 5.0 | 51.0 | 145.0 | 53.36 ± 32.99 | Multi-modal (high for apple/grapes: ~134, low for coffee/orange: ~17) |
| `K` | Available Potassium ratio | Index / kg/ha equivalent | 5.0 | 32.0 | 205.0 | 48.15 ± 50.65 | Clustered (high for apple/grapes/banana: ~200, low for pulses: ~20) |
| `temperature` | Ambient / root-zone temperature | °C | 8.83 | 25.60 | 43.68 | 25.62 ± 5.06 | Continuous unimodal Gaussian-like around 25°C |
| `humidity` | Relative ambient humidity | % | 14.26 | 80.47 | 99.98 | 71.48 ± 22.26 | Left-skewed (tropical crops peak at 80–85%) |
| `ph` | Soil pH reaction | pH standard (1–14) | 3.50 | 6.43 | 9.94 | 6.47 ± 0.77 | Normal distribution centered at neutral (6.47) |
| `rainfall` | Annual / seasonal water availability | mm | 20.21 | 94.87 | 298.56 | 103.46 ± 54.96 | Right-skewed (rice peaks at >200mm, legumes at ~50mm) |

### 1.3 Statistical Integrity Assessment & Findings
1. **Scientific Nature of the Data:**  
   Statistical inspection reveals that each crop class has characteristic mean values corresponding to established **FAO EcoCrop optimal envelopes** (e.g., Rice: $N \approx 80$, $P \approx 48$, $K \approx 40$, $\text{Rainfall} \approx 236\text{mm}$; Apple: $P \approx 134$, $K \approx 200$, $\text{Temp} \approx 22^\circ\text{C}$). The intra-class standard deviation is consistent ($\sigma_N \approx 11.5, \sigma_P \approx 12.0, \sigma_K \approx 3.1$), indicating that the dataset was synthesized from verified agronomic growth envelopes perturbed with continuous biological variance.
2. **Benchmark Suitability:**  
   Because the dataset possesses well-defined, non-linear decision boundaries grounded in real biological requirements, it serves as an excellent, standardized benchmark for comparing machine learning classifiers, multi-source fusion networks, and evaluating feature importance via SHAP.
3. **Data Leakage Safeguard:**  
   To prevent trivial label leakage during RAG and Gated Fusion, the RAG knowledge corpus will **never contain verbatim dataset rows or sample lookup rules**. Instead, the RAG corpus contains general agronomic monographs.

---

## 2. Deep Inspection of Dataset A (IVRI / ICAR KVK Bareilly Soil Laboratory Tests)

### 2.1 Properties & Origin
- **Source:** Krishi Vigyan Kendra, Indian Council of Agricultural Research (ICAR) – IVRI, Bareilly.
- **Verification:** Supported by scanned hard-copy laboratory soil testing certificates (`soilTestDataset-WAIJfM.pdf`).
- **Records:** 257 certified soil samples.
- **Variables:** `Sample`, `pH`, `EC`, `OC`, `N`, `P`, `K`, `S`, `Zn`, `B`, `Fe`, `Mn`, `Cu`, `Area in Acre`, `Bigha`.
- **Missing Values:** 0.

### 2.2 Agronomic Baseline in Dataset A
- Soil pH: $7.09 \pm 0.19$ (Neutral to slightly alkaline alluvial soil).
- EC: $0.14 \pm 0.04\text{ dS/m}$ (Well below the $0.8\text{ dS/m}$ non-saline threshold).
- Organic Carbon: $0.24 \pm 0.07\%$ (Deficient, <0.50% typical of intensive Gangetic alluvial cropping).
- Available Nitrogen: $138.4 \pm 38.2\text{ kg/ha}$ (Low, <280 kg/ha).
- Micronutrients: Zinc ($1.04\text{ ppm}$), Iron ($8.15\text{ ppm}$), Copper ($0.45\text{ ppm}$), Manganese ($2.05\text{ ppm}$).

**Role:** Used as the ground truth chemistry calibration for the **Soil Health Index (SHI)** and nutrient deficiency rule engine.

---

## 3. Deep Inspection of Dataset B (Berambadi 10-Year In-Situ Sensor Telemetry)

### 3.1 Properties & Origin
- **Source:** Berambadi / Bechanahalli Hydrological & Environmental Observatory (Southern India).
- **Coverage:** January 2016 to June 2025 (10 full years).
- **Sampling Frequency:** 15-minute continuous intervals (>300,000 observations).
- **Variables:** `timestamp`, `Temp_5cm`, `SM_5cm`, `EC_5cm`, `Temp_50cm`, `SM_50cm`, `EC_50cm`, `Precipitation`.
- **Data Quality:** True hardware missing values and logger fault codes (`E36`, `E6`) are preserved as `NaN` without synthetic smoothing.

**Role:** Powers the **Software Sensor Stream Simulator** (`iot/simulator.py`) and drives the real-time sensor reliability metric $r$ and noise robustness evaluations (Experiment E & F).

---

## 4. Final Dataset Integration & Separation Architecture

```
+-----------------------------------------------------------------------------------------------+
| LOGICALLY SEPARATED MULTI-SOURCE DATASET ARCHITECTURE                                          |
|                                                                                               |
|  [Dataset B: 10-Yr Berambadi Telemetry]        [Dataset A: IVRI Lab Chemistry]                |
|  - 300,000+ 15-min sensor readings             - 257 certified ICAR soil certificates         |
|  - SM, Temp, EC, Rainfall                      - N, P, K, pH, EC, OC, Micronutrients          |
|                 |                                              |                              |
|                 v                                              v                              |
|  +------------------------------+             +-------------------------------+               |
|  | Software Sensor Simulator    |             | Soil Health Intelligence Core |               |
|  | (Replay, Drift, Noise, r)    |             | (ICAR/FAO Standard Scoring)   |               |
|  +------------------------------+             +-------------------------------+               |
|                 |                                              |                              |
|                 +----------------------+-----------------------+                              |
|                                        |                                                      |
|                                        v                                                      |
|                       +---------------------------------+                                     |
|                       | Real-Time Soil State Vector     |                                     |
|                       +---------------------------------+                                     |
|                                        |                                                      |
|         +------------------------------+------------------------------+                       |
|         |                                                             |                       |
|         v                                                             v                       |
|  [Dataset C: 2,200 Benchmark Records]                   [Agronomic Knowledge Base (RAG)]      |
|  - 80/20 Stratified Split                               - Curated FAO/USDA/ICAR Corpus        |
|  - Train Numerical ML Baselines                         - Dense MiniLM Embeddings             |
|  - Train PyTorch Gated Fusion                           - Context-Conditioned Retrieval       |
|                 |                                                             |               |
|                 +------------------------------+------------------------------+               |
|                                                |                                              |
|                                                v                                              |
|                                +-------------------------------+                              |
|                                | Gated Fusion & XAI Provenance |                              |
|                                +-------------------------------+                              |
+-----------------------------------------------------------------------------------------------+
```
