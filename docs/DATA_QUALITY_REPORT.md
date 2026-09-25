# Data Quality & Integrity Audit Report

**Generated:** Automated Phase 1 Pipeline  
**Overall Quality Status:** `PASSED_QUALITY_GATES`  

---

## 1. Dataset C Audit (Agronomic ML Benchmark)
- **Source File:** `Dataset/3/Crop_recommendation.csv`
- **Total Records:** `2200`
- **Feature Columns:** `N, P, K, temperature, humidity, ph, rainfall, label`
- **Missing Values:** `0` (0.00%)
- **Exact Duplicate Rows:** `0`
- **Number of Crop Classes:** `22`
- **Class Balance:** `PERFECT_BALANCE_100_PER_CLASS` (Exactly 100 observations per class)
- **Physical Boundary Violations:** `0 violations across all 7 features`
- **Integrity Verdict:** `PASSED`

---

## 2. Dataset A Audit (IVRI Laboratory Soil Chemistry)
- **Source File:** `Dataset/1/AgriNet_Dataset-G7TD2K.csv`
- **Total Certified Samples:** `257`
- **Missing Values:** `0` (0.00%)
- **Measured pH Range:** `6.76 – 7.45` (Neutral to slightly alkaline alluvial soil)
- **Measured EC Range:** `0.08 – 0.27 dS/m` (Non-saline, optimal)
- **Measured Organic Carbon:** `0.11% – 0.41%` (Low fertility regime)
- **Measured Nitrogen Range:** `59.0 – 179.0 kg/ha` (Low to deficient)
- **Integrity Verdict:** `PASSED`

---

## 3. Dataset B Audit (Berambadi In-Situ Telemetry 2016)
- **Source File:** `Dataset/2/20782348/2016_V1.csv`
- **Total 15-Minute Logs:** `35041`
- **Missing Values:** `0.0068%` (Natural logger gaps preserved without synthetic imputation)
- **Surface Soil Temp Range (5cm):** `15.0°C – 34.6°C`
- **Surface Soil Moisture Range (5cm):** `0.0% – 42.4%`
- **Cumulative Annual Precipitation:** `545.70 mm`
- **Integrity Verdict:** `PASSED_WITH_PRESERVED_MISSING_VALS`

---

## 4. Train/Test Partitioning & Anti-Leakage Audit
- **Training Set Size:** `1760` samples (80.0%)
- **Test Set Size:** `440` samples (20.0%)
- **Stratification:** `100% Equal Class Proportions in Train (80/class) and Test (20/class)`
- **Scaler Fitting Protocol:** `X_train only (Zero Test Leakage)`
- **Leakage Compliance:** `100% COMPLIANT (Zero Test Statistics Used During Preprocessing)`
