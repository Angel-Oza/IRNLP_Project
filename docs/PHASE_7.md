# Phase 7: Soil Health Index (SHI) Engine & Dataset A Validation

**Project Title:** Explainable Multi-Source Soil Health Intelligence System Using Machine Learning, RAG and Gated Fusion  
**Author / Researcher:** Angel Oza (B.Tech ICT, Marwadi University — Batch 2027)  
**Module:** Phase 7 — Rule-Based Soil Health Index (SHI) Engine & Empirical Validation  
**Status:** Completed & Empirically Validated  
**Verification Date:** September 2026  

---

## 1. Executive Summary & Objective

In precision agriculture, "soil health" is not a single sensor-readable physical scalar. It is a **multivariate, multifunctional biological and chemical condition**. Presenting an arbitrary synthetic number as ground truth is academically invalid. 

**Phase 7 Objective:**
1. Implement a **Rule-Based Soil Health Index (SHI) Engine** (`soil_health/shi.py`) computing a transparent 0–100 composite index based on peer-reviewed agronomic standards (ICAR Soil Health Card guidelines, FAO Soils Bulletins 64 & 76, USDA Handbook 60, and the Andrews et al. SMAF framework).
2. Establish rigorous mathematical response curves (optimum range, less-is-better, more-is-better plateau) for all 12 chemical parameters present in **Dataset A** (IVRI / ICAR KVK Bareilly laboratory certificates).
3. Validate the engine across all 257 certified soil tests (`soil_health/validator.py`), ensuring numerical stability, 0–100 bounded output, deficiency diagnosis, and dynamic weight re-normalization for partial parameter sets.
4. Establish clear academic boundaries: SHI is a **secondary expert-derived indicator**, explicitly separate from supervised crop classification predictions.

---

## 2. Dataset A Parameter Schema & Provenance

Dataset A comprises **257 certified soil laboratory certificates** issued by Krishi Vigyan Kendra (KVK) Bareilly under the Indian Veterinary Research Institute (IVRI / ICAR).

| Parameter | Parameter Name | Measured Unit | Desirable / Optimal Range | Scoring Function Type | Baseline Weight ($w_i$) | Authoritative Scientific Provenance |
|---|---|---|---|---|---|---|
| **pH** | Soil pH Reaction | Standard pH | 6.0 – 7.5 (Neutral) | Optimum Range (Trapezoidal) | **0.15** | FAO Soils Bulletin 64 / ICAR NBSS&LUP |
| **EC** | Electrical Conductivity | dS/m | < 0.8 dS/m (Non-saline) | Less is Better | **0.10** | USDA Handbook 60 / FAO Paper 39 |
| **OC** | Soil Organic Carbon | % | $\ge 0.75\%$ (Optimal) | More is Better (Plateau) | **0.20** | FAO Soils Bulletin 76 / ICAR Rating Scale |
| **N** | Available Nitrogen | kg/ha | 280 – 560 kg/ha | Optimum Range (Excess penalty) | **0.15** | ICAR Soil Health Card Scheme Norms |
| **P** | Available Phosphorus | kg/ha | 10 – 25 kg/ha | Optimum Range (Excess penalty) | **0.10** | ICAR Soil Health Card Norms / FAO 64 |
| **K** | Available Potassium | kg/ha | 110 – 280 kg/ha | Optimum Range (Excess penalty) | **0.10** | ICAR Soil Health Card Norms / IFA |
| **S** | Available Sulphur | ppm | $\ge 10.0\text{ ppm}$ | Critical Limit Plateau | **0.04** | ICAR AICRP Micronutrient Compendium |
| **Zn** | Available Zinc | ppm | $\ge 0.60\text{ ppm}$ (DTPA) | Critical Limit Plateau | **0.04** | ICAR AICRP Micronutrient Compendium |
| **B** | Available Boron | ppm | $\ge 0.50\text{ ppm}$ (Hot Water) | Critical Limit Plateau | **0.03** | ICAR AICRP Micronutrient Compendium |
| **Fe** | Available Iron | ppm | $\ge 4.50\text{ ppm}$ (DTPA) | Critical Limit Plateau | **0.03** | ICAR AICRP Micronutrient Compendium |
| **Mn** | Available Manganese | ppm | $\ge 2.00\text{ ppm}$ (DTPA) | Critical Limit Plateau | **0.03** | ICAR AICRP Micronutrient Compendium |
| **Cu** | Available Copper | ppm | $\ge 0.20\text{ ppm}$ (DTPA) | Critical Limit Plateau | **0.03** | ICAR AICRP Micronutrient Compendium |

**Total Baseline Weight Sum:** $\sum_{i=1}^{12} w_i = 0.15 + 0.10 + 0.20 + 0.15 + 0.10 + 0.10 + 0.04 + 0.04 + 0.03 + 0.03 + 0.03 + 0.03 = 1.00$.

---

## 3. Mathematical Scoring Formulations

Each parameter $x_i$ is mapped to an agronomic quality score $S_i(x_i) \in [0, 100]$:

### 1. Soil pH Reaction ($S_{\text{pH}}$)
$$S_{\text{pH}}(x) = \begin{cases}
100.0 & \text{if } 6.0 \le x \le 7.5 \\
\max\left(0, 100 - \left(\frac{6.0 - x}{1.5}\right)^2 \times 80\right) & \text{if } x < 6.0 \text{ (Acidity: P-fixation, Al toxicity)} \\
\max\left(0, 100 - \left(\frac{x - 7.5}{1.5}\right)^2 \times 80\right) & \text{if } x > 7.5 \text{ (Alkalinity: Micronutrient precipitation)}
\end{cases}$$

### 2. Electrical Conductivity / Salinity ($S_{\text{EC}}$)
$$S_{\text{EC}}(x) = \begin{cases}
100.0 & \text{if } x \le 0.8\text{ dS/m (Non-saline)} \\
100.0 - \frac{x - 0.8}{0.8} \times 40.0 & \text{if } 0.8 < x \le 1.6\text{ dS/m (Slight salinity warning)} \\
\max\left(0, 60.0 - \frac{x - 1.6}{1.4} \times 40.0\right) & \text{if } 1.6 < x \le 3.0\text{ dS/m (Moderate osmotic stress)} \\
\max\left(0, 20.0 - \frac{x - 3.0}{2.0} \times 20.0\right) & \text{if } x > 3.0\text{ dS/m (Severe osmotic drought)}
\end{cases}$$

### 3. Soil Organic Carbon ($S_{\text{OC}}$)
$$S_{\text{OC}}(x) = \begin{cases}
\min\left(100.0, 90.0 + (x - 0.75) \times 10.0\right) & \text{if } x \ge 0.75\% \text{ (Optimal biological activity)} \\
60.0 + \frac{x - 0.50}{0.25} \times 30.0 & \text{if } 0.50 \le x < 0.75\% \text{ (Medium carbon pool)} \\
\max\left(0, \frac{x}{0.50} \times 60.0\right) & \text{if } x < 0.50\% \text{ (Critically degraded microbiome)}
\end{cases}$$

### 4. Primary Macronutrients ($N, P, K$)
Scored using standard piecewise ICAR deficiency and adequacy curves:
- **Nitrogen:** $S_{\text{N}} = \frac{x}{280} \times 70$ for $x < 280\text{ kg/ha}$; $70 + \frac{x - 280}{280} \times 30$ for $280 \le x \le 560\text{ kg/ha}$.
- **Phosphorus:** $S_{\text{P}} = \frac{x}{10} \times 60$ for $x < 10\text{ kg/ha}$; $60 + \frac{x - 10}{15} \times 40$ for $10 \le x \le 25\text{ kg/ha}$.
- **Potassium:** $S_{\text{K}} = \frac{x}{110} \times 60$ for $x < 110\text{ kg/ha}$; $60 + \frac{x - 110}{170} \times 40$ for $110 \le x \le 280\text{ kg/ha}$.

### 5. Secondary & Micronutrients ($S, Zn, B, Fe, Mn, Cu$)
$$S_{\text{micro}}(x, c) = \begin{cases}
100.0 & \text{if } x \ge c \\
\max\left(0, \frac{x}{c} \times 80.0\right) & \text{if } x < c \text{ (Deficient relative to critical limit } c)
\end{cases}$$

### 6. Dynamic Weighted Aggregation
When a partial parameter set is provided, weights are dynamically re-normalized over available channels:
$$\tilde{w}_i = \frac{w_i}{\sum_{j \in \text{available}} w_j}, \quad \text{SHI} = \sum_{i \in \text{available}} \tilde{w}_i \cdot S_i(x_i) \in [0, 100]$$

---

## 4. Health Categorization Framework

Based on Andrews et al. (2002) and ICAR Soil Quality Indexing:
- **High / Optimal Soil Health:** $\text{SHI} \ge 80.0$
- **Good / Moderate Soil Health:** $65.0 \le \text{SHI} < 80.0$
- **Fair / Low Soil Health:** $50.0 \le \text{SHI} < 65.0$
- **Degraded / Critical Soil Health:** $\text{SHI} < 50.0$

---

## 5. Empirical Results Across Dataset A (N=257 KVK Bareilly Samples)

Validation performed across all 257 certified laboratory soil samples:

### Overall Summary Statistics:
- **Mean SHI Score:** **66.55 ± 5.13**
- **Median SHI Score:** **63.78**
- **Score Range:** **[58.72, 76.78]** (100% strictly bounded within [0, 100])
- **NaN / Missing Scores:** **0.00%**

### Category Distribution:
- **Good / Moderate Soil Health (65–80):** 100 samples (38.91%)
- **Fair / Low Soil Health (50–65):** 157 samples (61.09%)
- **Degraded (< 50) or Optimal (> 80):** 0 samples (0.00%)

### Key Agronomic Findings:
1. **Soil Reaction & Salinity (Optimal):** Mean pH is 7.10 (Score 100/100) and mean EC is 0.14 dS/m (Score 100/100) with 0.0% saline samples.
2. **Micronutrient Sufficiency (High):** S, Zn, Fe, Cu are 100% adequate across all tested alluvial fields.
3. **Primary Limiting Factors (Severe Deficiencies):**
   - **Organic Carbon (OC):** 100% of samples are critically low ($0.24\% \pm 0.07\%$, Score: 29.12/100).
   - **Available Nitrogen (N):** 100% of samples are severely deficient ($114.32 \pm 31.96\text{ kg/ha}$, Score: 28.58/100).
   - **Available Phosphorus & Potassium:** 66.5% deficient in P, 72.0% deficient in K.

---

## 6. Artifacts Created & Output Structure

- **Core Module:** `soil_health/shi.py`
- **Validation Script:** `soil_health/validator.py`
- **Tests:** `tests/test_shi.py` (10 unit/integration tests)
- **Metrics JSON:** `results/metrics/phase7_shi_metrics.json`
- **Tables:** `results/tables/phase7_shi_summary.csv`, `results/tables/phase7_shi_summary.md`
- **Visualizations:**
  - `results/plots/shi/phase7_shi_distribution.png`
  - `results/plots/shi/phase7_parameter_scores_boxplot.png`
  - `results/plots/shi/phase7_shi_vs_nutrients.png`

### Example Structured Output Payload:
```json
{
  "shi_score": 68.45,
  "category": "Good / Moderate Soil Health",
  "parameter_scores": {
    "pH": 100.0,
    "EC": 100.0,
    "OC": 27.6,
    "N": 27.0,
    "P": 64.24,
    "K": 64.71,
    "S": 100.0,
    "Zn": 100.0,
    "B": 100.0,
    "Fe": 100.0,
    "Mn": 100.0,
    "Cu": 100.0
  },
  "weighted_contributions": {
    "pH": 15.0,
    "EC": 10.0,
    "OC": 5.52,
    "N": 4.05,
    "P": 6.42,
    "K": 6.47,
    "S": 4.0,
    "Zn": 4.0,
    "B": 3.0,
    "Fe": 3.0,
    "Mn": 3.0,
    "Cu": 3.0
  },
  "deficiencies": [
    "Soil Organic Carbon (OC): Critically Low Organic Carbon (< 0.50%)",
    "Available Nitrogen (N): Deficient / Low Nitrogen (< 280 kg/ha)"
  ],
  "warnings": [],
  "provenance_summary": "ICAR Soil Health Card Standards / FAO Bulletins 64 & 76 / USDA Handbook 60"
}
```
