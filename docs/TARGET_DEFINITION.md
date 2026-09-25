# Target Variable Definition & Soil Health Index Methodology

**Project Title:** Explainable Multi-Source Soil Health Intelligence System Using Machine Learning, RAG and Gated Fusion  
**Author:** Angel Oza  
**Document Purpose:** Scientifically Defensible Definition of Target Variables, Nutrient Classification Thresholds, and Soil Health Index Calculation  

---

## 1. Problem Statement on "Soil Health" Target Variable

In precision agriculture research, there is no single universal, sensor-readable scalar column named `soil_health`. Soil health is inherently a **multivariate, multi-functional agronomic condition** comprising:
1. **Chemical fertility** (macro and micro-nutrients: N, P, K, S, Zn, Fe, etc., and Organic Carbon),
2. **Physical / hydrological condition** (soil moisture, bulk density, texture, water retention),
3. **Biological & physiological suitability** (pH balance, electrical conductivity / salinity, crop root compatibility).

Presenting a fabricated or arbitrarily invented label as "ground truth" would be unscientific and invalidate academic research. Therefore, this project adopts a rigorous, dual-target approach:

- **Target 1: Agronomic Crop Suitability Class (Empirical Ground Truth)**
  - Supervised ground truth target from Dataset 3 (22 distinct crop classes).
  - Used for training and benchmark evaluation of Machine Learning models (Logistic Regression, Random Forest, XGBoost), Gated Fusion, and SHAP Explainable AI.
- **Target 2: Scientifically Defensible Soil Health Index (SHI) & Nutrient Status (Rule-Based Agronomic Evaluation)**
  - Multi-parameter rating score (0–100) calculated using documented, peer-reviewed agronomic standards (ICAR Soil Health Card guidelines, FAO Soil Bulletin, USDA NRCS).
  - Clearly documented as an **expert-derived composite index**, never misrepresented as a single raw sensor output.

---

## 2. Standard Scientific Thresholds (ICAR / FAO / USDA)

The Soil Health Intelligence Core relies on authoritative agricultural thresholds established by the **Indian Council of Agricultural Research (ICAR)** and the **Food and Agriculture Organization (FAO)**:

### A. Primary Nutrients (Macro-Nutrients)

| Parameter | Unit | Low / Deficient (Score: 30) | Medium / Optimal (Score: 100) | High / Excess (Score: 70) | Reference Standard |
|---|---|---|---|---|---|
| **Nitrogen (N)** | kg/ha | < 280 (or < 140 ppm) | 280 – 560 kg/ha | > 560 kg/ha | ICAR Soil Health Card Norms |
| **Phosphorus (P)** | kg/ha | < 10 (or < 23 kg P₂O₅/ha) | 10 – 25 kg/ha | > 25 kg/ha | ICAR Norms / FAO Bulletin 64 |
| **Potassium (K)** | kg/ha | < 110 (or < 132 kg K₂O/ha)| 110 – 280 kg/ha | > 280 kg/ha | ICAR Norms / FAO Bulletin 64 |

### B. Soil Chemical Balance & Reaction

| Parameter | Unit | Acidic / Low | Optimal / Neutral | Alkaline / High | Agronomic Implications |
|---|---|---|---|---|---|
| **pH** | standard pH | < 6.0 (Acidic, P-fixation risk) | **6.0 – 7.5 (Optimal)** | > 7.5 (Alkaline/Calcareous) | Controls micronutrient bioavailability |
| **Electrical Conductivity (EC)** | dS/m | **< 0.8 (Normal, Non-saline)** | 0.8 – 1.6 (Critical, slight salinity) | > 1.6 (Injurious/Saline) | High EC causes osmotic drought in crops |
| **Organic Carbon (OC)** | % | < 0.50% (Low) | 0.50% – 0.75% (Medium) | > 0.75% (High) | Essential indicator of soil microbiome & fertility |

### C. Physical Telemetry Parameters (IoT Sensors)

| Parameter | Unit | Deficit / Low | Optimal Range | Saturated / Excess | Agronomic Implications |
|---|---|---|---|---|---|
| **Soil Moisture (SM)** | % vol | < 15% (Water stress / Wilting point) | **20% – 35% (Field Capacity)** | > 40% (Waterlogging / Hypoxia) | Governs nutrient diffusion & root respiration |
| **Soil Temperature** | °C | < 12°C (Root dormancy) | **18°C – 28°C (Optimal)** | > 35°C (Heat stress / Microbial inhibition) | Affects metabolic uptake and nitrification |

---

## 3. Mathematical Formulation of the Soil Health Index (SHI)

To synthesize multi-source telemetry into an explainable score for the dashboard and recommendation engine, a **Weighted Additive Quality Index** (Andrews et al., 2002; ICAR Framework) is implemented:

$$\text{SHI} = \sum_{i=1}^{M} w_i \cdot S_i(x_i)$$

Where:
- $x_i$ is the measured value of parameter $i$ (e.g., N, P, K, pH, EC, Moisture, Temperature, OC).
- $S_i(x_i) \in [0, 100]$ is the standard non-linear scoring function for parameter $i$ based on ICAR/FAO agronomic response curves (triangular/trapezoidal scoring).
- $w_i$ is the agronomic weight assigned to parameter $i$, such that $\sum_{i=1}^{M} w_i = 1.0$.

### Parameter Weighting Distribution:
- Primary Nutrients ($N, P, K$): $w_N = 0.20, w_P = 0.15, w_K = 0.15$ (Total: 0.50)
- Soil Reaction & Salinity ($pH, EC$): $w_{pH} = 0.15, w_{EC} = 0.10$ (Total: 0.25)
- Physical & Biological Dynamics ($SM, OC, Temp$): $w_{SM} = 0.10, w_{OC} = 0.10, w_{Temp} = 0.05$ (Total: 0.25)

### Health Status Categorization:
- **Optimal / Excellent:** $\text{SHI} \ge 80$
- **Moderate / Fair:** $60 \le \text{SHI} < 80$
- **Degraded / Stressed:** $40 \le \text{SHI} < 60$
- **Critical / Deficient:** $\text{SHI} < 40$

---

## 4. Structured Output Format for Soil Intelligence Core

The Soil Intelligence Core produces a structured JSON output consumable by the RAG Retrieval Engine, Gated Fusion Network, and Web Dashboard:

```json
{
  "overall_status": "Moderate",
  "score": 68.5,
  "health_category": "Fair",
  "nutrient_status": {
    "nitrogen": {"value": 140.0, "status": "Low", "unit": "kg/ha", "deficiency_flag": true},
    "phosphorus": {"value": 18.5, "status": "Optimal", "unit": "kg/ha", "deficiency_flag": false},
    "potassium": {"value": 240.0, "status": "Optimal", "unit": "kg/ha", "deficiency_flag": false}
  },
  "parameters": {
    "ph": {"value": 6.8, "status": "Optimal", "range": "6.0 - 7.5"},
    "ec": {"value": 0.25, "status": "Non-saline", "unit": "dS/m"},
    "moisture": {"value": 28.4, "status": "Optimal", "unit": "%"},
    "temperature": {"value": 24.2, "status": "Optimal", "unit": "°C"},
    "organic_carbon": {"value": 0.38, "status": "Low", "unit": "%"}
  },
  "identified_stresses": [
    "Nitrogen Deficiency (< 280 kg/ha)",
    "Low Soil Organic Carbon (< 0.50%)"
  ],
  "retrieval_query_triggers": [
    "Nitrogen deficiency correction for target crop",
    "Organic matter enrichment and compost management",
    "Soil pH buffer stability"
  ]
}
```

---

## 5. Summary & Academic Safeguards

1. **Empirical ML models** are trained strictly on verifiable, publicly benchmarked ground truth (`label` in Dataset 3).
2. **Soil Health Indices and status classifications** are calculated strictly from published ICAR and FAO agronomic criteria and are explicitly designated as *Rule-Based Agronomic Indexes* in all evaluations and dashboards.
3. No synthetic or invented ground-truth labels are passed off as empirical measurements.
