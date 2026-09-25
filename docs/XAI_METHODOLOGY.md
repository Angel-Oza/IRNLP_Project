# Explainable AI (XAI) Methodology & Feature Attribution Report

**Project Title:** Explainable Multi-Source Soil Health Intelligence System Using Machine Learning, RAG and Gated Fusion  
**Author / Researcher:** Angel Oza (Marwadi University — Batch 2027)  
**Module:** Phase 3 — Explainable AI (SHAP)  
**Status:** Completed & Empirically Validated  

---

## 1. Theoretical Foundation of SHAP (SHapley Additive exPlanations)

To prevent opaque "black-box" decision-making in precision agriculture, this system implements **TreeSHAP** (Lundberg et al., Nature Machine Intelligence, 2020), grounding feature attributions in axiomatic cooperative game theory.

### 1.1 Mathematical Formulation of Shapley Values
For an agronomic input feature vector $x = (x_1, x_2, \dots, x_M) \in \mathbb{R}^M$, the Shapley value $\phi_j(f, x)$ quantifies the marginal contribution of feature $j$ to the predictive outcome $f(x)$ across all possible feature subsets $S \subseteq \mathcal{F} \setminus \{j\}$:

$$\phi_j(f, x) = \sum_{S \subseteq \mathcal{F} \setminus \{j\}} \frac{|S|! \, (|\mathcal{F}| - |S| - 1)!}{|\mathcal{F}|!} \Big( f_x(S \cup \{j\}) - f_x(S) \Big)$$

Where:
- $\mathcal{F} = \{N, P, K, \text{temp}, \text{humidity}, \text{pH}, \text{rainfall}\}$ ($|\mathcal{F}| = 7$).
- $f_x(S) = \mathbb{E}[f(X) \mid X_S = x_S]$ represents the conditional expectation of model prediction given observed feature subset $S$.

### 1.2 Axiomatic Guarantees of TreeSHAP
TreeSHAP uniquely satisfies four fundamental fairness and consistency axioms:
1. **Efficiency (Local Accuracy / Additivity):** The sum of all feature attributions plus the base expected value exactly equals the model's output:
   $$\sum_{j=1}^{M} \phi_j(x) = f(x) - \mathbb{E}[f(X)]$$
2. **Symmetry:** If two agronomic features $j$ and $k$ contribute identically to all possible subsets ($f(S \cup \{j\}) = f(S \cup \{k\})$), their SHAP values are identical ($\phi_j = \phi_k$).
3. **Dummy / Null Player:** If feature $j$ provides zero marginal contribution across all subsets ($f(S \cup \{j\}) = f(S)$), then $\phi_j = 0$.
4. **Additivity (Ensemble Consistency):** For an ensemble of decision trees $f(x) = \frac{1}{T} \sum_{t=1}^T f_t(x)$, the ensemble Shapley value is the arithmetic average of individual tree Shapley values: $\phi_j(f) = \frac{1}{T} \sum_{t=1}^T \phi_j(f_t)$.

---

## 2. Global Feature Attribution Results

Across the entire held-out test split ($N=440$), TreeSHAP was executed on the best-performing ensemble (`RandomForestClassifier`). The table below records the actual measured mean absolute Shapley values ($\mathbb{E}[|\phi_j|]$):

| Rank | Agronomic Feature | Mean Absolute SHAP Value ($\mathbb{E}[\|\phi_j\|]$) | Primary Agronomic Influence |
|:---:|:---|:---:|:---|
| **1** | **Rainfall** | **$0.1428$** | Distinguishes high water-requirement crops (Rice, Jute) from semi-arid crops (Mothbeans, Chickpea). |
| **2** | **Humidity** | **$0.1385$** | Separates tropical crops (Banana, Coconut, Papaya) from arid/temperate crops. |
| **3** | **Phosphorus (P)** | **$0.1294$** | Crucial discriminator for high-phosphate horticultural crops (Apple, Grapes) vs. low-demand cereals. |
| **4** | **Potassium (K)** | **$0.1210$** | Differentiates potassium-accumulator fruits (Grapes, Apple, Banana) from pulses. |
| **5** | **Nitrogen (N)** | **$0.1082$** | Distinguishes heavy vegetative feeders (Coffee, Cotton, Maize) from legume nitrogen-fixers. |
| **6** | **Temperature** | **$0.0891$** | Defines thermal tolerance boundaries (e.g., Apple cold-chill requirement vs. Mango/Papaya heat). |
| **7** | **Soil pH** | **$0.0645$** | Acts as an environmental boundary filter (Acidic tea/coffee vs. Alkaline-tolerant chickpea). |

---

## 3. Local Instance Attribution & Mathematical Additivity Check

For any incoming soil telemetry reading, the `ExplanationService` calculates the exact positive and negative force attributions.

### Concrete Example: Field Sample Tested (Rice Soil Profile)
- **Input Parameters:** $N=90.0\text{ kg/ha}, P=42.0\text{ kg/ha}, K=43.0\text{ kg/ha}, \text{Temp}=20.87^\circ\text{C}, \text{Humidity}=82.0\%, \text{pH}=6.50, \text{Rainfall}=202.93\text{ mm}$.
- **Base Expected Value $\mathbb{E}[f(X)]_{\text{rice}}$:** $0.0458$ ($1/22$ baseline prior class probability).
- **Predicted Class:** **Rice** (Confidence: $95.68\%$).
- **Local Feature Breakdown:**

| Feature | Physical Value | Scaled $x'_j$ | SHAP Value $\phi_j$ | Direction | Agronomic Interpretation |
|---|---|---|---|---|---|
| `rainfall` | $202.93\text{ mm}$ | $+1.81$ | **$+0.2430$** | Positive driver | High precipitation satisfies semi-aquatic paddy requirements. |
| `N` | $90.00\text{ kg/ha}$ | $+1.07$ | **$+0.1799$** | Positive driver | High available nitrogen supports tillering and vegetative canopy. |
| `humidity` | $82.00\%$ | $+0.48$ | **$+0.1520$** | Positive driver | High relative humidity matches tropical rice agro-climatic zones. |
| `K` | $43.00\text{ kg/ha}$ | $-0.10$ | **$+0.1368$** | Positive driver | Balanced potassium prevents lodging in mature rice culms. |
| `P` | $42.00\text{ kg/ha}$ | $-0.34$ | **$+0.1049$** | Positive driver | Adequate phosphorus promotes early root proliferation. |
| `temperature` | $20.87^\circ\text{C}$ | $-0.93$ | **$+0.1003$** | Positive driver | Temperature falls comfortably within standard rice growth limits. |
| `ph` | $6.50$ | $+0.03$ | **$-0.0058$** | Slight negative | Minor downward pressure (rice prefers slightly more acidic 5.5–6.2). |

$$\text{Reconstructed Output} = 0.0458 + (0.2430 + 0.1799 + 0.1520 + 0.1368 + 0.1049 + 0.1003 - 0.0058) = 0.9569 \approx 95.68\%$$
*(Exact mathematical additivity verified to $10^{-4}$).*

---

## 4. The Three-Tier Interpretability Framework

To eliminate ambiguity between statistical attribution and domain reasoning, the system establishes a strict **Three-Tier Semantic Boundary**:

```
+---------------------------------------------------------------------------------------------------+
| THREE-TIER INTERPRETABILITY FRAMEWORK                                                             |
|                                                                                                   |
|  [TIER 1: ML Feature Attribution (SHAP)]                                                          |
|  - Source: TreeSHAP on Random Forest / XGBoost classifiers.                                       |
|  - Scope: Quantifies the mathematical impact of numerical sensor features on model log-odds.      |
|  - Output: Exact positive/negative SHAP values, feature importance rankings, waterfall charts.    |
|                                                                                                   |
|  [TIER 2: Agronomic Literature Provenance (RAG Retrieval Engine)]                                 |
|  - Source: Dense passage retrieval over curated FAO, USDA-NRCS, and ICAR scientific monographs.  |
|  - Scope: Provides grounded biological principles, nutrient deficiency mechanisms, and citations. |
|  - Output: Verifiable document chunks, official publication titles, section headers, and DOIs.   |
|                                                                                                   |
|  [TIER 3: Structured Generative Agronomic Advisory (LLM Synthesis)]                              |
|  - Source: Context-constrained Gemini LLM (or deterministic fallback engine).                     |
|  - Scope: Translates numerical predictions (Tier 1) and scientific evidence (Tier 2) into         |
|           actionable farm management recommendations without hallucinations or fake dosages.     |
|  - Output: Structured JSON containing observation, risk analysis, fertilizer plan, and warnings.  |
+---------------------------------------------------------------------------------------------------+
```

---

## 5. Artifacts and Visualization Exports

All XAI artifacts are saved in the project repository:
- Global Feature Importance Bar Chart: `results/plots/xai/shap_global_bar.png`
- Per-Class Importance Matrix Heatmap: `results/plots/xai/shap_per_class_matrix.png`
- Local Instance Waterfall Plots: `results/plots/xai/shap_local_sample_*.png`
- Global Metrics JSON: `results/metrics/shap_global_importance.json`
