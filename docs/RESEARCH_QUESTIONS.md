# Core Research Questions & Hypotheses

**Project Title:** Explainable Multi-Source Soil Health Intelligence System Using Machine Learning, RAG and Gated Fusion  
**Author:** Angel Oza (Marwadi University)  
**Document Purpose:** Formal articulation of primary research questions, hypotheses, experimental mechanisms, and validation metrics for academic publication.

---

## Research Question 1 (RQ1): Multi-Source Knowledge Integration
> **RQ1:** *Does integrating retrieved external agronomic scientific knowledge (dense text representations) with numerical soil and environmental telemetry yield statistically superior predictive performance and calibration compared to unimodal numerical-only machine learning models?*

- **Theoretical Motivation:** Unimodal numerical models lack exposure to broad agronomic principles (e.g., pH-induced nutrient lockup, thermal tolerance boundaries). Fusing dense semantic embeddings from authoritative literature provides structural domain priors.
- **Experimental Test:** Compare **Experiment A** (Numerical ML baselines: RF, XGBoost, MLP) against **Experiment C & D** (Simple and Gated Multi-Source Fusion).
- **Primary Metrics:** Test Accuracy (%), Macro-F1 score, Expected Calibration Error (ECE), and Log-Loss on held-out test split $\mathcal{D}_{\text{test}}$.
- **Expected Evidence:** Multi-source fusion models achieve competitive or higher classification accuracy while significantly reducing high-confidence out-of-distribution error rates.

---

## Research Question 2 (RQ2): Gated vs. Unweighted Multi-Source Fusion
> **RQ2:** *Does a dynamic sigmoid gating mechanism ($g \in (0, 1)^d$) outperform unweighted concatenation and late-fusion ensembles when combining heterogeneous numerical sensor representations and retrieved text embeddings?*

- **Theoretical Motivation:** Different soil regimes require varying degrees of reliance on immediate numerical readings versus general biological constraints. Fixed concatenation treats both modalities uniformly, whereas dynamic gating learns instance-specific modality weights.
- **Experimental Test:** Compare **Experiment C** (Simple Concatenation Fusion) directly with **Experiment D** (Dynamic Gated Fusion).
- **Primary Metrics:** Macro-F1 gain ($\Delta \text{F1}$), convergence speed (epochs to minimal validation loss), and gate distribution analysis ($\mathbb{E}[g_{\text{sensor}}]$ across crop classes).
- **Expected Evidence:** Gated fusion achieves superior classification performance and lower parameter sensitivity due to adaptive element-wise feature routing.

---

## Research Question 3 (RQ3): Fault Tolerance & Reliability-Aware Gated Fusion
> **RQ3:** *Does incorporating a real-time sensor stream reliability score ($r \in [0, 1]$) into the gated fusion architecture provide superior resilience against synthetic sensor noise, calibration drift, and missing telemetry channels compared to unweighted fusion and single-source baselines?*

- **Theoretical Motivation:** In physical agricultural environments, IoT sensors suffer from drift, fouling, and intermittent connectivity. A reliability-aware gate can actively suppress the corrupted sensor branch and fallback on domain knowledge.
- **Experimental Test:** Execute **Experiment F** across noise levels $\eta \in \{0\%, 5\%, 10\%, 20\%\}$ and drift/missingness conditions, comparing Baseline ML, Simple Fusion, Standard Gated Fusion, and Reliability-Aware Gated Fusion.
- **Primary Metrics:** Relative Resilience Index ($\text{RRI} = \frac{\text{F1}_{\text{noisy}}}{\text{F1}_{\text{clean}}}$), Macro-F1 degradation slope ($\frac{d\text{F1}}{d\eta}$), and correlation between injected noise and gate activation $\text{Corr}(\eta, 1 - \bar{g})$.
- **Expected Evidence:** Reliability-Aware Gated Fusion demonstrates a flatter degradation curve, retaining over $85\%$ of clean performance at $20\%$ noise, while baseline models suffer steep performance drops.
