# Project Understanding & Research Specification (Revised)

**Project Title:** Explainable Multi-Source Soil Health Intelligence System Using Machine Learning, RAG and Gated Fusion  
**Author / Researcher:** Angel Oza (B.Tech ICT, Marwadi University — Batch 2027)  
**Course / Subject:** Information Retrieval & Natural Language Processing (IRNLP)  
**System Type:** Software-Only Academic Research & Demonstration Platform  
**Status:** Validated Architecture & Research Design  

---

## 1. Executive Summary & Problem Statement

Soil health is a critical determinant of global agricultural productivity, ecological sustainability, and food security. Conventional approaches to soil health assessment typically rely on:
1. **Periodic Laboratory Soil Testing:** Accurate in chemical analysis (NPK, micronutrients, pH, organic carbon), but static, labor-intensive, slow, and unable to capture real-time environmental dynamics.
2. **Standalone IoT Sensor Deployments:** Capable of high-frequency temporal telemetry (moisture, temperature, electrical conductivity), but blind to crop-specific agronomic requirements, soil chemistry context, and regional agronomic science.
3. **Black-Box Deep Learning / AI Models:** Provide predictive outputs without transparent, human-interpretable rationale or evidence tracing, reducing farmer trust and agronomist adoption.
4. **Isolated Operational Silos:** Numerical sensor feeds and unstructured scientific literature (FAO, USDA, ICAR guidelines) exist in total isolation, with no adaptive mechanism to align real-time field telemetry with authoritative agronomic knowledge.

### The Research Solution
This project develops an **Explainable Multi-Source Soil Health Intelligence System** that fuses:
- **Software-simulated real-time sensor telemetry** (Dataset B: 10-year in-situ environmental measurements: soil moisture, temperature, electrical conductivity, rainfall),
- **Laboratory soil chemical nutrient profiles** (Dataset A: certified ICAR KVK Bareilly laboratory tests: N, P, K, pH, OC, micronutrients),
- **Machine Learning baselines & benchmarks** (Dataset C: 22-crop agronomic suitability benchmark: Logistic Regression, Random Forest, XGBoost, MLP),
- **Adaptive Retrieval-Augmented Generation (RAG)** over authoritative agronomic scientific literature (FAO, USDA, ICAR corpus),
- **PyTorch-based Gated Fusion Network** with dynamic sensor reliability weighting,
- **Explainable AI via SHAP** (feature attribution) and **RAG citation provenance**,
- **Evidence-backed agronomic recommendation generation**, and
- **Interactive Web Dashboard and Empirical Experimentation Suite** (Experiments A–F).

NO PHYSICAL HARDWARE IS USED.

---

## 2. Logical Separation of Datasets

The system enforces strict dataset separation without unscientific row concatenation:

| Dataset Identifier | Physical Dataset Source | Modality & Scope | Operational Role |
|---|---|---|---|
| **Dataset A (Chemistry)** | IVRI / ICAR KVK Bareilly | 257 certified laboratory soil samples (NPK, pH, EC, OC, Micronutrients) | Calibrates the Rule-Based **Soil Health Index (SHI)** & nutrient deficiency rules. |
| **Dataset B (Telemetry)** | Berambadi Observatory (2016–2025) | 300,000+ 15-min temporal sensor records (Moisture, Temp, EC, Rain) | Drives the **Software Sensor Stream Simulator** (`iot/simulator.py`), sensor noise/drift injection, and reliability score ($r$). |
| **Dataset C (Agronomy)** | Precision Agriculture Benchmark | 2,200 observations across 22 crop classes (N, P, K, Temp, Humidity, pH, Rain) | Trains **ML Baselines**, powers **SHAP XAI**, and trains the **Gated Fusion Network**. |

---

## 3. Seven-Stage System Architecture

```
                  +----------------------------------------------+
                  |  Stage 1: IoT Sensing Layer (Simulation)     |
                  |  - Berambadi 10-Year In-Situ Telemetry Feed  |
                  |  - Replay Engine (interval, speed, faults)   |
                  +----------------------------------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |  Stage 2: Preprocessing & Reliability Engine |
                  |  - Denoising, Imputation, Train-only Scaling |
                  |  - Real-time Reliability Metric r in [0, 1]  |
                  +----------------------------------------------+
                                         |
                     +-------------------+-------------------+
                     |                                       |
                     v                                       v
+------------------------------------------+ +------------------------------------------+
| Stage 4a: Sensor Feature Encoder         | | Stage 3: Adaptive Contextual Retrieval   |
| - Numerical Feature Embedding E_sensor   | | - Sensor-conditioned Query Generator     |
| - ML Classifiers & Probabilities         | | - Dense MiniLM Embeddings + FAISS        |
+------------------------------------------+ +------------------------------------------+
                     |                                       |
                     +-------------------+-------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |  Stage 4b: Dynamic Gated Fusion Module       |
                  |  - Dynamic Gate: g = sigma(W_g [E_s; E_t; r])|
                  |  - Fused: Z = (r*g) * E_s + (1 - r*g) * E_t  |
                  +----------------------------------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |  Stage 5: Soil Health Intelligence Core      |
                  |  - Rule-Based Soil Health Index (0-100)      |
                  |  - ICAR / FAO Agronomic Standard Criteria    |
                  +----------------------------------------------+
                                         |
                     +-------------------+-------------------+
                     |                                       |
                     v                                       v
+------------------------------------------+ +------------------------------------------+
| Stage 6: Explainable AI (XAI) Module     | | Stage 7: Actionable Recommendation Gen   |
| - SHAP Feature Importance & Direction    | | - Constrained LLM Advisory Engine        |
| - Parameter Contribution Attribution     | | - Grounded Evidence Traceback Citations  |
+------------------------------------------+ +------------------------------------------+
                     |                                       |
                     +-------------------+-------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |  Interactive Web Dashboard (React + FastAPI) |
                  |  & Automated Experiment Runner (Exp A - F)   |
                  +----------------------------------------------+
```

---

## 4. Key Academic Safeguards

1. **No Data Leakage:** Preprocessing fitted on training split only; RAG corpus contains no dataset rows or sample IDs; retrieval queries use continuous soil parameters without label keywords.
2. **Empirical Rigor:** No fabricated metrics or synthetic datasets presented as real research data.
3. **Reproducibility:** All metrics calculated from actual executed test scripts (`pytest`, `experiments/run_all.py`).
