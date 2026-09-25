# Phase 9: Backend REST API & Interactive Web Dashboard

**Project Title:** Explainable Multi-Source Soil Health Intelligence System Using Machine Learning, RAG and Gated Fusion  
**Author / Researcher:** Angel Oza (B.Tech ICT, Marwadi University — Batch 2027)  
**Module:** Phase 9 — Production Backend REST API & Interactive Web Dashboard  
**Status:** Completed & Empirically Validated  
**Verification Date:** September 2026  

---

## 1. Executive Summary & Objective

Phase 9 transforms the end-to-end scientific research pipeline (Phases 1–8) into an interactive software application comprising:
1. **FastAPI Backend REST Service (`backend/`):** Exposes all research capabilities (Soil Health Index calculation, supervised ML crop prediction, TreeSHAP feature attributions, dense FAISS RAG retrieval, grounded advisory generation, and software sensor stream replay) without duplicating research logic.
2. **Interactive React + Vite Web Dashboard (`frontend/`):** High-contrast, responsive agronomic decision support interface featuring 6 specialized views and a unified multi-source intelligence pipeline.
3. **Comprehensive Automated Test Suite (`tests/test_api.py`):** 10 exhaustive unit and integration tests verifying all endpoints and schema compliance.
4. **Zero-Duplication Integration:** Directly imports singleton instances of research modules from Phases 1–8.

---

## 2. System Architecture & Component Flow

```
+----------------------------------------------------------------------------------------------------+
|                                    React + Vite Web Dashboard                                      |
|  [Dashboard/Home]  [Soil Analysis]  [Crop Prediction]  [Sensors]  [Advisory]  [Corpus]  [Pipeline] |
+----------------------------------------------------------------------------------------------------+
                                                  |  (REST / JSON)
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                      FastAPI REST Service (backend/)                                |
|  GET /health   POST /api/analyze   POST /api/predict   POST /api/advisory   POST /api/intelligence  |
|  GET /api/sensors   GET /api/corpus   GET /api/samples                                             |
+----------------------------------------------------------------------------------------------------+
       |                    |                     |                     |                    |
       v                    v                     v                     v                    v
 [soil_health.shi]    [models.evaluator]    [xai.service]       [rag.retriever]     [iot.simulator]
   SHI Engine           Best Model &          TreeSHAP            Dense FAISS         Telemetry Replay
  (0-100 Score)         Predictions          Attributions         Vector Store        & Reliability r(t)
       |                    |                     |                     |                    |
       +--------------------+---------------------+---------------------+--------------------+
                                                  |
                                                  v
                                     [advisory.generator]
                                   Grounded Advisory Engine
                                (Evidence Grounding & Validation)
```

---

## 3. Backend REST API Specifications

### Base URL: `http://localhost:8000`

### Endpoints:

#### 1. `GET /health`
- **Purpose:** Verifies backend health and readiness of all research singletons.
- **Response:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "timestamp": "2026-09-09 17:34:39",
  "components": {
    "shi_engine": "READY (ICAR/FAO Standards)",
    "ml_crop_classifier": "READY (Random Forest Baseline, N=22 classes)",
    "shap_explainer": "READY (TreeSHAP Explainer)",
    "rag_retriever": "READY (FAISS MiniLM Index, 40 Chunks)",
    "advisory_generator": "READY (Evidence-Grounded Synthesizer)",
    "sensor_simulator": "READY (Software Telemetry Replay)"
  }
}
```

#### 2. `POST /api/analyze`
- **Purpose:** Computes 0–100 Soil Health Index (SHI), parameter scores, deficiencies, and limiting factors.
- **Request Payload:**
```json
{
  "pH": 7.13,
  "EC": 0.13,
  "OC": 0.23,
  "N": 108.0,
  "P": 11.59,
  "K": 130.0,
  "S": 14.67,
  "Zn": 0.98,
  "B": 0.60,
  "Fe": 8.06,
  "Mn": 2.03,
  "Cu": 0.47
}
```
- **Response:** `shi_score`, `category`, `parameter_scores`, `parameter_details`, `deficiencies`, `strengths`, `limiting_factors`, `agronomic_summary`, `warnings`.

#### 3. `POST /api/predict`
- **Purpose:** Predicts optimal crop suitability and decomposes exact TreeSHAP feature attributions.
- **Request Payload:**
```json
{
  "N": 90.0,
  "P": 42.0,
  "K": 43.0,
  "temperature": 20.87,
  "humidity": 82.00,
  "ph": 6.50,
  "rainfall": 202.93
}
```
- **Response:** `predicted_crop`, `confidence`, `top_probabilities`, `feature_attributions` (with Shapley values and positive/negative contribution tags), `summary_statement`.

#### 4. `POST /api/advisory`
- **Purpose:** Generates actionable remediation recommendations validated against the FAO/USDA/ICAR RAG index ($\ge 0.35$ cosine threshold).
- **Request Payload:**
```json
{
  "soil_data": { "pH": 7.13, "EC": 0.13, "OC": 0.23, "N": 108.0, "P": 11.59, "K": 130.0 },
  "target_crop": "rice"
}
```
- **Response:** `recommendations` (topic, issue, action, rationale, urgency, validation_status, supporting_evidence with citations), `evidence_validation` summary.

#### 5. `POST /api/intelligence`
- **Purpose:** End-to-end multi-source execution pipeline combining all stages: Input &rarr; Preprocessing &rarr; ML Prediction &rarr; TreeSHAP &rarr; SHI Evaluation &rarr; RAG Retrieval &rarr; Grounded Advisory.
- **Request Payload:** Complete soil chemistry + weather parameters.
- **Response:** Consolidated report containing `pipeline_stages`, `soil_health`, `crop_prediction`, `advisory`, and `provenance`.

#### 6. `GET /api/sensors`
- **Purpose:** Replays historical in-situ telemetry from Dataset B (Berambadi Observatory 2016–2025) and evaluates online dynamic reliability score $r(t) \in [0, 1]$.
- **Query Parameters:** `count` (default: 1), `step` (default: 1), `year` (default: 2016), `reset` (boolean).
- **Response:** `frames` (Temp, Moisture, EC at 5cm & 50cm, Precipitation), `reliability` ($r$, $s_{\text{range}}, s_{\text{dist}}, s_{\text{avail}}, s_{\text{step}}$), `simulation_notice`.

#### 7. `GET /api/corpus`
- **Purpose:** Returns the indexed 40 scientific chunks and source authority metadata from FAISS.

#### 8. `GET /api/samples`
- **Purpose:** Returns curated empirical benchmark profiles from Dataset A (KVK Bareilly) and agronomic archetypes.

---

## 4. Frontend Web Dashboard Architecture

Built with **React 18 + Vite** using a modern, high-contrast agronomic design system:

| View Component | Research Phase Connected | Key Visual Capabilities |
|---|---|---|
| `Dashboard.jsx` | All Phases | System flowchart, KPI status grid, quick-start benchmark profile loaders. |
| `SoilAnalysis.jsx` | Phase 7 (SHI Engine) | 12-parameter form, circular SHI dial, parameter scoring bars, deficiencies list. |
| `CropPrediction.jsx` | Phase 2 & 3 (ML + TreeSHAP) | 7-feature input, crop prediction card, top-5 distribution, diverging SHAP bar chart. |
| `SensorMonitoring.jsx` | Phase 6 (IoT & Reliability) | Live/stepped telemetry cards, $r(t)$ reliability dial, signal decomposition, playback controls. |
| `Recommendations.jsx` | Phase 8 (Advisory & RAG) | Urgency-coded cards, validation badges, expandable verbatim scientific citations. |
| `EvidenceCorpus.jsx` | Phase 4 (FAISS RAG) | Searchable 40-chunk knowledge base viewer with authority and section filters. |
| `IntelligencePipeline.jsx` | Phases 1–8 Unified | Synchronized 5-stage pipeline runner with unified diagnostic report generation. |

---

## 5. How to Run the Application

### A. Start the Backend API (FastAPI)

```bash
# From the project root directory:
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
The API documentation (Swagger UI) is accessible at: `http://localhost:8000/docs`

### B. Start the Frontend Dashboard (React + Vite)

```bash
# Navigate to frontend directory and start the dev server:
cd frontend
npm run dev
```
The interactive web dashboard is accessible at: `http://localhost:5173`

---

## 6. Verification & Automated Test Results

The backend test suite (`tests/test_api.py`) runs with `pytest` and verifies all 10 endpoint test cases alongside the complete Phase 1–8 test suite:

```bash
python3 -m pytest
```

### Complete Test Results:
- **`tests/test_api.py`:** 10 / 10 PASSED (100%)
- **Phase 1–8 Tests:** 65 / 65 PASSED (100%)
- **Total Test Suite:** **75 / 75 PASSED**

---

## 7. Research Disclaimers & Boundaries

1. **Software Sensor Simulation:** All sensor telemetry data represents historical software replays from Dataset B (Berambadi Observatory). No physical IoT hardware is connected.
2. **Academic & Research Scope:** The system is intended as an explainable decision support prototype. Fertilizer dosages should be adjusted by local agronomists based on field topography and specific crop growth stages.
3. **Zero-Hallucination Grounding:** All recommendations undergo strict cosine relevance thresholding ($\ge 0.35$) against peer-reviewed FAO/USDA/ICAR literature before presentation.
