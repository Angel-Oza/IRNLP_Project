"""
FastAPI REST API Main Entrypoint.

Explainable Multi-Source Soil Health Intelligence System Using Machine Learning, RAG and Gated Fusion.
Phase 9 — Production Backend API.
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.schemas import (
    HealthResponse,
    SoilAnalysisRequest,
    SoilAnalysisResponse,
    CropPredictionRequest,
    CropPredictionResponse,
    AdvisoryRequest,
    AdvisoryResponse,
    UnifiedIntelligenceRequest,
    UnifiedIntelligenceResponse,
    SensorStreamResponse,
    KnowledgeCorpusResponse,
    SampleProfile,
)
from backend.services import (
    get_analysis_service,
    get_ml_service,
    get_sensor_service,
    get_advisory_service,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes and pre-warms all service singletons on startup."""
    print("🚀 Pre-warming Soil Health Intelligence Backend Services...")
    try:
        get_analysis_service()
        get_ml_service()
        get_sensor_service()
        get_advisory_service()
        print("✅ All research singletons (SHI, ML/XAI, FAISS RAG, Advisory, IoT) successfully initialized.")
    except Exception as e:
        print(f"⚠️ Warning during service pre-warm: {e}")
    yield
    print("🛑 Soil Health Intelligence Backend shutdown complete.")


app = FastAPI(
    title="Soil Health Intelligence System API",
    description=(
        "Explainable Multi-Source Soil Health Intelligence System Using Machine Learning, "
        "RAG and Gated Fusion (Phases 1–9 REST Interface)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local React/Vite development and dashboard integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------------
# 1. Health Check Endpoint
# ------------------------------------------------------------------------------
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Backend Health & Subsystem Readiness Check",
)
def health_check() -> HealthResponse:
    """Returns backend status and readiness of all research subsystems."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        components={
            "shi_engine": "READY (ICAR/FAO Standards)",
            "ml_crop_classifier": "READY (Random Forest Baseline, N=22 classes)",
            "shap_explainer": "READY (TreeSHAP Explainer)",
            "rag_retriever": "READY (FAISS MiniLM Index, 48 Chunks)",
            "advisory_generator": "READY (Evidence-Grounded Synthesizer)",
            "sensor_simulator": "READY (Software Telemetry Replay)",
        },
    )


# ------------------------------------------------------------------------------
# 2. Soil Health Analysis Endpoint (Phase 7)
# ------------------------------------------------------------------------------
@app.post(
    "/api/analyze",
    response_model=SoilAnalysisResponse,
    tags=["Soil Analysis"],
    summary="Compute Soil Health Index (SHI) and Parameter Scores",
)
def analyze_soil(request: SoilAnalysisRequest) -> SoilAnalysisResponse:
    """
    Evaluates soil health based strictly on ICAR/FAO/USDA agronomic response standards.
    Accepts: pH, EC, OC, N, P, K, S, Zn, B, Fe, Mn, Cu.
    Returns: Composite SHI (0-100), parameter scores, deficiencies, and limiting factors.
    """
    try:
        service = get_analysis_service()
        data_dict = request.model_dump()
        return service.analyze_soil(data_dict)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Soil Analysis error: {str(e)}",
        )


# ------------------------------------------------------------------------------
# 3. Crop Prediction & XAI Endpoint (Phase 2 & 3)
# ------------------------------------------------------------------------------
@app.post(
    "/api/predict",
    response_model=CropPredictionResponse,
    tags=["Crop Prediction"],
    summary="Supervised Crop Suitability Prediction with TreeSHAP XAI",
)
def predict_crop(request: CropPredictionRequest) -> CropPredictionResponse:
    """
    Predicts optimal crop recommendation and computes exact TreeSHAP feature attributions.
    Accepts: N, P, K, temperature, humidity, ph, rainfall.
    Returns: Predicted crop, confidence, class probabilities, and positive/negative drivers.
    """
    try:
        service = get_ml_service()
        feature_dict = request.model_dump()
        return service.predict_crop(feature_dict)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Crop Prediction error: {str(e)}",
        )


# ------------------------------------------------------------------------------
# 4. Grounded Agricultural Advisory Endpoint (Phase 8)
# ------------------------------------------------------------------------------
@app.post(
    "/api/advisory",
    response_model=AdvisoryResponse,
    tags=["Advisory"],
    summary="Evidence-Grounded Agricultural Advisory Generation",
)
def generate_advisory(request: AdvisoryRequest) -> AdvisoryResponse:
    """
    Generates actionable remediation recommendations backed by scientific literature from the FAO/ICAR RAG index.
    Filters recommendations with cosine similarity >= 0.35 threshold.
    """
    try:
        service = get_advisory_service()
        return service.generate_advisory(
            soil_data=request.soil_data,
            target_crop=request.target_crop,
            crop_prediction_data=request.crop_prediction_data,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Advisory Generation error: {str(e)}",
        )


# ------------------------------------------------------------------------------
# 5. Complete Multi-Source Unified Intelligence Pipeline Endpoint
# ------------------------------------------------------------------------------
@app.post(
    "/api/intelligence",
    response_model=UnifiedIntelligenceResponse,
    tags=["Unified Intelligence"],
    summary="End-to-End Multi-Source Intelligence Pipeline",
)
def complete_intelligence_pipeline(request: UnifiedIntelligenceRequest) -> UnifiedIntelligenceResponse:
    """
    Executes the entire unified pipeline:
    Input Data -> Preprocessing -> ML Crop Prediction -> SHI Evaluation -> TreeSHAP XAI -> RAG Retrieval -> Grounded Advisory.
    """
    try:
        service = get_advisory_service()
        return service.run_unified_intelligence(request)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Intelligence Pipeline error: {str(e)}",
        )


# ------------------------------------------------------------------------------
# 6. Software Sensor Telemetry Stream Simulation Endpoint (Phase 6)
# ------------------------------------------------------------------------------
@app.get(
    "/api/sensors",
    response_model=SensorStreamResponse,
    tags=["Sensors"],
    summary="Software Sensor Telemetry Stream & Dynamic Reliability",
)
def get_sensor_stream(
    count: int = Query(1, ge=1, le=100, description="Number of consecutive frames to retrieve"),
    step: int = Query(1, ge=1, le=50, description="Chronological playback step size"),
    year: int = Query(2016, description="Dataset B historical telemetry year (2016-2025)"),
    reset: bool = Query(False, description="Reset stream playback index to beginning"),
) -> SensorStreamResponse:
    """
    Replays historical in-situ environmental sensor telemetry from Dataset B (Berambadi Observatory)
    and evaluates online dynamic reliability score r(t) in [0, 1].
    NOTE: Software simulation only — no physical IoT hardware connected.
    """
    try:
        service = get_sensor_service()
        return service.get_stream_frames(count=count, step=step, year=year, reset=reset)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Sensor Simulation error: {str(e)}",
        )


# ------------------------------------------------------------------------------
# 7. Curated Scientific Corpus Browser Endpoint
# ------------------------------------------------------------------------------
@app.get(
    "/api/corpus",
    response_model=KnowledgeCorpusResponse,
    tags=["Knowledge"],
    summary="Browse Curated FAO/USDA/ICAR Scientific Knowledge Base",
)
def get_knowledge_corpus() -> KnowledgeCorpusResponse:
    """Returns the curated agricultural literature documents and chunks indexed in FAISS."""
    try:
        service = get_advisory_service()
        return service.get_knowledge_corpus()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Corpus retrieval error: {str(e)}",
        )


# ------------------------------------------------------------------------------
# 8. Empirical Benchmark Sample Profiles Endpoint
# ------------------------------------------------------------------------------
@app.get(
    "/api/samples",
    response_model=List[SampleProfile],
    tags=["Knowledge"],
    summary="Retrieve Empirical Benchmark Soil Sample Profiles",
)
def get_sample_profiles() -> List[SampleProfile]:
    """Returns curated benchmark sample profiles from Dataset A and C for instant testing."""
    try:
        service = get_advisory_service()
        return service.get_sample_profiles()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Sample retrieval error: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
