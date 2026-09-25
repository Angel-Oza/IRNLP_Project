"""
Unit and Integration Tests for Phase 9 FastAPI Backend REST API.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify health check endpoint returns 200 and subsystem readiness."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "1.0.0"
    assert "components" in data
    assert "shi_engine" in data["components"]
    assert "ml_crop_classifier" in data["components"]
    assert "rag_retriever" in data["components"]


def test_analyze_soil_valid():
    """Verify soil analysis endpoint with full valid parameter set."""
    payload = {
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
        "Cu": 0.47,
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "shi_score" in data
    assert 0.0 <= data["shi_score"] <= 100.0
    assert "category" in data
    assert "parameter_scores" in data
    assert "parameter_details" in data
    assert "pH" in data["parameter_details"]
    assert "deficiencies" in data
    assert len(data["deficiencies"]) > 0  # Low OC and N should trigger deficiencies


def test_analyze_soil_partial():
    """Verify soil analysis endpoint with partial parameter set and dynamic weight normalization."""
    payload = {
        "pH": 6.5,
        "N": 350.0,
        "P": 20.0,
        "K": 220.0,
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "shi_score" in data
    assert len(data["parameter_scores"]) == 4
    assert len(data["warnings"]) > 0  # Should warn about partial parameters


def test_analyze_soil_empty_invalid():
    """Verify soil analysis endpoint rejects empty or invalid payload."""
    payload = {}
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 400
    assert "At least one soil parameter" in response.json()["detail"]


def test_predict_crop_valid():
    """Verify ML crop suitability prediction and TreeSHAP feature attributions."""
    payload = {
        "N": 90.0,
        "P": 42.0,
        "K": 43.0,
        "temperature": 20.87,
        "humidity": 82.00,
        "ph": 6.50,
        "rainfall": 202.93,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_crop" in data
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert "feature_attributions" in data
    assert len(data["feature_attributions"]) == 7
    assert "top_probabilities" in data
    assert len(data["top_probabilities"]) > 0


def test_predict_crop_missing_fields():
    """Verify ML crop prediction fails on missing required features."""
    payload = {
        "N": 90.0,
        "P": 42.0,
        # missing K, temperature, humidity, ph, rainfall
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


def test_advisory_endpoint():
    """Verify grounded agricultural advisory generation and scientific evidence retrieval."""
    payload = {
        "soil_data": {
            "pH": 7.13,
            "EC": 0.13,
            "OC": 0.23,
            "N": 108.0,
            "P": 11.59,
            "K": 130.0,
        },
        "target_crop": "rice",
    }
    response = client.post("/api/advisory", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert len(data["recommendations"]) > 0
    assert "evidence_validation" in data
    assert data["evidence_validation"]["grounding_pass_rate"] >= 80.0
    
    # Check structure of first recommendation
    rec = data["recommendations"][0]
    assert "topic" in rec
    assert "action" in rec
    assert "urgency" in rec
    assert "validation_status" in rec
    assert len(rec["supporting_evidence"]) > 0
    assert rec["supporting_evidence"][0]["relevance_score"] >= 0.35


def test_complete_intelligence_endpoint():
    """Verify end-to-end unified intelligence pipeline integrating all research stages."""
    payload = {
        "pH": 6.8,
        "EC": 0.3,
        "OC": 0.85,
        "N": 120.0,
        "P": 35.0,
        "K": 180.0,
        "S": 15.0,
        "temperature": 24.5,
        "humidity": 78.0,
        "rainfall": 160.0,
    }
    response = client.post("/api/intelligence", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "pipeline_stages" in data
    assert len(data["pipeline_stages"]) == 5
    assert "soil_health" in data
    assert "crop_prediction" in data
    assert "advisory" in data
    assert "provenance" in data


def test_sensor_simulation_endpoint():
    """Verify software sensor telemetry replay and online reliability estimation."""
    response = client.get("/api/sensors?count=5&step=1&year=2016")
    assert response.status_code == 200
    data = response.json()
    assert "frames" in data
    assert len(data["frames"]) == 5
    assert "reliability" in data
    assert 0.0 <= data["reliability"]["composite_reliability"] <= 1.0
    assert "simulation_notice" in data
    assert "Software Sensor Simulation" in data["simulation_notice"]


def test_corpus_and_samples_endpoint():
    """Verify knowledge corpus metadata and benchmark sample profiles retrieval."""
    # Test Corpus Endpoint
    corpus_res = client.get("/api/corpus")
    assert corpus_res.status_code == 200
    corpus_data = corpus_res.json()
    assert corpus_data["total_chunks"] >= 30
    assert corpus_data["total_chunks"] == len(corpus_data["chunks"])
    assert len(corpus_data["authorities"]) >= 3
    assert len(corpus_data["documents"]) >= 5

    # Test Samples Endpoint
    samples_res = client.get("/api/samples")
    assert samples_res.status_code == 200
    samples_data = samples_res.json()
    assert len(samples_data) >= 3
    assert "soil_data" in samples_data[0]
    assert "crop_features" in samples_data[0]
