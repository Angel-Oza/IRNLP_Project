"""
Unit and Integration Tests for Grounded Agronomic Advisory Generator.
"""

import os
import pytest
import numpy as np
import pandas as pd

from advisory.generator import AgronomicAdvisoryGenerator, get_advisory_generator
from advisory.schemas import AgronomicAdvisoryPayload


@pytest.fixture
def advisory_generator():
    return get_advisory_generator()


def test_advisory_generation_full_sample(advisory_generator):
    sample = {
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

    advisory = advisory_generator.generate_advisory(sample)

    assert isinstance(advisory, AgronomicAdvisoryPayload)
    assert 0.0 <= advisory.soil_health_summary.shi_score <= 100.0
    assert len(advisory.recommendations) > 0

    # Verify detected deficiencies for Bareilly sample (Low OC and Low N)
    topics = [r.topic for r in advisory.recommendations]
    assert any("Organic Carbon" in t for t in topics)
    assert any("Nitrogen" in t for t in topics)

    # Verify recommendations have supporting evidence
    for rec in advisory.recommendations:
        assert rec.validation_status in ["VALIDATED", "LIMITED_EVIDENCE"]
        assert len(rec.supporting_evidence) > 0
        assert rec.supporting_evidence[0].relevance_score > 0.30
        assert "FAO" in rec.supporting_evidence[0].source_authority or "ICAR" in rec.supporting_evidence[0].source_authority


def test_advisory_with_crop_prediction_and_xai(advisory_generator):
    sample = {
        "pH": 6.5,
        "EC": 0.2,
        "OC": 0.8,
        "N": 90.0,
        "P": 42.0,
        "K": 43.0,
        "S": 12.0,
        "Zn": 1.0,
        "B": 0.7,
        "Fe": 8.0,
        "Mn": 2.2,
        "Cu": 0.5,
    }

    mock_ml_data = {
        "prediction": {"crop": "rice", "confidence": 0.92, "class_id": 0},
        "ml_feature_explanation": {
            "feature_attributions": [
                {"feature": "rainfall", "shap_value": 0.45, "contribution": "positive"},
                {"feature": "humidity", "shap_value": 0.32, "contribution": "positive"},
                {"feature": "temperature", "shap_value": -0.15, "contribution": "negative"},
            ]
        }
    }

    advisory = advisory_generator.generate_advisory(sample, crop_prediction_data=mock_ml_data)

    assert advisory.crop_prediction is not None
    assert advisory.crop_prediction.predicted_crop == "rice"
    assert advisory.crop_prediction.confidence == 0.92
    assert len(advisory.crop_prediction.key_positive_drivers) > 0


def test_advisory_partial_parameters(advisory_generator):
    # Only NPK and pH available
    partial_sample = {
        "pH": 5.2,
        "N": 120.0,
        "P": 8.0,
        "K": 200.0,
    }

    advisory = advisory_generator.generate_advisory(partial_sample)

    assert advisory.soil_health_summary.shi_score > 0.0
    assert len(advisory.disclaimers_and_limitations) > 0
    # Must identify acidic pH and low N and low P
    topics = [r.topic for r in advisory.recommendations]
    assert any("Acidity" in t for t in topics)
    assert any("Nitrogen" in t for t in topics)
    assert any("Phosphorus" in t for t in topics)


def test_no_data_leakage_in_advisory_generation(advisory_generator):
    sample = {"pH": 7.0, "OC": 0.8, "N": 350.0, "P": 20.0, "K": 200.0}
    advisory = advisory_generator.generate_advisory(sample)

    # Serialized dictionary must not leak internal training labels
    d = advisory.to_dict()
    assert "label_encoder" not in str(d)
    assert "X_train" not in str(d)
