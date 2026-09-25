"""
Unit and Integration Tests for Explainable AI (SHAP) Module.
"""

import os
import pytest
import numpy as np
from xai.shap_explainer import SoilSHAPExplainer
from xai.service import ExplanationService, get_explanation_service


@pytest.fixture
def explainer():
    return SoilSHAPExplainer()


@pytest.fixture
def service():
    return get_explanation_service()


def test_explainer_initialization(explainer):
    assert explainer.model is not None
    assert explainer.explainer is not None
    assert len(explainer.feature_names) == 7
    assert len(explainer.class_names) == 22


def test_global_explanations(explainer):
    global_exp = explainer.compute_global_explanations(sample_size=100)
    assert "overall_feature_ranking" in global_exp
    assert "per_class_feature_importance" in global_exp

    ranking = global_exp["overall_feature_ranking"]
    assert len(ranking) == 7
    # Verify sorted descending
    scores = [r["mean_abs_shap"] for r in ranking]
    assert scores == sorted(scores, reverse=True)
    assert all(s >= 0.0 for s in scores)


def test_local_instance_explanation(explainer):
    sample_vec = explainer.X_test[0:1]
    exp = explainer.explain_instance(sample_vec)

    assert "predicted_class_id" in exp
    assert "predicted_class_name" in exp
    assert exp["predicted_class_name"] in explainer.class_names
    assert 0.0 <= exp["prediction_probability"] <= 1.0

    attrs = exp["feature_attributions"]
    assert len(attrs) == 7

    # Verify additivity property: sum(shap) + base_value ~= pred_prob
    shap_sum = sum(a["shap_value"] for a in attrs)
    reconstructed_prob = shap_sum + exp["base_expected_value"]
    np.testing.assert_allclose(reconstructed_prob, exp["prediction_probability"], atol=1e-2)


def test_explanation_service(service):
    raw_reading = {
        "N": 90.0,
        "P": 42.0,
        "K": 43.0,
        "temperature": 20.87,
        "humidity": 82.00,
        "ph": 6.50,
        "rainfall": 202.93,
    }

    res = service.explain_reading(raw_reading)

    assert "prediction" in res
    assert res["prediction"]["crop"] == "rice"
    assert res["prediction"]["confidence"] > 0.80

    assert "ml_feature_explanation" in res
    assert "feature_attributions" in res["ml_feature_explanation"]
    assert "summary_statement" in res["ml_feature_explanation"]

    assert "provenance_and_boundaries" in res
    assert "distinction_note" in res["provenance_and_boundaries"]
