"""
Unit and Integration Tests for Soil Health Index (SHI) Engine.
"""

import os
import pytest
import numpy as np
import pandas as pd

from soil_health.shi import SoilHealthIndexEngine, get_shi_engine
from data.loader import get_data_loader


@pytest.fixture
def shi_engine():
    return get_shi_engine()


# --------------------------------------------------------------------------
# 1. Parameter Scoring Tests
# --------------------------------------------------------------------------
def test_ph_scoring_optimum_and_extremes(shi_engine):
    # Optimal pH (6.0 - 7.5) -> Score 100.0
    score_opt, status_opt = shi_engine.score_ph(6.8)
    assert score_opt == 100.0
    assert "Optimal" in status_opt

    # Moderate acidity -> Score decreases
    score_acid, status_acid = shi_engine.score_ph(5.5)
    assert 0.0 < score_acid < 100.0
    assert "Acidic" in status_acid

    # Extreme acidity -> Near zero
    score_ext_acid, _ = shi_engine.score_ph(4.2)
    assert score_ext_acid < 30.0

    # Moderate alkalinity -> Score decreases
    score_alk, status_alk = shi_engine.score_ph(8.0)
    assert 0.0 < score_alk < 100.0
    assert "Alkaline" in status_alk

    # Invalid input
    score_inv, _ = shi_engine.score_ph(-1.0)
    assert score_inv == 0.0


def test_ec_scoring_salinity_curve(shi_engine):
    # Non-saline (< 0.8 dS/m) -> Score 100.0
    score_clean, status_clean = shi_engine.score_ec(0.25)
    assert score_clean == 100.0
    assert "Non-Saline" in status_clean or "Optimal" in status_clean

    # Slight salinity (0.8 - 1.6 dS/m)
    score_slight, _ = shi_engine.score_ec(1.2)
    assert 60.0 <= score_slight < 100.0

    # Severe salinity (> 3.0 dS/m)
    score_severe, status_severe = shi_engine.score_ec(4.5)
    assert score_severe <= 20.0
    assert "Saline" in status_severe


def test_organic_carbon_scoring(shi_engine):
    # Optimal OC (>= 0.75%)
    score_opt, status_opt = shi_engine.score_oc(0.85)
    assert score_opt >= 90.0
    assert "Optimal" in status_opt or "High" in status_opt

    # Medium OC (0.50 - 0.75%)
    score_med, _ = shi_engine.score_oc(0.60)
    assert 60.0 <= score_med <= 90.0

    # Critically low OC (< 0.50%)
    score_low, status_low = shi_engine.score_oc(0.20)
    assert score_low < 60.0
    assert "Low" in status_low or "Critically" in status_low


def test_macronutrients_npk_scoring(shi_engine):
    # Nitrogen (ICAR: 280 - 560 kg/ha is optimal)
    score_n_def, _ = shi_engine.score_nitrogen(120.0)
    assert score_n_def < 70.0
    score_n_opt, _ = shi_engine.score_nitrogen(350.0)
    assert score_n_opt >= 70.0

    # Phosphorus (ICAR: 10 - 25 kg/ha is optimal)
    score_p_def, _ = shi_engine.score_phosphorus(5.0)
    assert score_p_def < 60.0
    score_p_opt, _ = shi_engine.score_phosphorus(18.0)
    assert score_p_opt >= 60.0

    # Potassium (ICAR: 110 - 280 kg/ha is optimal)
    score_k_def, _ = shi_engine.score_potassium(80.0)
    assert score_k_def < 60.0
    score_k_opt, _ = shi_engine.score_potassium(200.0)
    assert score_k_opt >= 60.0


def test_micronutrients_critical_limits(shi_engine):
    # Zinc critical limit: 0.60 ppm
    score_zn_opt, _ = shi_engine.score_micronutrient("Zn", 1.2, 0.60)
    assert score_zn_opt == 100.0

    score_zn_def, status_zn = shi_engine.score_micronutrient("Zn", 0.30, 0.60)
    assert score_zn_def == pytest.approx(40.0, abs=1e-2)
    assert "Deficient" in status_zn


# --------------------------------------------------------------------------
# 2. Composite SHI & Dynamic Weighting Tests
# --------------------------------------------------------------------------
def test_weights_sum_to_one(shi_engine):
    total_weights = sum(shi_engine.DEFAULT_WEIGHTS.values())
    np.testing.assert_allclose(total_weights, 1.0, atol=1e-4)


def test_shi_bounds_and_determinism(shi_engine):
    sample = {
        "pH": 7.10,
        "EC": 0.15,
        "OC": 0.25,
        "N": 110.0,
        "P": 12.0,
        "K": 140.0,
        "S": 14.0,
        "Zn": 0.95,
        "B": 0.65,
        "Fe": 8.0,
        "Mn": 2.1,
        "Cu": 0.45,
    }

    res1 = shi_engine.calculate_shi(sample)
    res2 = shi_engine.calculate_shi(sample)

    assert 0.0 <= res1["shi_score"] <= 100.0
    assert res1["shi_score"] == res2["shi_score"]
    assert "category" in res1
    assert "weighted_contributions" in res1
    assert "provenance_summary" in res1


def test_partial_parameter_set_weight_renormalization(shi_engine):
    # Sample with only N, P, K, pH (IoT / Field subset)
    partial_sample = {
        "pH": 6.8,
        "N": 300.0,
        "P": 20.0,
        "K": 200.0,
    }

    res = shi_engine.calculate_shi(partial_sample)
    assert 0.0 <= res["shi_score"] <= 100.0
    assert len(res["warnings"]) > 0

    # Normalized weights must still sum to 1.0
    norm_w_sum = sum(res["normalized_weights"].values())
    np.testing.assert_allclose(norm_w_sum, 1.0, atol=1e-4)


def test_empty_or_invalid_sample_handling(shi_engine):
    res_empty = shi_engine.calculate_shi({})
    assert res_empty["shi_score"] == 0.0
    assert res_empty["category"] == "Insufficient Data"

    res_nan = shi_engine.calculate_shi({"pH": np.nan, "N": None})
    assert res_nan["shi_score"] == 0.0


# --------------------------------------------------------------------------
# 3. Explanation Service & Batch Evaluation Tests
# --------------------------------------------------------------------------
def test_explain_shi_structure(shi_engine):
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

    exp = shi_engine.explain_shi(sample)
    assert "overall_evaluation" in exp
    assert "strengths" in exp
    assert "limiting_factors" in exp
    assert "agronomic_summary" in exp
    assert len(exp["limiting_factors"]) > 0  # Low OC and N in Bareilly sample


def test_batch_scoring_on_dataset_a(shi_engine):
    loader = get_data_loader()
    df_chem = loader.load_soil_chemistry_dataset()

    assert len(df_chem) == 257
    df_scored = shi_engine.calculate_shi_batch(df_chem)

    assert "SHI_Score" in df_scored.columns
    assert "SHI_Category" in df_scored.columns
    assert not df_scored["SHI_Score"].isna().any()
    assert (df_scored["SHI_Score"] >= 0.0).all() and (df_scored["SHI_Score"] <= 100.0).all()
