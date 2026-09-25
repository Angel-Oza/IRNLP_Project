"""
Unit and Integration Tests for IoT Simulation, Fault Injection, Dynamic Reliability, and Experiment F.
"""

import os
import pytest
import numpy as np
import pandas as pd
import torch

from iot.simulator import TelemetryStreamSimulator, get_telemetry_simulator
from iot.noise_simulator import SensorFaultInjector, get_fault_injector
from iot.reliability import SensorReliabilityEstimator, get_reliability_estimator
from iot.robustness_experiment import RobustnessExperimentRunner
from fusion.gated_fusion import ReliabilityGatedFusionNet


@pytest.fixture
def simulator():
    return get_telemetry_simulator(year=2016)


@pytest.fixture
def fault_injector():
    return get_fault_injector(random_state=42)


@pytest.fixture
def reliability_estimator():
    return get_reliability_estimator()


# --------------------------------------------------------------------------
# 1. Telemetry Simulator Tests
# --------------------------------------------------------------------------
def test_simulator_initialization_and_chronology(simulator):
    assert simulator.total_records > 1000
    assert simulator.has_next() is True

    # Sample consecutive frames
    f1 = simulator.next_frame()
    f2 = simulator.next_frame()

    assert f1 is not None and f2 is not None
    assert f1["index"] == 0
    assert f2["index"] == 1
    assert pd.to_datetime(f1["timestamp"]) <= pd.to_datetime(f2["timestamp"])


def test_simulator_reset_and_determinism(simulator):
    simulator.reset()
    f_first_pass = [f for f in simulator.stream_generator(max_frames=5)]

    simulator.reset()
    f_second_pass = [f for f in simulator.stream_generator(max_frames=5)]

    assert len(f_first_pass) == 5
    assert len(f_second_pass) == 5
    for f1, f2 in zip(f_first_pass, f_second_pass):
        assert f1["timestamp"] == f2["timestamp"]
        assert f1["Temp_5cm"] == f2["Temp_5cm"]
        assert f1["SM_5cm"] == f2["SM_5cm"]


def test_simulator_window_extraction(simulator):
    window = simulator.get_window(start_idx=10, num_records=20)
    assert len(window) == 20
    assert "Temp_5cm" in window.columns
    assert "timestamp" in window.columns


def test_simulator_summary_statistics(simulator):
    stats = simulator.get_summary_statistics()
    assert "Temp_5cm" in stats
    assert "SM_5cm" in stats
    assert stats["Temp_5cm"]["mean"] > -10.0
    assert stats["SM_5cm"]["min"] >= 0.0


# --------------------------------------------------------------------------
# 2. Fault Injection Tests
# --------------------------------------------------------------------------
def test_fault_injector_immutability(fault_injector):
    clean_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    clean_copy = clean_arr.copy()

    _ = fault_injector.inject_gaussian_noise(clean_arr, noise_level=0.10)
    np.testing.assert_array_equal(clean_arr, clean_copy)

    _ = fault_injector.inject_sensor_drift(clean_arr, drift_strength=1.5)
    np.testing.assert_array_equal(clean_arr, clean_copy)

    _ = fault_injector.inject_missingness(clean_arr, missing_rate=0.50)
    np.testing.assert_array_equal(clean_arr, clean_copy)


def test_gaussian_noise_proportionality(fault_injector):
    clean_data = np.ones((100, 4), dtype=np.float32)
    stds = np.array([1.0, 2.0, 5.0, 10.0])

    noisy = fault_injector.inject_gaussian_noise(clean_data, noise_level=0.20, feature_stds=stds)
    diff = noisy - clean_data

    # Check that channel standard deviations scale proportionally
    diff_stds = np.std(diff, axis=0)
    assert diff_stds[0] < diff_stds[1] < diff_stds[2] < diff_stds[3]


def test_sensor_drift_linear_progression(fault_injector):
    clean_data = np.zeros((100, 2), dtype=np.float32)
    drifted = fault_injector.inject_sensor_drift(clean_data, drift_strength=2.0)

    # First row drift should be 0.0, last row drift should be ~2.0
    assert drifted[0, 0] == pytest.approx(0.0, abs=1e-3)
    assert drifted[-1, 0] == pytest.approx(2.0, abs=1e-2)
    # Drift must be monotonically non-decreasing
    assert (np.diff(drifted[:, 0]) >= 0.0).all()


def test_missingness_injection_and_mask(fault_injector):
    clean_data = np.ones((50, 4), dtype=np.float32)
    corrupted, mask = fault_injector.inject_missingness(clean_data, missing_rate=0.25, impute_value=-999.0)

    assert corrupted.shape == clean_data.shape
    assert mask.shape == clean_data.shape
    assert (corrupted[mask] == -999.0).all()
    assert (corrupted[~mask] == 1.0).all()


# --------------------------------------------------------------------------
# 3. Dynamic Reliability Estimation Tests
# --------------------------------------------------------------------------
def test_reliability_bounds(reliability_estimator):
    # Test random, extreme, and zero vectors
    for _ in range(50):
        vec = np.random.randn(7) * 5.0
        r = reliability_estimator.estimate_single_online(vec)
        assert 0.0 <= r <= 1.0


def test_reliability_response_clean_vs_corrupted(reliability_estimator):
    clean_vec = np.array([0.1, -0.2, 0.3, 0.0, -0.1, 0.2, -0.3], dtype=np.float32)
    r_clean = reliability_estimator.estimate_single_online(clean_vec)

    # Moderate noise
    noisy_vec = clean_vec + np.array([1.5, -2.0, 1.8, -1.2, 2.1, -1.5, 2.0])
    r_noisy = reliability_estimator.estimate_single_online(noisy_vec)

    # Extreme outlier
    extreme_vec = clean_vec + np.array([6.0, -7.0, 8.0, -9.0, 10.0, -8.0, 7.0])
    r_extreme = reliability_estimator.estimate_single_online(extreme_vec)

    assert r_clean > 0.85
    assert r_noisy < r_clean
    assert r_extreme < 0.20


def test_reliability_missingness_penalty(reliability_estimator):
    clean_vec = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    r_full = reliability_estimator.estimate_single_online(clean_vec, missing_mask=None)

    # 3 missing channels
    mask_3 = np.array([True, True, True, False, False, False, False], dtype=bool)
    r_partial = reliability_estimator.estimate_single_online(clean_vec, missing_mask=mask_3)

    # All missing channels
    mask_all = np.ones(7, dtype=bool)
    r_all_missing = reliability_estimator.estimate_single_online(clean_vec, missing_mask=mask_all)

    assert r_full > r_partial > r_all_missing
    assert r_all_missing <= 0.05


# --------------------------------------------------------------------------
# 4. Gated Fusion Integration & Robustness Runner Tests
# --------------------------------------------------------------------------
def test_dynamic_reliability_integration_with_fusion():
    model = ReliabilityGatedFusionNet(latent_dim=64, num_classes=22)
    x_clean = torch.tensor([[0.1, -0.2, 0.3, 0.0, -0.1, 0.2, -0.3]], dtype=torch.float32)
    x_corrupt = torch.tensor([[5.5, -6.2, 7.1, -5.0, 6.8, -7.2, 8.0]], dtype=torch.float32)
    z_text = torch.randn(1, 384)

    estimator = get_reliability_estimator()
    r_clean = torch.tensor([[estimator.estimate_single_online(x_clean.numpy()[0])]], dtype=torch.float32)
    r_corrupt = torch.tensor([[estimator.estimate_single_online(x_corrupt.numpy()[0])]], dtype=torch.float32)

    _, g_clean = model(x_clean, z_text, reliability=r_clean, return_gate=True)
    _, g_corrupt = model(x_corrupt, z_text, reliability=r_corrupt, return_gate=True)

    # Effective sensor weight under corruption must be substantially suppressed
    mean_g_clean = g_clean.mean().item()
    mean_g_corrupt = g_corrupt.mean().item()

    assert mean_g_clean > mean_g_corrupt


def test_robustness_experiment_runner_initialization():
    runner = RobustnessExperimentRunner()
    assert runner.X_test.shape == (440, 7)
    assert runner.Z_test.shape == (440, 384)
    assert runner.y_test.shape == (440,)
    assert "ReliabilityGatedFusionNet" in runner.models
    assert "SensorOnlyNet" in runner.models
