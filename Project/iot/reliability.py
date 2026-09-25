"""
Dynamic Sensor Stream Reliability Estimation Engine.

Computes a real-time reliability score r(t) in [0, 1] conditioned strictly on
online-observable telemetry signals WITHOUT accessing ground-truth clean values.

Online Signals Monitored:
1. Physical / Statistical Range Compliance (s_range): Penalizes out-of-distribution extremes (|z| > 2.5).
2. Multivariate Distribution Plausibility (s_dist): Penalizes excessive squared Mahalanobis / standard norm.
3. Channel Availability / Missingness Penalty (s_avail): Penalizes dropout or flatline/imputed values.
4. High-Frequency Temporal Jitter / Step Discontinuity (s_step): Penalizes erratic physical spikes.

Also provides offline diagnostic tools for scientific evaluation of reliability response.
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd


class SensorReliabilityEstimator:
    """
    Online Sensor Reliability Estimator evaluating data quality metrics in real-time.
    """

    def __init__(
        self,
        num_features: int = 7,
        range_threshold_z: float = 2.5,
        norm_threshold: float = 1.2,
        min_reliability: float = 0.01,
        max_reliability: float = 1.0,
    ):
        """
        Args:
            num_features: Dimensionality of standardized sensor feature vector (default: 7).
            range_threshold_z: Z-score threshold beyond which values incur range penalties.
            norm_threshold: Baseline normalized variance threshold.
            min_reliability: Lower bound floor for numerical stability.
            max_reliability: Maximum upper bound.
        """
        self.num_features = num_features
        self.range_threshold_z = range_threshold_z
        self.norm_threshold = norm_threshold
        self.min_reliability = min_reliability
        self.max_reliability = max_reliability

        # Online state tracking for sequential telemetry
        self._prev_reading: Optional[np.ndarray] = None

    def reset_state(self) -> None:
        """Resets sequential tracking history."""
        self._prev_reading = None

    def estimate_single_online(
        self,
        standardized_vector: np.ndarray,
        missing_mask: Optional[np.ndarray] = None,
    ) -> float:
        """
        Computes online dynamic reliability score r in [0, 1] for a single standardized reading.
        
        Args:
            standardized_vector: 1D array of standardized feature values (mean=0, std=1 nominal).
            missing_mask: Optional boolean array indicating missing/imputed channels.
            
        Returns:
            Scalar reliability score r in [0, 1].
        """
        vec = np.asarray(standardized_vector, dtype=np.float32).ravel()
        D = len(vec)

        # 1. Range Validity Score (s_range)
        # Penalize individual features that deviate far from standard normal bounds
        excess_z = np.maximum(0.0, np.abs(vec) - self.range_threshold_z)
        range_penalty = np.mean((excess_z / 1.5) ** 2)
        s_range = float(np.exp(-range_penalty))

        # 2. Distributional Plausibility (s_dist)
        # Average squared standardized norm: Nominal E[d^2] = 1.0 for clean data
        d_sq = float(np.mean(vec ** 2))
        if d_sq > self.norm_threshold:
            excess_norm = d_sq - self.norm_threshold
            s_dist = float(1.0 / (1.0 + (excess_norm ** 1.2)))
        else:
            s_dist = 1.0

        # 3. Channel Availability Score (s_avail)
        if missing_mask is not None:
            missing_ratio = float(np.mean(missing_mask))
            s_avail = float((1.0 - missing_ratio) ** 1.5)
        else:
            s_avail = 1.0

        # 4. Temporal Step Discontinuity Score (s_step)
        if self._prev_reading is not None and len(self._prev_reading) == D:
            delta = np.linalg.norm(vec - self._prev_reading) / np.sqrt(D)
            # Moderate continuous changes are normal; sudden massive jumps (>3.0 sigma) are penalized
            excess_jump = max(0.0, delta - 2.5)
            s_step = float(np.exp(-excess_jump))
        else:
            s_step = 1.0

        # Update historical state
        self._prev_reading = vec.copy()

        # Multiplicative composite reliability
        r = s_range * s_dist * s_avail * s_step
        r_bounded = float(np.clip(r, self.min_reliability, self.max_reliability))
        return r_bounded

    def estimate_batch(
        self,
        X_standardized: np.ndarray,
        missing_masks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Computes reliability scores for a batch of standardized instances.
        
        Args:
            X_standardized: 2D array of shape (N, D).
            missing_masks: Optional boolean array of shape (N, D) or (N,).
            
        Returns:
            1D array of reliability scores of shape (N,).
        """
        N, D = X_standardized.shape
        reliabilities = np.zeros(N, dtype=np.float32)

        for i in range(N):
            mask_i = None
            if missing_masks is not None:
                if missing_masks.ndim == 2:
                    mask_i = missing_masks[i]
                elif missing_masks.ndim == 1:
                    mask_i = missing_masks[i]
            reliabilities[i] = self.estimate_single_online(
                X_standardized[i], missing_mask=mask_i
            )

        return reliabilities

    # --------------------------------------------------------------------------
    # OFFLINE DIAGNOSTIC EVALUATION (Used for Experiment Validation Only)
    # --------------------------------------------------------------------------
    @staticmethod
    def compute_reliability_diagnostics(
        reliabilities: np.ndarray,
        corruption_label: str = "nominal",
    ) -> Dict[str, Any]:
        """
        Computes aggregate statistical diagnostics for evaluation reports.
        """
        r_arr = np.asarray(reliabilities, dtype=np.float32)
        return {
            "corruption_condition": corruption_label,
            "mean_reliability": float(np.mean(r_arr)),
            "median_reliability": float(np.median(r_arr)),
            "std_reliability": float(np.std(r_arr)),
            "min_reliability": float(np.min(r_arr)),
            "max_reliability": float(np.max(r_arr)),
            "p25_reliability": float(np.percentile(r_arr, 25)),
            "p75_reliability": float(np.percentile(r_arr, 75)),
            "sample_count": len(r_arr),
        }


def get_reliability_estimator() -> SensorReliabilityEstimator:
    """Helper factory for SensorReliabilityEstimator."""
    return SensorReliabilityEstimator()


if __name__ == "__main__":
    estimator = get_reliability_estimator()

    # Clean standardized sample (mean 0, std 1)
    clean_vec = np.array([0.2, -0.4, 0.8, -0.1, 0.5, -0.3, 0.1])
    r_clean = estimator.estimate_single_online(clean_vec)
    print(f"Clean Reading Reliability: {r_clean:.4f}")

    # Corrupted noisy sample
    noisy_vec = np.array([3.8, -4.2, 5.1, -0.1, 4.5, -3.9, 6.2])
    r_noisy = estimator.estimate_single_online(noisy_vec)
    print(f"Noisy Corrupted Reading Reliability: {r_noisy:.4f}")

    # Missingness sample
    missing_mask = np.array([False, True, True, False, True, False, False])
    r_missing = estimator.estimate_single_online(clean_vec, missing_mask=missing_mask)
    print(f"Missing Channel Reading Reliability: {r_missing:.4f}")
