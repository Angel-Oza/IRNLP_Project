"""
Sensor Fault Injection and Synthetic Noise Generation Engine.

Applies mathematically rigorous, reproducible corruptions to sensor data:
1. Proportional Gaussian Noise: eta in {5%, 10%, 20%} scaled to feature standard deviations.
2. Progressive Sensor Calibration Drift: Linear / step offsets simulating physical sensor aging.
3. Intermittent Sensor Missingness / Channel Dropout: Random dropouts simulating telemetry loss.
4. Transient Spikes / Outliers: High-amplitude impulses simulating electrical glitches.

ALL OPERATIONS PRESERVE ORIGINAL DATA IMMUTABILITY AND USE REPRODUCIBLE RANDOM SEEDS.
"""

import copy
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd


class SensorFaultInjector:
    """
    Controlled Fault Injection Engine for Sensor Robustness Benchmarking.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def set_seed(self, seed: int) -> None:
        """Sets the internal random seed for deterministic reproducibility."""
        self.random_state = seed
        self.rng = np.random.RandomState(seed)

    def inject_gaussian_noise(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        noise_level: float = 0.05,
        feature_stds: Optional[Union[np.ndarray, Dict[str, float]]] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Injects zero-mean Gaussian noise scaled proportionally to each feature's standard deviation:
            x'_j = x_j + N(0, (noise_level * sigma_j)^2)
        
        Args:
            data: NumPy array (N, D) or Pandas DataFrame.
            noise_level: Proportion of feature std to inject (e.g., 0.05 for 5%, 0.10 for 10%, 0.20 for 20%).
            feature_stds: Optional precomputed std per feature. If None, computed from data.
            
        Returns:
            Corrupted copy of data.
        """
        if isinstance(data, pd.DataFrame):
            corrupted_df = data.copy()
            numeric_cols = corrupted_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                std = (
                    feature_stds[col]
                    if (feature_stds and col in feature_stds)
                    else corrupted_df[col].std(skipna=True)
                )
                if std == 0 or np.isnan(std):
                    std = 1.0
                noise = self.rng.normal(0.0, noise_level * std, size=len(corrupted_df))
                corrupted_df[col] = corrupted_df[col] + noise
            return corrupted_df

        elif isinstance(data, np.ndarray):
            corrupted_arr = data.copy()
            if feature_stds is None:
                stds = np.std(corrupted_arr, axis=0, keepdims=True)
                stds[stds == 0] = 1.0
            elif isinstance(feature_stds, np.ndarray):
                stds = feature_stds.reshape(1, -1)
            else:
                stds = np.array([feature_stds.get(i, 1.0) for i in range(data.shape[1])]).reshape(1, -1)

            noise = self.rng.normal(0.0, noise_level * stds, size=corrupted_arr.shape)
            return corrupted_arr + noise

        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def inject_sensor_drift(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        drift_strength: float = 1.5,
        channels: Optional[List[Union[str, int]]] = None,
        feature_stds: Optional[Union[np.ndarray, Dict[str, float]]] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Injects progressive linear calibration drift:
            x'_j(t) = x_j(t) + drift_strength * (t / T) * sigma_j
            
        Args:
            data: Input array or DataFrame.
            drift_strength: Maximum drift in units of feature standard deviation at the end of the sequence.
            channels: Specific channels/indices to drift. If None, all numeric features drift.
            feature_stds: Optional feature standard deviations.
            
        Returns:
            Corrupted copy of data.
        """
        if isinstance(data, pd.DataFrame):
            corrupted_df = data.copy()
            N = len(corrupted_df)
            t_normalized = np.linspace(0.0, 1.0, N)
            target_cols = channels if channels is not None else corrupted_df.select_dtypes(include=[np.number]).columns

            for col in target_cols:
                if col in corrupted_df.columns:
                    std = (
                        feature_stds[col]
                        if (feature_stds and col in feature_stds)
                        else corrupted_df[col].std(skipna=True)
                    )
                    if std == 0 or np.isnan(std):
                        std = 1.0
                    drift_vector = drift_strength * std * t_normalized
                    corrupted_df[col] = corrupted_df[col] + drift_vector
            return corrupted_df

        elif isinstance(data, np.ndarray):
            corrupted_arr = data.copy()
            N, D = corrupted_arr.shape
            t_normalized = np.linspace(0.0, 1.0, N)[:, np.newaxis]

            if feature_stds is None:
                stds = np.std(corrupted_arr, axis=0, keepdims=True)
                stds[stds == 0] = 1.0
            elif isinstance(feature_stds, np.ndarray):
                stds = feature_stds.reshape(1, -1)
            else:
                stds = np.ones((1, D))

            target_indices = channels if channels is not None else list(range(D))
            drift_matrix = np.zeros_like(corrupted_arr)
            for idx in target_indices:
                drift_matrix[:, idx] = (drift_strength * stds[0, idx] * t_normalized.ravel())

            return corrupted_arr + drift_matrix

        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def inject_missingness(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        missing_rate: float = 0.20,
        impute_value: float = 0.0,
        channels: Optional[List[Union[str, int]]] = None,
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], np.ndarray]:
        """
        Simulates telemetry communication loss by zeroing or setting NaN to random entries.
        
        Args:
            data: Input array or DataFrame.
            missing_rate: Probability of channel dropout per entry (e.g., 0.20 for 20%).
            impute_value: Value used for imputation fallback (e.g., 0.0 for scaled features or NaN for telemetry).
            channels: Specific channels/indices to drop out.
            
        Returns:
            Tuple of (corrupted_data, missing_mask).
        """
        if isinstance(data, pd.DataFrame):
            corrupted_df = data.copy()
            numeric_cols = channels if channels is not None else corrupted_df.select_dtypes(include=[np.number]).columns
            mask = self.rng.rand(len(corrupted_df), len(numeric_cols)) < missing_rate

            for i, col in enumerate(numeric_cols):
                if col in corrupted_df.columns:
                    col_mask = mask[:, i]
                    corrupted_df.loc[col_mask, col] = np.nan if np.isnan(impute_value) else impute_value

            return corrupted_df, mask

        elif isinstance(data, np.ndarray):
            corrupted_arr = data.copy()
            mask = self.rng.rand(*corrupted_arr.shape) < missing_rate
            if channels is not None:
                # Mask only designated channels
                channel_filter = np.zeros(corrupted_arr.shape[1], dtype=bool)
                for c in channels:
                    channel_filter[int(c)] = True
                mask = mask & channel_filter

            corrupted_arr[mask] = impute_value
            return corrupted_arr, mask

        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def inject_transient_spikes(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        spike_rate: float = 0.05,
        spike_magnitude: float = 3.5,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Injects sudden high-amplitude electrical glitch spikes.
        
        Args:
            data: Input array or DataFrame.
            spike_rate: Fraction of readings affected by transient spikes.
            spike_magnitude: Multiplier on feature std for spike amplitude.
            
        Returns:
            Corrupted copy of data.
        """
        if isinstance(data, np.ndarray):
            corrupted_arr = data.copy()
            stds = np.std(corrupted_arr, axis=0, keepdims=True)
            stds[stds == 0] = 1.0

            spike_mask = self.rng.rand(*corrupted_arr.shape) < spike_rate
            spike_signs = self.rng.choice([-1.0, 1.0], size=corrupted_arr.shape)
            spikes = spike_signs * spike_magnitude * stds * spike_mask

            return corrupted_arr + spikes

        elif isinstance(data, pd.DataFrame):
            corrupted_df = data.copy()
            numeric_cols = corrupted_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                std = corrupted_df[col].std(skipna=True)
                if std == 0 or np.isnan(std):
                    std = 1.0
                mask = self.rng.rand(len(corrupted_df)) < spike_rate
                signs = self.rng.choice([-1.0, 1.0], size=len(corrupted_df))
                corrupted_df.loc[mask, col] += signs[mask] * spike_magnitude * std
            return corrupted_df
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")


def get_fault_injector(random_state: int = 42) -> SensorFaultInjector:
    """Helper factory to create SensorFaultInjector."""
    return SensorFaultInjector(random_state=random_state)


if __name__ == "__main__":
    injector = get_fault_injector()
    clean_sample = np.array([[10.0, 20.0, 30.0], [12.0, 22.0, 32.0], [14.0, 24.0, 34.0]])
    print("Clean Sample:\n", clean_sample)

    noisy_5 = injector.inject_gaussian_noise(clean_sample, noise_level=0.05)
    print("\n5% Gaussian Noise Sample:\n", noisy_5)

    drifted = injector.inject_sensor_drift(clean_sample, drift_strength=1.0)
    print("\nDrifted Sample:\n", drifted)

    missing, mask = injector.inject_missingness(clean_sample, missing_rate=0.33)
    print("\nMissingness Sample (Imputed 0.0):\n", missing)
