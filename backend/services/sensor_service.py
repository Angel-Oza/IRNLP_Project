"""
Software Sensor Telemetry Simulation & Reliability Service.

Bridges the FastAPI application with the Phase 6 TelemetryStreamSimulator and SensorReliabilityEstimator.
Operates purely on software replays of historical Dataset B (Berambadi Observatory).
"""

import os
import sys
from typing import Dict, List, Any, Optional
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from iot.simulator import TelemetryStreamSimulator, get_telemetry_simulator
from iot.reliability import SensorReliabilityEstimator, get_reliability_estimator
from backend.schemas import (
    SensorFrameSchema,
    ReliabilityDetailSchema,
    SensorStreamResponse,
)


class SensorSimulationService:
    """
    Service wrapper providing chronological software sensor stream playback and real-time reliability estimation.
    """

    def __init__(self, base_dir: Optional[str] = None, default_year: int = 2016):
        self.base_dir = os.path.abspath(base_dir or BASE_DIR)
        self.current_year = default_year
        self.simulator = get_telemetry_simulator(base_dir=self.base_dir, year=default_year)
        self.estimator = get_reliability_estimator()
        
        # Precompute channel mean and std for standardization
        self.stats = self.simulator.get_summary_statistics()
        self.channels = self.simulator.SENSOR_CHANNELS

    def get_stream_frames(
        self,
        count: int = 1,
        step: int = 1,
        year: Optional[int] = None,
        reset: bool = False,
    ) -> SensorStreamResponse:
        """
        Retrieves next batch of sensor frames and computes dynamic reliability score.
        """
        if year is not None and year != self.current_year:
            self.current_year = year
            self.simulator = get_telemetry_simulator(base_dir=self.base_dir, year=year)
            self.stats = self.simulator.get_summary_statistics()
            self.estimator.reset_state()

        if reset:
            self.simulator.reset()
            self.estimator.reset_state()

        self.simulator.playback_step = max(1, step)
        frames_out: List[SensorFrameSchema] = []
        last_reliability = 1.0
        last_subscores = {"s_range": 1.0, "s_dist": 1.0, "s_avail": 1.0, "s_step": 1.0}

        count = min(max(1, count), 100)  # Bound count between 1 and 100

        for _ in range(count):
            if not self.simulator.has_next():
                # Loop back around if at end of series
                self.simulator.reset()

            raw_frame = self.simulator.next_frame()
            if raw_frame is None:
                break

            # Standardize frame values for reliability estimation
            std_vals = []
            missing_mask = []
            for ch in self.channels:
                val = raw_frame.get(ch)
                if val is None or np.isnan(val):
                    std_vals.append(0.0)
                    missing_mask.append(True)
                else:
                    mean_val = self.stats[ch]["mean"]
                    std_val = max(self.stats[ch]["std"], 1e-4)
                    z = (float(val) - mean_val) / std_val
                    std_vals.append(z)
                    missing_mask.append(False)

            std_arr = np.array(std_vals, dtype=np.float32)
            mask_arr = np.array(missing_mask, dtype=bool)

            # Compute dynamic reliability
            last_reliability = float(self.estimator.estimate_single_online(std_arr, missing_mask=mask_arr))

            # Compute individual sub-scores for transparency
            excess_z = np.maximum(0.0, np.abs(std_arr) - self.estimator.range_threshold_z)
            range_penalty = np.mean((excess_z / 1.5) ** 2)
            s_range = float(np.exp(-range_penalty))

            d_sq = float(np.mean(std_arr ** 2))
            s_dist = float(1.0 / (1.0 + (max(0.0, d_sq - self.estimator.norm_threshold) ** 1.2)))
            s_avail = float((1.0 - float(np.mean(mask_arr))) ** 1.5)
            s_step = 1.0  # internal state captured in estimator

            last_subscores = {
                "s_range": round(s_range, 3),
                "s_dist": round(s_dist, 3),
                "s_avail": round(s_avail, 3),
                "s_step": round(max(0.0, min(1.0, last_reliability / max(1e-4, s_range * s_dist * s_avail))), 3),
            }

            frames_out.append(
                SensorFrameSchema(
                    timestamp=raw_frame["timestamp"],
                    index=raw_frame["index"],
                    Temp_5cm=None if np.isnan(raw_frame["Temp_5cm"]) else round(raw_frame["Temp_5cm"], 2),
                    SM_5cm=None if np.isnan(raw_frame["SM_5cm"]) else round(raw_frame["SM_5cm"], 2),
                    EC_5cm=None if np.isnan(raw_frame["EC_5cm"]) else round(raw_frame["EC_5cm"], 3),
                    Temp_50cm=None if np.isnan(raw_frame["Temp_50cm"]) else round(raw_frame["Temp_50cm"], 2),
                    SM_50cm=None if np.isnan(raw_frame["SM_50cm"]) else round(raw_frame["SM_50cm"], 2),
                    EC_50cm=None if np.isnan(raw_frame["EC_50cm"]) else round(raw_frame["EC_50cm"], 3),
                    Precipitation=None if np.isnan(raw_frame["Precipitation"]) else round(raw_frame["Precipitation"], 2),
                )
            )

        # Status categorization
        if last_reliability >= 0.85:
            status_label = "Optimal Telemetry (High Quality)"
        elif last_reliability >= 0.60:
            status_label = "Moderate Telemetry (Minor Drift/Noise)"
        elif last_reliability >= 0.35:
            status_label = "Degraded Telemetry (High Jitter/Missingness)"
        else:
            status_label = "Critical Sensor Fault (Severe Corruption)"

        reliability_detail = ReliabilityDetailSchema(
            composite_reliability=round(last_reliability, 4),
            s_range=last_subscores["s_range"],
            s_dist=last_subscores["s_dist"],
            s_avail=last_subscores["s_avail"],
            s_step=last_subscores["s_step"],
            status_label=status_label,
        )

        return SensorStreamResponse(
            frames=frames_out,
            reliability=reliability_detail,
            total_records=self.simulator.total_records,
            current_index=int(self.simulator._current_index),
            year=self.current_year,
            simulation_notice=(
                "Software Sensor Simulation (Dataset B Berambadi Replay) — "
                "No Physical IoT Hardware Connected."
            ),
        )


_sensor_service_instance: Optional[SensorSimulationService] = None


def get_sensor_service() -> SensorSimulationService:
    global _sensor_service_instance
    if _sensor_service_instance is None:
        _sensor_service_instance = SensorSimulationService()
    return _sensor_service_instance
