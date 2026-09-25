"""
Software Sensor Stream Simulator.

Replays historical in-situ environmental sensor telemetry from Dataset B (Berambadi Observatory, 2016-2025)
in deterministic chronological sequence for software-based research and evaluation.

NO PHYSICAL HARDWARE (ESP32 / Raspberry Pi / Arduino) IS USED.
This module provides a pure software data streaming abstraction.
"""

import os
import sys
from typing import Dict, List, Optional, Iterator, Union, Any

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pandas as pd
import numpy as np

from data.loader import DatasetLoader, get_data_loader


class TelemetryStreamSimulator:
    """
    Software Sensor Stream Simulator replaying historical in-situ temporal telemetry.
    """

    SENSOR_CHANNELS = [
        "Temp_5cm",
        "SM_5cm",
        "EC_5cm",
        "Temp_50cm",
        "SM_50cm",
        "EC_50cm",
        "Precipitation",
    ]

    def __init__(
        self,
        base_dir: Optional[str] = None,
        year: Optional[int] = 2016,
        years: Optional[List[int]] = None,
        playback_step: int = 1,
    ):
        """
        Initializes the simulator with historical telemetry records.
        
        Args:
            base_dir: Project root directory.
            year: Specific year to load (default: 2016 for standard benchmarking).
            years: Optional list of years to concatenate chronologically.
            playback_step: Step size for advancing through records (default: 1).
        """
        self.loader = get_data_loader(base_dir=base_dir)
        self.year = year
        self.years = years
        self.playback_step = max(1, playback_step)

        # Load historical telemetry
        self._raw_df = self.loader.load_sensor_telemetry_dataset(year=year, years=years)
        # Ensure chronological ordering
        self._df = self._raw_df.sort_values("timestamp").reset_index(drop=True)
        self.total_records = len(self._df)
        self._current_index = 0

    def reset(self) -> None:
        """Resets the playback position to the start of the stream."""
        self._current_index = 0

    def has_next(self) -> bool:
        """Checks if further telemetry frames are available."""
        return self._current_index < self.total_records

    def next_frame(self) -> Optional[Dict[str, Any]]:
        """
        Retrieves the next telemetry frame in chronological order.
        
        Returns:
            Dictionary containing timestamp and sensor channel values, or None if stream is exhausted.
        """
        if not self.has_next():
            return None

        row = self._df.iloc[self._current_index]
        self._current_index += self.playback_step

        frame = {
            "timestamp": str(row["timestamp"]),
            "index": int(self._current_index - self.playback_step),
        }
        for channel in self.SENSOR_CHANNELS:
            val = row.get(channel, np.nan)
            frame[channel] = float(val) if pd.notnull(val) else np.nan

        return frame

    def stream_generator(
        self, max_frames: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Generator yielding sensor frames chronologically.
        
        Args:
            max_frames: Optional upper limit on frames to stream.
        """
        count = 0
        while self.has_next():
            if max_frames is not None and count >= max_frames:
                break
            frame = self.next_frame()
            if frame is not None:
                yield frame
                count += 1

    def get_window(
        self, start_idx: int = 0, num_records: int = 100
    ) -> pd.DataFrame:
        """
        Extracts a slice/window of historical telemetry for batch experimentation.
        
        Args:
            start_idx: Starting index in the chronological series.
            num_records: Number of consecutive records to extract.
            
        Returns:
            DataFrame containing the requested window of sensor data.
        """
        end_idx = min(start_idx + num_records, self.total_records)
        window = self._df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
        return window

    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Computes baseline summary statistics (mean, std, min, max, missingness)
        for each sensor channel across the clean historical record.
        """
        stats = {}
        for col in self.SENSOR_CHANNELS:
            series = self._df[col].dropna()
            stats[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "p25": float(series.quantile(0.25)),
                "p75": float(series.quantile(0.75)),
                "missing_count": int(self._df[col].isna().sum()),
                "missing_pct": float(self._df[col].isna().mean() * 100),
            }
        return stats


def get_telemetry_simulator(
    base_dir: Optional[str] = None, year: int = 2016
) -> TelemetryStreamSimulator:
    """Helper factory to create TelemetryStreamSimulator."""
    return TelemetryStreamSimulator(base_dir=base_dir, year=year)


if __name__ == "__main__":
    sim = get_telemetry_simulator(year=2016)
    print(f"Initialized TelemetryStreamSimulator with {sim.total_records} records.")
    print("Sampling first 3 chronological frames:")
    for f in sim.stream_generator(max_frames=3):
        print(f"  Frame @ {f['timestamp']} | Temp_5cm: {f['Temp_5cm']:.2f}°C, SM_5cm: {f['SM_5cm']:.2f}%, EC_5cm: {f['EC_5cm']:.2f}")
