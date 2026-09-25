"""
IoT Sensing, Simulation, Fault Injection, and Reliability Estimation Layer.

Software-only historical telemetry stream replay and controlled degradation engine.
"""

from iot.simulator import TelemetryStreamSimulator, get_telemetry_simulator
from iot.noise_simulator import SensorFaultInjector, get_fault_injector
from iot.reliability import SensorReliabilityEstimator, get_reliability_estimator
from iot.robustness_experiment import RobustnessExperimentRunner

__all__ = [
    "TelemetryStreamSimulator",
    "get_telemetry_simulator",
    "SensorFaultInjector",
    "get_fault_injector",
    "SensorReliabilityEstimator",
    "get_reliability_estimator",
    "RobustnessExperimentRunner",
]
