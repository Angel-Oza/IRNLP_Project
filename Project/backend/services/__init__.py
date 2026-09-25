"""
Services package initialization.
Exports singletons and factory functions.
"""

from backend.services.analysis_service import SoilAnalysisService, get_analysis_service
from backend.services.ml_service import MLCropPredictionService, get_ml_service
from backend.services.sensor_service import SensorSimulationService, get_sensor_service
from backend.services.advisory_service import AdvisoryService, get_advisory_service

__all__ = [
    "SoilAnalysisService",
    "get_analysis_service",
    "MLCropPredictionService",
    "get_ml_service",
    "SensorSimulationService",
    "get_sensor_service",
    "AdvisoryService",
    "get_advisory_service",
]
