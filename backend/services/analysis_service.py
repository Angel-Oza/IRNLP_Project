"""
Soil Health Analysis Service Layer.

Bridges the FastAPI application with the Phase 7 SoilHealthIndexEngine.
"""

import os
import sys
from typing import Dict, Any, Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from soil_health.shi import SoilHealthIndexEngine, get_shi_engine
from backend.schemas import SoilAnalysisRequest, SoilAnalysisResponse, ParameterDetail


class SoilAnalysisService:
    """
    Service wrapper for Soil Health Index (SHI) calculations.
    """

    def __init__(self, engine: Optional[SoilHealthIndexEngine] = None):
        self.engine = engine or get_shi_engine()

    def analyze_soil(self, data: Dict[str, Optional[float]]) -> SoilAnalysisResponse:
        """
        Executes SHI evaluation over provided soil chemistry parameters.
        """
        # Filter out None values
        clean_sample = {k: float(v) for k, v in data.items() if v is not None}

        if not clean_sample:
            raise ValueError("At least one soil parameter must be provided for soil analysis.")

        # Run Phase 7 SHI Engine
        shi_explanation = self.engine.explain_shi(clean_sample)
        shi_eval = shi_explanation["overall_evaluation"]
        raw_param_details = shi_explanation["parameter_details"]

        # Format parameter details
        param_details_dict = {}
        for k, p in raw_param_details.items():
            meta = p["metadata"]
            param_details_dict[k] = ParameterDetail(
                raw_value=float(p["raw_value"]),
                score=float(p["score"]),
                status=str(p["status"]),
                is_deficient=bool(p["is_deficient"]),
                name=str(meta["name"]),
                unit=str(meta["unit"]),
                desirable_range=str(meta["desirable_range"]),
                provenance=str(meta["provenance"]),
            )

        return SoilAnalysisResponse(
            shi_score=float(shi_eval["shi_score"]),
            category=str(shi_eval["category"]),
            parameter_scores=shi_eval["parameter_scores"],
            normalized_weights=shi_eval["normalized_weights"],
            weighted_contributions=shi_eval["weighted_contributions"],
            parameter_details=param_details_dict,
            deficiencies=shi_eval["deficiencies"],
            strengths=shi_explanation["strengths"],
            limiting_factors=shi_explanation["limiting_factors"],
            agronomic_summary=shi_explanation["agronomic_summary"],
            warnings=shi_eval.get("warnings", []),
            provenance_summary=shi_eval.get("provenance_summary", "ICAR / FAO / USDA Standards"),
        )


_analysis_service_instance: Optional[SoilAnalysisService] = None


def get_analysis_service() -> SoilAnalysisService:
    global _analysis_service_instance
    if _analysis_service_instance is None:
        _analysis_service_instance = SoilAnalysisService()
    return _analysis_service_instance
