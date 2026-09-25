"""
Machine Learning and XAI Service Layer.

Bridges the FastAPI application with the Phase 2 trained ML baselines and Phase 3 TreeSHAP Explainer.
"""

import os
import sys
import pickle
from typing import Dict, List, Any, Optional
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from xai.service import ExplanationService, get_explanation_service
from backend.schemas import (
    CropPredictionRequest,
    CropPredictionResponse,
    FeatureAttributionItem,
    ClassProbability,
)


class MLCropPredictionService:
    """
    Service wrapper for Crop Suitability Prediction and TreeSHAP Explainability.
    """

    def __init__(self, explanation_service: Optional[ExplanationService] = None):
        self.exp_service = explanation_service or get_explanation_service(base_dir=BASE_DIR)
        self.preprocessor = self.exp_service.preprocessor
        self.model = self.exp_service.explainer.model
        self.class_names = self.preprocessor.class_names

    def predict_crop(self, features: Dict[str, float]) -> CropPredictionResponse:
        """
        Executes crop recommendation inference with TreeSHAP feature attributions and top probabilities.
        """
        # Validate required feature keys
        for feat in self.preprocessor.FEATURE_COLS:
            if feat not in features:
                raise ValueError(f"Missing required crop prediction feature: '{feat}'")

        # Run Phase 3 ExplanationService
        xai_result = self.exp_service.explain_reading(features)
        pred_info = xai_result["prediction"]
        ml_exp = xai_result["ml_feature_explanation"]

        # Scale features to compute full probability distribution
        scaled_vec = self.preprocessor.transform_single(features)
        top_probs: List[ClassProbability] = []

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(scaled_vec)[0]
            # Get top 5 classes sorted by probability
            top_indices = np.argsort(probs)[::-1][:5]
            for idx in top_indices:
                top_probs.append(
                    ClassProbability(
                        crop=self.class_names[idx],
                        probability=round(float(probs[idx]), 4),
                    )
                )
        else:
            top_probs.append(
                ClassProbability(
                    crop=pred_info["crop"],
                    probability=round(float(pred_info["confidence"]), 4),
                )
            )

        # Convert feature attributions
        attributions = []
        for attr in ml_exp["feature_attributions"]:
            magnitude = float(attr.get("magnitude", abs(attr.get("shap_value", 0.0))))
            attributions.append(
                FeatureAttributionItem(
                    feature=attr["feature"],
                    raw_value=float(attr["raw_value"]),
                    shap_value=float(attr["shap_value"]),
                    contribution=str(attr["contribution"]),
                    abs_importance=round(magnitude, 4),
                )
            )

        return CropPredictionResponse(
            predicted_crop=str(pred_info["crop"]),
            confidence=round(float(pred_info["confidence"]), 4),
            class_id=int(pred_info["class_id"]),
            top_probabilities=top_probs,
            base_expected_value=float(ml_exp["base_expected_value"]),
            feature_attributions=attributions,
            summary_statement=str(ml_exp["summary_statement"]),
            scope_note=(
                "TreeSHAP quantifies exact mathematical feature contributions toward the classifier's "
                "log-odds prediction. It is distinct from Soil Health Index (SHI) and RAG literature retrieval."
            ),
        )


_ml_service_instance: Optional[MLCropPredictionService] = None


def get_ml_service() -> MLCropPredictionService:
    global _ml_service_instance
    if _ml_service_instance is None:
        _ml_service_instance = MLCropPredictionService()
    return _ml_service_instance
