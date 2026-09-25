"""
XAI Explanation Service Layer.

Provides high-level programmatic interface for generating structured, auditable
explanations for backend API endpoints and web dashboard visualizations.

Enforces clear semantic separation:
ML Feature Attribution (SHAP) != RAG Evidence Retrieval != LLM Generative Reasoning.
"""

import os
import json
from typing import Dict, List, Any, Optional
import numpy as np

from xai.shap_explainer import SoilSHAPExplainer
from data.preprocessor import CropDataPreprocessor


class ExplanationService:
    """
    High-level Explanation Service integrating Preprocessor, Model, and SHAP.
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = os.path.abspath(base_dir or os.path.join(os.path.dirname(__file__), ".."))
        self.preprocessor = CropDataPreprocessor(
            output_dir=os.path.join(self.base_dir, "data", "processed")
        )
        self.preprocessor.load_saved_artifacts()

        self.explainer = SoilSHAPExplainer(
            model_path=os.path.join(self.base_dir, "models", "checkpoints", "best_model.pkl"),
            data_dir=os.path.join(self.base_dir, "data", "processed"),
        )

    def get_global_importance(self) -> Dict[str, Any]:
        """Returns global feature importance rankings."""
        return self.explainer.compute_global_explanations()

    def explain_reading(self, raw_features: Dict[str, float]) -> Dict[str, Any]:
        """
        Takes raw physical agronomic readings, scales them, executes model inference,
        and generates an interpretable multi-tier explanation.
        """
        # Validate input features
        for f in self.preprocessor.FEATURE_COLS:
            if f not in raw_features:
                raise ValueError(f"Missing required feature: {f}")

        # Scale features using train-fitted scaler
        scaled_vec = self.preprocessor.transform_single(raw_features)

        # Generate local SHAP explanation
        local_exp = self.explainer.explain_instance(
            feature_vector=scaled_vec,
            raw_feature_dict=raw_features,
        )

        pred_crop = local_exp["predicted_class_name"]
        pred_prob = local_exp["prediction_probability"]
        attrs = local_exp["feature_attributions"]

        top_positive = [a for a in attrs if a["contribution"] == "positive"][:3]
        top_negative = [a for a in attrs if a["contribution"] == "negative"][:2]

        # Construct concise agronomic explanation summary
        pos_summary = ", ".join([f"{a['feature']} ({a['raw_value']:.1f})" for a in top_positive])
        neg_summary = ", ".join([f"{a['feature']} ({a['raw_value']:.1f})" for a in top_negative])

        summary_text = (
            f"The model predicted '{pred_crop.upper()}' with {pred_prob * 100:.1f}% confidence. "
            f"Key factors driving this prediction positively were: {pos_summary}. "
        )
        if top_negative:
            summary_text += f"Factors providing downward pressure: {neg_summary}."

        return {
            "prediction": {
                "crop": pred_crop,
                "confidence": pred_prob,
                "class_id": local_exp["predicted_class_id"],
            },
            "ml_feature_explanation": {
                "explainer_type": "TreeSHAP (Path-dependent Shapley values)",
                "base_expected_value": local_exp["base_expected_value"],
                "feature_attributions": attrs,
                "summary_statement": summary_text,
            },
            "provenance_and_boundaries": {
                "scope": "ML Feature Attribution (Explains mathematical classifier behavior)",
                "distinction_note": (
                    "This explanation quantifies how input features altered the ML classifier's log-odds. "
                    "It is mathematically distinct from external agronomic evidence retrieved by RAG "
                    "and biological reasoning synthesized by the LLM."
                ),
            },
            "input_reading_raw": raw_features,
        }


def get_explanation_service(base_dir: Optional[str] = None) -> ExplanationService:
    """Helper factory for ExplanationService."""
    return ExplanationService(base_dir=base_dir)


if __name__ == "__main__":
    service = get_explanation_service()
    sample_reading = {
        "N": 90.0,
        "P": 42.0,
        "K": 43.0,
        "temperature": 20.87,
        "humidity": 82.00,
        "ph": 6.50,
        "rainfall": 202.93,
    }
    res = service.explain_reading(sample_reading)
    print(json.dumps(res, indent=2))
