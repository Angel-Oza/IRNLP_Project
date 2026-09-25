"""
SHAP Explainability Module for Explainable Multi-Source Soil Health Intelligence System.

Implements TreeSHAP for tree-based ensemble models (RandomForest, XGBoost).
Provides:
1. Global Feature Attribution across the entire dataset.
2. Per-Class Feature Attribution profiles.
3. Local Feature Attribution for individual prediction instances.
4. Publication-quality XAI visualization exports.
"""

import os
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import shap


class SoilSHAPExplainer:
    """
    SHAP Feature Attribution Explainer for Agronomic Predictive Models.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        data_dir: Optional[str] = None,
        output_plot_dir: Optional[str] = None,
        output_metrics_dir: Optional[str] = None,
    ):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.model_path = model_path or os.path.join(base_dir, "models", "checkpoints", "best_model.pkl")
        self.data_dir = data_dir or os.path.join(base_dir, "data", "processed")
        self.plot_dir = output_plot_dir or os.path.join(base_dir, "results", "plots", "xai")
        self.metrics_dir = output_metrics_dir or os.path.join(base_dir, "results", "metrics")

        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Trained model checkpoint not found at: {self.model_path}")

        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(os.path.join(self.data_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)

        self.feature_names: List[str] = self.metadata["feature_names"]
        self.class_names: List[str] = self.metadata["class_names"]
        self.num_classes: int = len(self.class_names)
        self.num_features: int = len(self.feature_names)

        # Initialize TreeExplainer
        self.explainer = shap.TreeExplainer(self.model)

        # Load background reference data
        self.X_train = np.load(os.path.join(self.data_dir, "X_train.npy"))
        self.X_test = np.load(os.path.join(self.data_dir, "X_test.npy"))
        self.y_test = np.load(os.path.join(self.data_dir, "y_test.npy"))

    def compute_global_explanations(self, sample_size: int = 440) -> Dict[str, Any]:
        """
        Computes global mean absolute SHAP values across test samples.
        """
        print(f"Computing global SHAP values on {sample_size} test samples...")
        X_sample = self.X_test[:sample_size]
        shap_vals = self.explainer.shap_values(X_sample)

        # Handle 3D array (N, num_features, num_classes) or list of (N, num_features)
        if isinstance(shap_vals, list):
            # List of arrays per class -> stack to (N, num_features, num_classes)
            shap_array = np.stack(shap_vals, axis=-1)
        else:
            shap_array = shap_vals

        # Mean absolute SHAP per feature across all samples and classes
        # shap_array shape: (N, num_features, num_classes)
        mean_abs_per_feature_class = np.mean(np.abs(shap_array), axis=0) # (num_features, num_classes)
        overall_mean_abs_per_feature = np.mean(mean_abs_per_feature_class, axis=1) # (num_features,)

        feature_ranking = sorted(
            [
                {"feature": feat, "mean_abs_shap": float(score)}
                for feat, score in zip(self.feature_names, overall_mean_abs_per_feature)
            ],
            key=lambda x: x["mean_abs_shap"],
            reverse=True,
        )

        # Per-class top features
        per_class_importance = {}
        for c_idx, c_name in enumerate(self.class_names):
            class_scores = mean_abs_per_feature_class[:, c_idx]
            per_class_importance[c_name] = {
                feat: float(score) for feat, score in zip(self.feature_names, class_scores)
            }

        global_results = {
            "overall_feature_ranking": feature_ranking,
            "per_class_feature_importance": per_class_importance,
            "sample_size": sample_size,
        }

        # Save JSON
        json_path = os.path.join(self.metrics_dir, "shap_global_importance.json")
        with open(json_path, "w") as f:
            json.dump(global_results, f, indent=2)

        # Generate Global Visualizations
        self._plot_global_feature_importance(feature_ranking)
        self._plot_class_importance_heatmap(per_class_importance)

        print(f"Global SHAP analysis complete. Plots saved to: {self.plot_dir}")
        return global_results

    def explain_instance(
        self,
        feature_vector: np.ndarray,
        raw_feature_dict: Optional[Dict[str, float]] = None,
        target_class: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generates local SHAP explanation for a single input instance.
        """
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)

        # Model prediction and probabilities
        probs = self.model.predict_proba(feature_vector)[0]
        pred_class_id = int(np.argmax(probs)) if target_class is None else target_class
        pred_class_name = self.class_names[pred_class_id]
        pred_prob = float(probs[pred_class_id])

        # Compute SHAP values for single instance
        shap_vals = self.explainer.shap_values(feature_vector)
        if isinstance(shap_vals, list):
            instance_shap = shap_vals[pred_class_id][0] # (num_features,)
        else:
            instance_shap = shap_vals[0, :, pred_class_id] # (num_features,)

        # Expected base value for predicted class
        expected_val = (
            float(self.explainer.expected_value[pred_class_id])
            if hasattr(self.explainer.expected_value, "__getitem__")
            else float(self.explainer.expected_value)
        )

        # Pair features with SHAP attribution
        feature_attributions = []
        for i, feat in enumerate(self.feature_names):
            val_scaled = float(feature_vector[0, i])
            val_raw = raw_feature_dict.get(feat, val_scaled) if raw_feature_dict else val_scaled
            attr_val = float(instance_shap[i])

            feature_attributions.append({
                "feature": feat,
                "raw_value": val_raw,
                "scaled_value": val_scaled,
                "shap_value": attr_val,
                "contribution": "positive" if attr_val > 0 else "negative",
                "magnitude": abs(attr_val),
            })

        # Sort by absolute SHAP impact
        feature_attributions.sort(key=lambda x: x["magnitude"], reverse=True)

        return {
            "predicted_class_id": pred_class_id,
            "predicted_class_name": pred_class_name,
            "prediction_probability": pred_prob,
            "base_expected_value": expected_val,
            "feature_attributions": feature_attributions,
            "all_class_probabilities": {
                c_name: float(p) for c_name, p in zip(self.class_names, probs)
            },
        }

    def generate_sample_local_plots(self, num_samples: int = 3) -> List[str]:
        """
        Generates and saves local waterfall/bar plots for representative test samples.
        """
        saved_plots = []
        raw_test = pd.read_csv(os.path.join(self.data_dir, "test_raw.csv"))

        for idx in range(min(num_samples, len(self.X_test))):
            vec = self.X_test[idx : idx + 1]
            raw_dict = raw_test.iloc[idx][self.feature_names].to_dict()
            true_label = raw_test.iloc[idx]["label"]

            exp = self.explain_instance(vec, raw_feature_dict=raw_dict)
            pred_name = exp["predicted_class_name"]

            fig, ax = plt.subplots(figsize=(9, 5))
            attrs = exp["feature_attributions"][::-1] # Ascending order for barh
            names = [f"{a['feature']} ({a['raw_value']:.1f})" for a in attrs]
            vals = [a["shap_value"] for a in attrs]
            colors = ["#2e7d32" if v > 0 else "#c62828" for v in vals]

            ax.barh(names, vals, color=colors, edgecolor="black", alpha=0.85)
            ax.axvline(0, color="black", linestyle="--", lw=0.8)
            ax.set_title(
                f"Local SHAP Feature Attribution (Sample #{idx+1})\n"
                f"Predicted: {pred_name.upper()} (P={exp['prediction_probability']:.2f}) | True: {true_label.upper()}",
                fontsize=12,
            )
            ax.set_xlabel("SHAP Value (Impact on Model Confidence)")
            plt.tight_layout()

            plot_path = os.path.join(self.plot_dir, f"shap_local_sample_{idx+1}_{pred_name}.png")
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            saved_plots.append(plot_path)

        return saved_plots

    def _plot_global_feature_importance(self, ranking: List[Dict[str, Any]]) -> None:
        """Plots global feature importance ranking bar chart."""
        df_rank = pd.DataFrame(ranking).sort_values("mean_abs_shap", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(
            df_rank["feature"],
            df_rank["mean_abs_shap"],
            color="#1565c0",
            edgecolor="black",
            alpha=0.85,
        )
        ax.set_title("Global Feature Attribution (Mean Absolute SHAP Value)", fontsize=13)
        ax.set_xlabel("Mean |SHAP Value| Across All Classes")
        ax.set_ylabel("Agronomic Feature")
        plt.tight_layout()

        fig.savefig(os.path.join(self.plot_dir, "shap_global_bar.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_class_importance_heatmap(self, per_class_imp: Dict[str, Dict[str, float]]) -> None:
        """Plots class-by-feature SHAP importance heatmap."""
        df_hm = pd.DataFrame(per_class_imp).T # (22 crops, 7 features)

        fig, ax = plt.subplots(figsize=(10, 11))
        sns.heatmap(
            df_hm,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            cbar_kws={"label": "Mean |SHAP Value|"},
            ax=ax,
        )
        ax.set_title("Per-Crop Feature Attribution Matrix (TreeSHAP)", fontsize=13, pad=10)
        ax.set_xlabel("Agronomic Feature", fontsize=11)
        ax.set_ylabel("Target Crop Class", fontsize=11)
        plt.tight_layout()

        fig.savefig(os.path.join(self.plot_dir, "shap_per_class_matrix.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    explainer = SoilSHAPExplainer()
    explainer.compute_global_explanations()
    plots = explainer.generate_sample_local_plots(num_samples=3)
    print(f"Generated sample local plots: {plots}")
