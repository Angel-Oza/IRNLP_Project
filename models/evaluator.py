"""
Model Evaluator and Performance Visualizer.

Calculates multi-class classification metrics:
- Accuracy, Macro-Precision, Macro-Recall, Macro-F1, Weighted-F1, Log-Loss
- Generates Confusion Matrix heatmaps and Model Comparison bar charts
- Saves artifacts to `results/metrics/` and `results/plots/baselines/`
"""

import os
import json
from typing import Dict, List, Any, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
    classification_report,
)


class ModelEvaluator:
    """
    Evaluator computing standard multi-class metrics and generating publication-ready charts.
    """

    def __init__(
        self,
        class_names: List[str],
        feature_names: List[str],
        output_metrics_dir: Optional[str] = None,
        output_plots_dir: Optional[str] = None,
    ):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.metrics_dir = output_metrics_dir or os.path.join(base_dir, "results", "metrics")
        self.plots_dir = output_plots_dir or os.path.join(base_dir, "results", "plots", "baselines")
        self.tables_dir = os.path.join(base_dir, "results", "tables")

        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.tables_dir, exist_ok=True)

        self.class_names = class_names
        self.feature_names = feature_names

    def evaluate_model(
        self,
        model_name: str,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        cv_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates a single model on held-out test data.
        """
        y_pred = model.predict(X_test)
        has_proba = hasattr(model, "predict_proba")
        y_prob = model.predict_proba(X_test) if has_proba else None

        acc = float(accuracy_score(y_test, y_pred))
        prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        loss = float(log_loss(y_test, y_prob)) if y_prob is not None else None
        cm = confusion_matrix(y_test, y_pred)

        report_dict = classification_report(
            y_test, y_pred, target_names=self.class_names, output_dict=True, zero_division=0
        )

        metrics = {
            "model_name": model_name,
            "test_accuracy": acc,
            "macro_precision": prec_macro,
            "macro_recall": rec_macro,
            "macro_f1": f1_macro,
            "weighted_f1": f1_weighted,
            "log_loss": loss,
            "cv_scores": cv_scores or {},
            "confusion_matrix": cm.tolist(),
            "per_class_metrics": report_dict,
        }

        # Plot individual Confusion Matrix
        self._plot_confusion_matrix(model_name, cm)

        # Plot feature importances if tree model
        if hasattr(model, "feature_importances_"):
            self._plot_feature_importance(model_name, model.feature_importances_)

        return metrics

    def _plot_confusion_matrix(self, model_name: str, cm: np.ndarray) -> None:
        """Plots high-resolution normalized confusion matrix."""
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={"label": "Normalized Proportion"},
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix: {model_name} (Held-out Test Set, N=440)", fontsize=14, pad=12)
        ax.set_xlabel("Predicted Crop Class", fontsize=12)
        ax.set_ylabel("True Ground-Truth Crop Class", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        fig_path = os.path.join(self.plots_dir, f"cm_{model_name.lower()}.png")
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_feature_importance(self, model_name: str, importances: np.ndarray) -> None:
        """Plots feature importance bar chart."""
        df_imp = pd.DataFrame({"Feature": self.feature_names, "Importance": importances})
        df_imp = df_imp.sort_values(by="Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(df_imp["Feature"], df_imp["Importance"], color="#2b5c8f", edgecolor="black", alpha=0.85)
        ax.set_title(f"Feature Importance ({model_name})", fontsize=13)
        ax.set_xlabel("Relative Importance Score")
        ax.set_ylabel("Agronomic Feature")
        plt.tight_layout()

        fig_path = os.path.join(self.plots_dir, f"feature_importance_{model_name.lower()}.png")
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def generate_model_comparison(self, all_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Generates comparison tables and bar charts across all evaluated models.
        """
        rows = []
        for name, res in all_results.items():
            cv_f1 = res.get("cv_scores", {}).get("cv_f1_macro_mean", np.nan)
            cv_f1_std = res.get("cv_scores", {}).get("cv_f1_macro_std", np.nan)
            rows.append({
                "Model": name,
                "CV Macro-F1 (Mean ± Std)": f"{cv_f1:.4f} ± {cv_f1_std:.4f}" if not np.isnan(cv_f1) else "N/A",
                "Test Accuracy": res["test_accuracy"],
                "Test Macro-Precision": res["macro_precision"],
                "Test Macro-Recall": res["macro_recall"],
                "Test Macro-F1": res["macro_f1"],
                "Test Weighted-F1": res["weighted_f1"],
                "Log-Loss": res["log_loss"] if res["log_loss"] is not None else "N/A",
            })

        df_comp = pd.DataFrame(rows).sort_values(by="Test Macro-F1", ascending=False).reset_index(drop=True)

        # Save table to CSV and Markdown
        df_comp.to_csv(os.path.join(self.tables_dir, "baseline_comparison.csv"), index=False)
        md_table = df_comp.to_markdown(index=False)
        with open(os.path.join(self.tables_dir, "baseline_comparison.md"), "w") as f:
            f.write(f"# Machine Learning Baseline Models Comparison\n\n{md_table}\n")

        # Plot comparison chart
        self._plot_comparison_barchart(df_comp)

        return df_comp

    def _plot_comparison_barchart(self, df_comp: pd.DataFrame) -> None:
        """Plots comparative bar chart of model metrics."""
        fig, ax = plt.subplots(figsize=(10, 6))

        metrics = ["Test Accuracy", "Test Macro-Precision", "Test Macro-Recall", "Test Macro-F1"]
        x = np.arange(len(df_comp))
        width = 0.18

        for idx, metric in enumerate(metrics):
            ax.bar(
                x + idx * width,
                df_comp[metric],
                width,
                label=metric,
                alpha=0.85,
                edgecolor="black",
            )

        ax.set_title("Machine Learning Baseline Models Comparison (Held-out Test Split, N=440)", fontsize=13)
        ax.set_xlabel("Model Architecture", fontsize=11)
        ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(df_comp["Model"], fontsize=10)
        ax.set_ylim(0.80, 1.02)
        ax.legend(loc="lower right")
        plt.tight_layout()

        fig_path = os.path.join(self.plots_dir, "baseline_comparison_barchart.png")
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
