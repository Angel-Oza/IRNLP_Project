"""
Dataset A Soil Health Index (SHI) Validation & Statistical Benchmarking Orchestrator.

Runs the Rule-Based SHI Engine over all 257 certified ICAR KVK Bareilly laboratory soil certificates (Dataset A),
validates numerical bounds and stability, and generates publication tables and visualizations.
"""

import os
import sys
import json
from typing import Dict, List, Any, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data.loader import get_data_loader
from soil_health.shi import SoilHealthIndexEngine, get_shi_engine


class SoilHealthValidator:
    """
    Validates SHI scoring over Dataset A and produces research reporting artifacts.
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = os.path.abspath(base_dir or os.path.join(os.path.dirname(__file__), ".."))
        self.loader = get_data_loader(base_dir=self.base_dir)
        self.engine = get_shi_engine()

        self.metrics_dir = os.path.join(self.base_dir, "results", "metrics")
        self.tables_dir = os.path.join(self.base_dir, "results", "tables")
        self.plots_dir = os.path.join(self.base_dir, "results", "plots", "shi")

        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.tables_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    def run_validation(self) -> Dict[str, Any]:
        """
        Executes full validation suite over Dataset A.
        """
        print("===============================================================")
        print("Running Soil Health Index (SHI) Validation on Dataset A (KVK Bareilly)")
        print("===============================================================")

        # Load Dataset A (257 samples)
        df_chem = self.loader.load_soil_chemistry_dataset()
        print(f"Loaded {len(df_chem)} certified laboratory soil samples.")

        # Batch scoring
        df_scored = self.engine.calculate_shi_batch(df_chem)

        shi_scores = df_scored["SHI_Score"].values

        # 1. Verify bounds
        assert (shi_scores >= 0.0).all() and (shi_scores <= 100.0).all(), "SHI scores violated [0, 100] bounds!"
        assert not df_scored["SHI_Score"].isna().any(), "NaN values found in SHI scores!"

        # 2. Overall Summary Statistics
        overall_stats = {
            "total_samples": int(len(df_scored)),
            "mean_shi": float(np.mean(shi_scores)),
            "median_shi": float(np.median(shi_scores)),
            "std_shi": float(np.std(shi_scores)),
            "min_shi": float(np.min(shi_scores)),
            "max_shi": float(np.max(shi_scores)),
            "p25_shi": float(np.percentile(shi_scores, 25)),
            "p75_shi": float(np.percentile(shi_scores, 75)),
        }

        # 3. Category Breakdown
        cat_counts = df_scored["SHI_Category"].value_counts().to_dict()
        cat_percentages = (df_scored["SHI_Category"].value_counts(normalize=True) * 100).to_dict()

        category_summary = [
            {
                "Category": cat,
                "Sample Count": int(cat_counts.get(cat, 0)),
                "Percentage": f"{cat_percentages.get(cat, 0.0):.2f}%",
            }
            for cat in [
                "High / Optimal Soil Health",
                "Good / Moderate Soil Health",
                "Fair / Low Soil Health",
                "Degraded / Critical Soil Health",
            ]
        ]

        # 4. Parameter Score Summary
        param_cols = [c for c in self.engine.DEFAULT_WEIGHTS.keys()]
        param_stats = []
        for p in param_cols:
            score_col = f"Score_{p}"
            scores = df_scored[score_col].dropna().values
            raw_vals = df_chem[p].dropna().values

            meta = self.engine.PARAMETER_METADATA[p]
            param_stats.append({
                "Parameter": p,
                "Full Name": meta["name"],
                "Unit": meta["unit"],
                "Weight": f"{self.engine.DEFAULT_WEIGHTS[p]:.2f}",
                "Raw Mean ± Std": f"{np.mean(raw_vals):.2f} ± {np.std(raw_vals):.2f}",
                "Score Mean": f"{np.mean(scores):.2f}",
                "Score Median": f"{np.median(scores):.2f}",
                "Score Min - Max": f"{np.min(scores):.1f} - {np.max(scores):.1f}",
                "Deficient Samples (%)": f"{(scores < 70.0).mean() * 100:.1f}%",
                "Scientific Source": meta["provenance"],
            })

        df_param_summary = pd.DataFrame(param_stats)

        # 5. Save Artifacts
        # Save metrics JSON
        metrics_payload = {
            "overall_summary": overall_stats,
            "category_distribution": category_summary,
            "parameter_statistics": param_stats,
            "weight_distribution": self.engine.DEFAULT_WEIGHTS,
        }
        json_path = os.path.join(self.metrics_dir, "phase7_shi_metrics.json")
        with open(json_path, "w") as f:
            json.dump(metrics_payload, f, indent=2)

        # Save summary CSV & Markdown
        csv_path = os.path.join(self.tables_dir, "phase7_shi_summary.csv")
        df_param_summary.to_csv(csv_path, index=False)

        md_path = os.path.join(self.tables_dir, "phase7_shi_summary.md")
        with open(md_path, "w") as f:
            f.write("# Dataset A Soil Health Index (SHI) Validation & Parameter Statistics\n\n")
            f.write(f"**Total Certified Laboratory Samples:** {len(df_scored)}  \n")
            f.write(f"**Mean SHI Score:** {overall_stats['mean_shi']:.2f} ± {overall_stats['std_shi']:.2f} (Median: {overall_stats['median_shi']:.2f})  \n")
            f.write(f"**SHI Range:** [{overall_stats['min_shi']:.2f}, {overall_stats['max_shi']:.2f}]  \n\n")

            f.write("### Soil Health Category Distribution\n\n")
            f.write(pd.DataFrame(category_summary).to_markdown(index=False))
            f.write("\n\n### Parameter-Specific Score Statistics & Provenance\n\n")
            f.write(df_param_summary.to_markdown(index=False))
            f.write("\n")

        # 6. Generate Plots
        self._plot_shi_distribution(df_scored, overall_stats)
        self._plot_parameter_scores_boxplot(df_scored, param_cols)
        self._plot_shi_vs_nutrients(df_scored)

        print(f"\nValidation complete. Mean SHI: {overall_stats['mean_shi']:.2f}/100. Artifacts saved to {self.tables_dir}")
        print("===============================================================")

        return metrics_payload

    def _plot_shi_distribution(self, df_scored: pd.DataFrame, stats: Dict[str, float]) -> None:
        fig, ax = plt.subplots(figsize=(8.5, 5))

        sns.histplot(
            df_scored["SHI_Score"],
            kde=True,
            bins=20,
            color="#2e7d32",
            edgecolor="black",
            alpha=0.75,
            ax=ax,
        )

        ax.axvline(stats["mean_shi"], color="#c62828", linestyle="--", lw=2, label=f"Mean SHI ({stats['mean_shi']:.1f})")
        ax.axvline(stats["median_shi"], color="#1565c0", linestyle=":", lw=2, label=f"Median SHI ({stats['median_shi']:.1f})")

        ax.set_title("Distribution of Soil Health Index (SHI) Across Dataset A (N=257 KVK Bareilly Samples)", fontsize=12)
        ax.set_xlabel("Soil Health Index Score (0–100 Scale)", fontsize=11)
        ax.set_ylabel("Number of Laboratory Soil Samples", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=10)
        plt.tight_layout()

        fig.savefig(os.path.join(self.plots_dir, "phase7_shi_distribution.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_parameter_scores_boxplot(self, df_scored: pd.DataFrame, param_cols: List[str]) -> None:
        score_cols = [f"Score_{p}" for p in param_cols]
        df_melt = df_scored[score_cols].melt(var_name="Parameter", value_name="Score")
        df_melt["Parameter"] = df_melt["Parameter"].str.replace("Score_", "")

        fig, ax = plt.subplots(figsize=(11, 5.5))
        sns.boxplot(
            data=df_melt,
            x="Parameter",
            y="Score",
            palette="Set2",
            ax=ax,
            boxprops=dict(alpha=0.85),
        )

        ax.axhline(100, color="green", linestyle="--", alpha=0.6, lw=1)
        ax.axhline(60, color="orange", linestyle="--", alpha=0.6, lw=1, label="Deficiency Threshold (Score < 60)")

        ax.set_title("Parameter-Level Agronomic Score Distributions Across Dataset A", fontsize=12.5)
        ax.set_xlabel("Agronomic Soil Parameter", fontsize=11)
        ax.set_ylabel("Agronomic Quality Score (0–100)", fontsize=11)
        ax.set_ylim(-5, 105)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="lower left", fontsize=10)
        plt.tight_layout()

        fig.savefig(os.path.join(self.plots_dir, "phase7_parameter_scores_boxplot.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_shi_vs_nutrients(self, df_scored: pd.DataFrame) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

        # 1. SHI vs Organic Carbon (OC)
        axes[0, 0].scatter(df_scored["OC"], df_scored["SHI_Score"], color="#2e7d32", alpha=0.6, edgecolors="none")
        axes[0, 0].set_title("SHI vs. Soil Organic Carbon (%)", fontsize=11)
        axes[0, 0].set_xlabel("Organic Carbon (%)")
        axes[0, 0].set_ylabel("Soil Health Index (SHI)")
        axes[0, 0].grid(True, linestyle="--", alpha=0.5)

        # 2. SHI vs Available Nitrogen (N)
        axes[0, 1].scatter(df_scored["N"], df_scored["SHI_Score"], color="#1565c0", alpha=0.6, edgecolors="none")
        axes[0, 1].set_title("SHI vs. Available Nitrogen (kg/ha)", fontsize=11)
        axes[0, 1].set_xlabel("Available Nitrogen (kg/ha)")
        axes[0, 1].set_ylabel("Soil Health Index (SHI)")
        axes[0, 1].grid(True, linestyle="--", alpha=0.5)

        # 3. SHI vs Available Phosphorus (P)
        axes[1, 0].scatter(df_scored["P"], df_scored["SHI_Score"], color="#e65100", alpha=0.6, edgecolors="none")
        axes[1, 0].set_title("SHI vs. Available Phosphorus (kg/ha)", fontsize=11)
        axes[1, 0].set_xlabel("Available Phosphorus (kg/ha)")
        axes[1, 0].set_ylabel("Soil Health Index (SHI)")
        axes[1, 0].grid(True, linestyle="--", alpha=0.5)

        # 4. SHI vs Soil pH
        axes[1, 1].scatter(df_scored["pH"], df_scored["SHI_Score"], color="#6a1b9a", alpha=0.6, edgecolors="none")
        axes[1, 1].set_title("SHI vs. Soil pH Reaction", fontsize=11)
        axes[1, 1].set_xlabel("Soil pH Reaction")
        axes[1, 1].set_ylabel("Soil Health Index (SHI)")
        axes[1, 1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        fig.savefig(os.path.join(self.plots_dir, "phase7_shi_vs_nutrients.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    validator = SoilHealthValidator()
    validator.run_validation()
