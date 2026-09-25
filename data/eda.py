"""
Automated Exploratory Data Analysis (EDA) Module.

Generates publication-quality visualizations and statistical profiling across
Dataset A (IVRI Soil Chemistry), Dataset B (Berambadi Telemetry), and Dataset C (Crop Benchmark).
Saves figures to `results/plots/eda/` and summary statistics to `results/metrics/`.
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Optional, Any
import numpy as np

from data.loader import DatasetLoader


class AgronomicEDA:
    """
    Automated EDA generator producing publication-ready charts and statistical profiles.
    """

    def __init__(self, output_plot_dir: Optional[str] = None, output_metrics_dir: Optional[str] = None):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.plot_dir = output_plot_dir or os.path.join(base_dir, "results", "plots", "eda")
        self.metrics_dir = output_metrics_dir or os.path.join(base_dir, "results", "metrics")
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

        self.loader = DatasetLoader(base_dir=base_dir)

        # Style configuration
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 15,
        })

    def run_all_eda(self) -> Dict[str, Any]:
        """Runs full EDA pipeline across all three datasets."""
        print("Starting Automated EDA Pipeline...")
        summary = {}

        # 1. Dataset C Analysis
        df_crop = self.loader.load_crop_benchmark_dataset()
        summary["dataset_c"] = self._eda_dataset_c(df_crop)

        # 2. Dataset A Analysis
        df_chem = self.loader.load_soil_chemistry_dataset()
        summary["dataset_a"] = self._eda_dataset_a(df_chem)

        # 3. Dataset B Analysis
        df_tel = self.loader.load_sensor_telemetry_dataset(year=2016)
        summary["dataset_b"] = self._eda_dataset_b(df_tel)

        # Save summary JSON
        summary_path = os.path.join(self.metrics_dir, "eda_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"EDA Complete. Plots saved to: {self.plot_dir}")
        print(f"Summary metrics saved to: {summary_path}")
        return summary

    def _eda_dataset_c(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generates plots and statistical profiling for Dataset C."""
        print("-> Generating Dataset C EDA plots...")
        features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

        # 1. Feature Distribution Histograms with KDE
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        axes = axes.flatten()

        for idx, feat in enumerate(features):
            ax = axes[idx]
            sns.histplot(df[feat], kde=True, ax=ax, color="#2b5c8f", edgecolor="black", alpha=0.6)
            ax.set_title(f"Distribution of {feat}")
            ax.set_xlabel(feat)
            ax.set_ylabel("Frequency")

        axes[7].axis("off")
        fig.suptitle("Dataset C: Feature Distributions Across 22 Crop Classes", fontsize=16, y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(self.plot_dir, "dataset_c_distributions.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        # 2. Correlation Heatmap
        fig, ax = plt.subplots(figsize=(9, 7))
        corr = df[features].corr()
        sns.heatmap(corr, annot=True, cmap="vlag", fmt=".2f", vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Pearson Correlation'})
        ax.set_title("Dataset C: Pairwise Feature Correlation Matrix")
        plt.tight_layout()
        fig.savefig(os.path.join(self.plot_dir, "dataset_c_correlation_matrix.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        # 3. Macro-Nutrient Boxplots by Crop
        fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
        for idx, nut in enumerate(["N", "P", "K"]):
            ax = axes[idx]
            sns.boxplot(x="label", y=nut, data=df, ax=ax, palette="tab20")
            ax.set_title(f"{nut} Nutrient Requirements by Crop")
            ax.set_ylabel(f"{nut} Index / kg/ha")
            ax.set_xlabel("")
            ax.tick_params(axis='x', rotation=45)

        axes[2].set_xlabel("Crop Category")
        fig.suptitle("Dataset C: Macro-Nutrient Requirements (N, P, K) Across Crops", fontsize=16, y=1.01)
        plt.tight_layout()
        fig.savefig(os.path.join(self.plot_dir, "dataset_c_npk_by_crop.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        # 4. Environmental Envelope Scatter: Temperature vs Rainfall colored by Crop
        fig, ax = plt.subplots(figsize=(12, 8))
        # Pick 8 major representative crops for clarity
        rep_crops = ["rice", "maize", "chickpea", "coffee", "apple", "cotton", "banana", "watermelon"]
        df_rep = df[df["label"].isin(rep_crops)]
        sns.scatterplot(
            data=df_rep,
            x="temperature",
            y="rainfall",
            hue="label",
            style="label",
            s=80,
            alpha=0.85,
            ax=ax,
            palette="Set1"
        )
        ax.set_title("Dataset C: Bioclimatic Envelopes (Temperature vs. Rainfall)")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Rainfall (mm)")
        ax.legend(title="Crop Class", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        fig.savefig(os.path.join(self.plot_dir, "dataset_c_bioclimatic_envelope.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        stats = df[features].describe().to_dict()
        return {
            "total_samples": len(df),
            "num_classes": df["label"].nunique(),
            "class_distribution": df["label"].value_counts().to_dict(),
            "feature_statistics": stats,
        }

    def _eda_dataset_a(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generates plots and statistical profiling for Dataset A (IVRI)."""
        print("-> Generating Dataset A (IVRI Soil Chemistry) EDA plots...")
        chem_features = ["pH", "EC", "OC", "N", "P", "K", "Zn", "Fe", "Mn", "Cu"]

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()

        for idx, feat in enumerate(chem_features):
            ax = axes[idx]
            sns.boxplot(y=df[feat], ax=ax, color="#2e7d32")
            ax.set_title(f"IVRI: {feat}")
            ax.set_ylabel("Value")

        fig.suptitle("Dataset A: IVRI Certified Soil Laboratory Chemistry Distributions (N=257)", fontsize=16, y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(self.plot_dir, "dataset_a_chemistry_distributions.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        return {
            "total_certified_samples": len(df),
            "feature_statistics": df[chem_features].describe().to_dict(),
        }

    def _eda_dataset_b(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generates plots for Dataset B (Berambadi In-Situ Telemetry)."""
        print("-> Generating Dataset B (Berambadi Telemetry) EDA plots...")

        # Plot 1-month continuous telemetry sample (e.g. July 2016 monsoon dynamics)
        df_sub = df[(df["timestamp"] >= "2016-07-01") & (df["timestamp"] <= "2016-07-31")].copy()

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

        ax1.plot(df_sub["timestamp"], df_sub["SM_5cm"], label="Soil Moisture 5cm (%)", color="#0288d1", lw=1.5)
        ax1.plot(df_sub["timestamp"], df_sub["SM_50cm"], label="Soil Moisture 50cm (%)", color="#01579b", lw=1.5, ls="--")
        ax1.set_ylabel("Moisture (% vol)")
        ax1.legend(loc="upper right")
        ax1.set_title("Dataset B: In-Situ Temporal Telemetry (Monsoon Dynamics July 2016)")

        ax2.plot(df_sub["timestamp"], df_sub["Temp_5cm"], label="Soil Temp 5cm (°C)", color="#f57c00", lw=1.5)
        ax2.plot(df_sub["timestamp"], df_sub["Temp_50cm"], label="Soil Temp 50cm (°C)", color="#b71c1c", lw=1.5, ls="--")
        ax2.set_ylabel("Temperature (°C)")
        ax2.legend(loc="upper right")

        ax3.bar(df_sub["timestamp"], df_sub["Precipitation"], label="Precipitation (mm/15min)", color="#388e3c", width=0.01)
        ax3.set_ylabel("Precipitation (mm)")
        ax3.set_xlabel("Date & Time")
        ax3.legend(loc="upper right")

        plt.tight_layout()
        fig.savefig(os.path.join(self.plot_dir, "dataset_b_telemetry_time_series.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        return {
            "total_records_2016": len(df),
            "time_span_start": str(df["timestamp"].min()),
            "time_span_end": str(df["timestamp"].max()),
            "missing_counts": df.isnull().sum().to_dict(),
        }


if __name__ == "__main__":
    eda = AgronomicEDA()
    eda.run_all_eda()
