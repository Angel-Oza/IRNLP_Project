"""
Experiment F Orchestrator: Multi-Source Noise Robustness & Fault Tolerance Benchmark.

Evaluates performance resilience of:
1. SensorOnlyNet (Unimodal PyTorch baseline)
2. SimpleFusionNet (Unweighted early concatenation)
3. GatedFusionNet (Dynamic sigmoid gating without reliability awareness)
4. ReliabilityGatedFusionNet (Reliability-conditioned dynamic gated fusion)
5. Traditional Baselines (Random Forest, XGBoost, MLP)

Across progressive corruption regimes:
- Clean Telemetry (Baseline)
- 5% Gaussian Noise
- 10% Gaussian Noise
- 20% Gaussian Noise
- Sensor Drift (1.5 sigma linear offset)
- Missing Sensor Telemetry (20% channel dropout)

Calculates Relative Resilience Index (RRI = F1_corrupted / F1_clean),
degradation slope, dynamic reliability response, and gate activation shift.
"""

import os
import sys
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

from fusion.gated_fusion import SensorOnlyNet, SimpleFusionNet, GatedFusionNet, ReliabilityGatedFusionNet
from iot.noise_simulator import SensorFaultInjector, get_fault_injector
from iot.reliability import SensorReliabilityEstimator, get_reliability_estimator


class RobustnessExperimentRunner:
    """
    Executes Experiment F and outputs research benchmarking metrics and visualizations.
    """

    CORRUPTION_CONDITIONS = [
        {"name": "Clean (Nominal)", "type": "clean", "param": 0.0},
        {"name": "Gaussian Noise 5%", "type": "noise", "param": 0.05},
        {"name": "Gaussian Noise 10%", "type": "noise", "param": 0.10},
        {"name": "Gaussian Noise 20%", "type": "noise", "param": 0.20},
        {"name": "Sensor Drift (1.5σ)", "type": "drift", "param": 1.5},
        {"name": "Missing Channels (20%)", "type": "missing", "param": 0.20},
    ]

    def __init__(self, base_dir: Optional[str] = None, random_state: int = 42):
        self.base_dir = os.path.abspath(base_dir or os.path.join(os.path.dirname(__file__), ".."))
        self.data_dir = os.path.join(self.base_dir, "data", "processed")
        self.ckpt_dir = os.path.join(self.base_dir, "models", "checkpoints")
        self.metrics_dir = os.path.join(self.base_dir, "results", "metrics")
        self.tables_dir = os.path.join(self.base_dir, "results", "tables")
        self.plots_dir = os.path.join(self.base_dir, "results", "plots", "fusion")

        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.tables_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        self.random_state = random_state
        self.fault_injector = get_fault_injector(random_state=random_state)
        self.reliability_estimator = get_reliability_estimator()

        # Load held-out test data
        self.X_test = np.load(os.path.join(self.data_dir, "X_test.npy"))
        self.Z_test = np.load(os.path.join(self.data_dir, "Z_test_text.npy"))
        self.y_test = np.load(os.path.join(self.data_dir, "y_test.npy"))

        with open(os.path.join(self.data_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)

        self.num_classes = len(self.metadata["class_names"])
        self.latent_dim = 64

        # Load neural models
        self.models: Dict[str, torch.nn.Module] = self._load_pytorch_models()
        # Load traditional ML baselines
        self.sklearn_models = self._load_sklearn_models()

    def _load_pytorch_models(self) -> Dict[str, torch.nn.Module]:
        models = {
            "SensorOnlyNet": SensorOnlyNet(latent_dim=self.latent_dim, num_classes=self.num_classes),
            "SimpleFusionNet": SimpleFusionNet(latent_dim=self.latent_dim, num_classes=self.num_classes),
            "GatedFusionNet": GatedFusionNet(latent_dim=self.latent_dim, num_classes=self.num_classes),
            "ReliabilityGatedFusionNet": ReliabilityGatedFusionNet(latent_dim=self.latent_dim, num_classes=self.num_classes),
        }

        for name, model in models.items():
            ckpt_path = os.path.join(self.ckpt_dir, f"{name.lower()}.pt")
            if os.path.exists(ckpt_path):
                model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
            model.eval()

        return models

    def _load_sklearn_models(self) -> Dict[str, Any]:
        name_map = {
            "randomforest": "RandomForest",
            "mlp": "MLP",
            "logisticregression": "LogisticRegression",
        }
        sklearn_models = {}
        for file_key, display_name in name_map.items():
            pkl_path = os.path.join(self.ckpt_dir, f"{file_key}.pkl")
            if os.path.exists(pkl_path):
                try:
                    with open(pkl_path, "rb") as f:
                        sklearn_models[display_name] = pickle.load(f)
                except Exception as e:
                    print(f"Warning: Could not load {file_key}.pkl: {e}")
        return sklearn_models

    def generate_corrupted_dataset(
        self, condition: Dict[str, Any]
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Applies designated corruption to test split and estimates dynamic reliability scores.
        
        Returns:
            Tuple of (X_corrupted, missing_mask, reliability_scores)
        """
        c_type = condition["type"]
        param = condition["param"]

        # Reset RNG for deterministic corruption
        self.fault_injector.set_seed(self.random_state)
        self.reliability_estimator.reset_state()

        if c_type == "clean":
            X_corr = self.X_test.copy()
            mask = None

        elif c_type == "noise":
            X_corr = self.fault_injector.inject_gaussian_noise(self.X_test, noise_level=param)
            mask = None

        elif c_type == "drift":
            X_corr = self.fault_injector.inject_sensor_drift(self.X_test, drift_strength=param)
            mask = None

        elif c_type == "missing":
            X_corr, mask = self.fault_injector.inject_missingness(
                self.X_test, missing_rate=param, impute_value=0.0
            )
        else:
            raise ValueError(f"Unknown corruption type: {c_type}")

        # Compute online dynamic reliability vector
        r_scores = self.reliability_estimator.estimate_batch(X_corr, missing_masks=mask)
        return X_corr, mask, r_scores

    def evaluate_model_on_corrupted(
        self,
        model_name: str,
        model: Any,
        X_corr: np.ndarray,
        r_scores: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluates a single model under corrupted test condition.
        """
        if isinstance(model, torch.nn.Module):
            model.eval()
            x_t = torch.tensor(X_corr, dtype=torch.float32)
            z_t = torch.tensor(self.Z_test, dtype=torch.float32)
            r_t = torch.tensor(r_scores[:, np.newaxis], dtype=torch.float32)

            with torch.no_grad():
                if isinstance(model, SensorOnlyNet):
                    logits = model(x_t)
                    gate_val = None
                elif isinstance(model, SimpleFusionNet):
                    logits = model(x_t, z_t)
                    gate_val = None
                elif isinstance(model, GatedFusionNet):
                    logits, g = model(x_t, z_t, return_gate=True)
                    gate_val = float(g.mean().item())
                elif isinstance(model, ReliabilityGatedFusionNet):
                    logits, g = model(x_t, z_t, reliability=r_t, return_gate=True)
                    gate_val = float(g.mean().item())
                else:
                    logits = model(x_t, z_t)
                    gate_val = None

                probs = F.softmax(logits, dim=-1).numpy()
                preds = np.argmax(probs, axis=1)

        else:
            # Sklearn baseline model
            probs = model.predict_proba(X_corr)
            preds = np.argmax(probs, axis=1)
            gate_val = None

        acc = float(accuracy_score(self.y_test, preds))
        prec = float(precision_score(self.y_test, preds, average="macro", zero_division=0))
        rec = float(recall_score(self.y_test, preds, average="macro", zero_division=0))
        f1 = float(f1_score(self.y_test, preds, average="macro", zero_division=0))
        loss = float(log_loss(self.y_test, probs, labels=list(range(self.num_classes))))

        return {
            "accuracy": acc,
            "macro_precision": prec,
            "macro_recall": rec,
            "macro_f1": f1,
            "log_loss": loss,
            "mean_gate_weight": gate_val,
        }

    def run_experiment_f(self) -> Dict[str, Any]:
        """
        Executes Experiment F across all conditions and models.
        """
        print("===============================================================")
        print("Executing Experiment F: Sensor Fault Robustness & Degradation Benchmark")
        print("===============================================================")

        all_models = {}
        # Neural Fusion Architectures
        all_models.update(self.models)
        # Traditional ML Baselines
        all_models.update(self.sklearn_models)

        benchmark_results = {}
        reliability_stats = {}
        table_rows = []

        # First obtain clean baseline scores for relative resilience index (RRI)
        clean_f1_scores = {}
        X_clean, _, r_clean = self.generate_corrupted_dataset(self.CORRUPTION_CONDITIONS[0])
        for m_name, model in all_models.items():
            res = self.evaluate_model_on_corrupted(m_name, model, X_clean, r_clean)
            clean_f1_scores[m_name] = res["macro_f1"]

        for cond in self.CORRUPTION_CONDITIONS:
            cond_name = cond["name"]
            print(f"\nEvaluating Regime: {cond_name}...")
            X_corr, mask, r_scores = self.generate_corrupted_dataset(cond)

            r_diag = SensorReliabilityEstimator.compute_reliability_diagnostics(r_scores, cond_name)
            reliability_stats[cond_name] = r_diag
            print(f"  Dynamic Reliability: Mean={r_diag['mean_reliability']:.4f} | Median={r_diag['median_reliability']:.4f}")

            benchmark_results[cond_name] = {}

            for m_name, model in all_models.items():
                res = self.evaluate_model_on_corrupted(m_name, model, X_corr, r_scores)
                clean_f1 = clean_f1_scores[m_name]
                curr_f1 = res["macro_f1"]
                rri = float(curr_f1 / clean_f1) if clean_f1 > 0 else 0.0
                deg = float(clean_f1 - curr_f1)

                res["rri"] = rri
                res["f1_degradation"] = deg
                benchmark_results[cond_name][m_name] = res

                table_rows.append({
                    "Condition": cond_name,
                    "Model": m_name,
                    "Test Accuracy": f"{res['accuracy'] * 100:.2f}%",
                    "Macro-F1": f"{res['macro_f1']:.4f}",
                    "Resilience Index (RRI)": f"{rri * 100:.2f}%",
                    "F1 Drop (ΔF1)": f"{deg:.4f}",
                    "Log-Loss": f"{res['log_loss']:.4f}",
                    "Mean Gate E[g]": f"{res['mean_gate_weight']:.4f}" if res["mean_gate_weight"] is not None else "N/A",
                    "Mean Reliability r": f"{r_diag['mean_reliability']:.4f}",
                })

                print(f"    - {m_name:26s} | Acc: {res['accuracy']*100:6.2f}% | F1: {res['macro_f1']:.4f} | RRI: {rri*100:6.2f}%")

        # Save Metrics JSONs
        metrics_out_path = os.path.join(self.metrics_dir, "phase6_robustness_metrics.json")
        with open(metrics_out_path, "w") as f:
            json.dump(benchmark_results, f, indent=2)

        rel_out_path = os.path.join(self.metrics_dir, "phase6_reliability_metrics.json")
        with open(rel_out_path, "w") as f:
            json.dump(reliability_stats, f, indent=2)

        # Save CSV and Markdown Tables
        df_results = pd.DataFrame(table_rows)
        csv_path = os.path.join(self.tables_dir, "phase6_robustness_comparison.csv")
        df_results.to_csv(csv_path, index=False)

        md_path = os.path.join(self.tables_dir, "phase6_robustness_comparison.md")
        with open(md_path, "w") as f:
            f.write("# Experiment F: Sensor Fault Robustness & Degradation Benchmark\n\n")
            f.write(df_results.to_markdown(index=False))
            f.write("\n")

        # Save Resilience Summary Table (Aggregating 20% Noise, Drift, and Missingness)
        self._generate_summary_resilience_table(benchmark_results, clean_f1_scores)

        # Generate Publication-Quality Plots
        self._plot_accuracy_vs_noise(benchmark_results)
        self._plot_f1_degradation_curves(benchmark_results)
        self._plot_reliability_vs_corruption(reliability_stats)
        self._plot_gate_activation_shift(benchmark_results)

        print("\n===============================================================")
        print(f"Experiment F Completed. Artifacts exported to {self.tables_dir} and {self.plots_dir}")
        print("===============================================================")

        return {
            "robustness_metrics": benchmark_results,
            "reliability_diagnostics": reliability_stats,
            "results_dataframe": df_results,
        }

    def _generate_summary_resilience_table(
        self,
        benchmark_results: Dict[str, Dict[str, Dict[str, Any]]],
        clean_f1_scores: Dict[str, float],
    ) -> None:
        """Generates condensed model resilience comparison table."""
        summary_rows = []
        models = list(clean_f1_scores.keys())

        for m in models:
            f1_clean = clean_f1_scores[m]
            f1_noise20 = benchmark_results["Gaussian Noise 20%"][m]["macro_f1"]
            rri_noise20 = benchmark_results["Gaussian Noise 20%"][m]["rri"]
            f1_drift = benchmark_results["Sensor Drift (1.5σ)"][m]["macro_f1"]
            rri_drift = benchmark_results["Sensor Drift (1.5σ)"][m]["rri"]
            f1_miss = benchmark_results["Missing Channels (20%)"][m]["macro_f1"]
            rri_miss = benchmark_results["Missing Channels (20%)"][m]["rri"]

            # Overall Mean RRI across all 5 degraded regimes
            all_rri = [
                benchmark_results[c["name"]][m]["rri"]
                for c in self.CORRUPTION_CONDITIONS[1:]
            ]
            mean_rri = float(np.mean(all_rri))

            summary_rows.append({
                "Architecture": m,
                "Clean Macro-F1": f"{f1_clean:.4f}",
                "20% Noise F1 (RRI)": f"{f1_noise20:.4f} ({rri_noise20*100:.1f}%)",
                "Drift F1 (RRI)": f"{f1_drift:.4f} ({rri_drift*100:.1f}%)",
                "Missingness F1 (RRI)": f"{f1_miss:.4f} ({rri_miss*100:.1f}%)",
                "Average Degraded RRI": f"{mean_rri*100:.2f}%",
            })

        df_sum = pd.DataFrame(summary_rows).sort_values("Average Degraded RRI", ascending=False)
        out_path = os.path.join(self.tables_dir, "phase6_model_resilience_summary.md")
        with open(out_path, "w") as f:
            f.write("# Model Resilience & Degradation Retention Summary\n\n")
            f.write(df_sum.to_markdown(index=False))
            f.write("\n")

    def _plot_accuracy_vs_noise(self, results: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        noise_levels = [0, 5, 10, 20]
        noise_conds = ["Clean (Nominal)", "Gaussian Noise 5%", "Gaussian Noise 10%", "Gaussian Noise 20%"]
        target_models = ["SensorOnlyNet", "SimpleFusionNet", "GatedFusionNet", "ReliabilityGatedFusionNet", "RandomForest"]

        fig, ax = plt.subplots(figsize=(9, 5.5))
        markers = {"SensorOnlyNet": "s", "SimpleFusionNet": "^", "GatedFusionNet": "D", "ReliabilityGatedFusionNet": "o", "RandomForest": "v"}
        colors = {"SensorOnlyNet": "#757575", "SimpleFusionNet": "#ab47bc", "GatedFusionNet": "#1976d2", "ReliabilityGatedFusionNet": "#2e7d32", "RandomForest": "#e65100"}

        for m in target_models:
            accs = [results[c][m]["accuracy"] * 100 for c in noise_conds]
            ax.plot(
                noise_levels,
                accs,
                marker=markers[m],
                color=colors[m],
                linewidth=2.2,
                markersize=7,
                label=m,
            )

        ax.set_title("Model Accuracy vs. Progressive Gaussian Sensor Noise", fontsize=13, pad=10)
        ax.set_xlabel("Injected Sensor Noise Level (η % of σ)", fontsize=11)
        ax.set_ylabel("Test Accuracy (%)", fontsize=11)
        ax.set_xticks(noise_levels)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower left", fontsize=10)
        plt.tight_layout()

        fig.savefig(os.path.join(self.plots_dir, "phase6_accuracy_vs_noise.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_f1_degradation_curves(self, results: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        cond_labels = [c["name"] for c in self.CORRUPTION_CONDITIONS]
        target_models = ["ReliabilityGatedFusionNet", "GatedFusionNet", "SimpleFusionNet", "SensorOnlyNet", "RandomForest"]

        fig, ax = plt.subplots(figsize=(11, 5.5))
        x = np.arange(len(cond_labels))
        width = 0.15

        colors = ["#2e7d32", "#1976d2", "#ab47bc", "#757575", "#e65100"]
        for i, (m, col) in enumerate(zip(target_models, colors)):
            f1s = [results[c][m]["macro_f1"] for c in cond_labels]
            ax.bar(x + (i - 2) * width, f1s, width, label=m, color=col, alpha=0.88, edgecolor="black")

        ax.set_title("Macro-F1 Score Across Sensor Corruption Regimes (Experiment F)", fontsize=13, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels, rotation=15, ha="right", fontsize=9.5)
        ax.set_ylabel("Macro-F1 Score", fontsize=11)
        ax.set_ylim(0.50, 1.03)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="lower left", fontsize=9.5)
        plt.tight_layout()

        fig.savefig(os.path.join(self.plots_dir, "phase6_f1_degradation_curves.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_reliability_vs_corruption(self, rel_stats: Dict[str, Dict[str, Any]]) -> None:
        conds = list(rel_stats.keys())
        means = [rel_stats[c]["mean_reliability"] for c in conds]
        medians = [rel_stats[c]["median_reliability"] for c in conds]
        stds = [rel_stats[c]["std_reliability"] for c in conds]

        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(conds))

        ax.bar(x - 0.15, means, 0.3, label="Mean Reliability", color="#0288d1", edgecolor="black", alpha=0.85)
        ax.bar(x + 0.15, medians, 0.3, label="Median Reliability", color="#26a69a", edgecolor="black", alpha=0.85)

        ax.set_title("Online Dynamic Sensor Reliability r(t) Response Across Corruption Regimes", fontsize=12.5)
        ax.set_xticks(x)
        ax.set_xticklabels(conds, rotation=15, ha="right", fontsize=9.5)
        ax.set_ylabel("Dynamic Reliability Score r in [0, 1]", fontsize=11)
        ax.set_ylim(0.0, 1.1)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=10)
        plt.tight_layout()

        fig.savefig(os.path.join(self.plots_dir, "phase6_reliability_vs_corruption.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_gate_activation_shift(self, results: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        conds = [c["name"] for c in self.CORRUPTION_CONDITIONS]
        gated_gates = [results[c]["GatedFusionNet"]["mean_gate_weight"] for c in conds]
        rel_gates = [results[c]["ReliabilityGatedFusionNet"]["mean_gate_weight"] for c in conds]

        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(conds))

        ax.plot(x, gated_gates, marker="D", lw=2.2, color="#1976d2", label="GatedFusionNet Gate E[g]")
        ax.plot(x, rel_gates, marker="o", lw=2.5, color="#2e7d32", label="ReliabilityGatedFusionNet Effective Gate E[r·g]")

        ax.set_title("Adaptive Sensor Gate Suppression Under Injected Corruption (RQ3)", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(conds, rotation=15, ha="right", fontsize=9.5)
        ax.set_ylabel("Effective Sensor Modality Weight", fontsize=11)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower left", fontsize=10)
        plt.tight_layout()

        fig.savefig(os.path.join(self.plots_dir, "phase6_gate_activation_shift.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    runner = RobustnessExperimentRunner()
    runner.run_experiment_f()
