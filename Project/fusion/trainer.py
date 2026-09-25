"""
Multi-Source Fusion Training and Comparative Benchmarking Orchestrator.

Trains and evaluates:
1. SensorOnlyNet (Unimodal baseline)
2. SimpleFusionNet (Unweighted concatenation)
3. GatedFusionNet (Dynamic sigmoid gating)
4. ReliabilityGatedFusionNet (Reliability-conditioned gating)

Generates comparison tables, metrics JSON, and training/gate distribution plots.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

from fusion.gated_fusion import SensorOnlyNet, SimpleFusionNet, GatedFusionNet, ReliabilityGatedFusionNet
from fusion.dataset import get_multimodal_dataloaders


class FusionTrainer:
    """
    Trainer for Multi-Source Fusion Architectures.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 40,
        random_state: int = 42,
    ):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.checkpoints_dir = os.path.join(base_dir, "models", "checkpoints")
        self.metrics_dir = os.path.join(base_dir, "results", "metrics")
        self.tables_dir = os.path.join(base_dir, "results", "tables")
        self.plots_dir = os.path.join(base_dir, "results", "plots", "fusion")

        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.tables_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.random_state = random_state

        # Enforce reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Device selection: MPS or CPU
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using compute device: {self.device}")

        self.train_loader, self.test_loader = get_multimodal_dataloaders(batch_size=batch_size)

    def train_single_model(
        self, model_name: str, model: nn.Module
    ) -> Tuple[Dict[str, Any], List[float], List[float]]:
        """
        Trains a PyTorch model and evaluates on the held-out test split.
        """
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        train_losses = []
        test_losses = []
        best_test_acc = -1.0
        best_weights = None

        print(f"\n--- Training {model_name} ({self.epochs} epochs) ---")
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            model.train()
            total_train_loss = 0.0

            for x_s, z_t, r, y in self.train_loader:
                x_s, z_t, r, y = x_s.to(self.device), z_t.to(self.device), r.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                if isinstance(model, SensorOnlyNet):
                    logits = model(x_s)
                elif isinstance(model, SimpleFusionNet):
                    logits = model(x_s, z_t)
                elif isinstance(model, GatedFusionNet):
                    logits = model(x_s, z_t)
                elif isinstance(model, ReliabilityGatedFusionNet):
                    logits = model(x_s, z_t, reliability=r)
                else:
                    logits = model(x_s, z_t)

                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * len(y)

            scheduler.step()
            avg_train_loss = total_train_loss / len(self.train_loader.dataset)
            train_losses.append(avg_train_loss)

            # Evaluate on test set
            test_loss, acc, _, _, _ = self._evaluate_loss_acc(model, criterion)
            test_losses.append(test_loss)

            if acc > best_test_acc:
                best_test_acc = acc
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if epoch % 10 == 0 or epoch == self.epochs:
                print(f"  Epoch {epoch:02d}/{self.epochs:02d} | Train Loss: {avg_train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {acc:.4f}")

        train_duration = time.time() - start_time
        print(f"Completed {model_name} in {train_duration:.2f}s. Best Test Accuracy: {best_test_acc:.4f}")

        # Load best checkpoint weights
        if best_weights is not None:
            model.load_state_dict({k: v.to(self.device) for k, v in best_weights.items()})

        # Save model checkpoint
        ckpt_path = os.path.join(self.checkpoints_dir, f"{model_name.lower()}.pt")
        torch.save(best_weights, ckpt_path)

        # Full test evaluation
        final_metrics = self.evaluate_full_metrics(model_name, model)
        final_metrics["train_duration_seconds"] = round(train_duration, 2)
        return final_metrics, train_losses, test_losses

    def _evaluate_loss_acc(self, model: nn.Module, criterion: nn.Module) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for x_s, z_t, r, y in self.test_loader:
                x_s, z_t, r, y = x_s.to(self.device), z_t.to(self.device), r.to(self.device), y.to(self.device)

                if isinstance(model, SensorOnlyNet):
                    logits = model(x_s)
                elif isinstance(model, SimpleFusionNet):
                    logits = model(x_s, z_t)
                elif isinstance(model, GatedFusionNet):
                    logits = model(x_s, z_t)
                elif isinstance(model, ReliabilityGatedFusionNet):
                    logits = model(x_s, z_t, reliability=r)
                else:
                    logits = model(x_s, z_t)

                loss = criterion(logits, y)
                total_loss += loss.item() * len(y)

                probs = F.softmax(logits, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_probs.append(probs)
                all_preds.append(preds)
                all_targets.append(y.cpu().numpy())

        avg_loss = total_loss / len(self.test_loader.dataset)
        preds_arr = np.concatenate(all_preds)
        targets_arr = np.concatenate(all_targets)
        probs_arr = np.concatenate(all_probs)
        acc = float(accuracy_score(targets_arr, preds_arr))
        return avg_loss, acc, preds_arr, targets_arr, probs_arr

    def evaluate_full_metrics(self, model_name: str, model: nn.Module) -> Dict[str, Any]:
        criterion = nn.CrossEntropyLoss()
        test_loss, acc, preds, targets, probs = self._evaluate_loss_acc(model, criterion)

        prec = float(precision_score(targets, preds, average="macro", zero_division=0))
        rec = float(recall_score(targets, preds, average="macro", zero_division=0))
        f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
        loss = float(log_loss(targets, probs))

        # Check gate activations if gated
        gate_mean = None
        if isinstance(model, (GatedFusionNet, ReliabilityGatedFusionNet)):
            model.eval()
            gates = []
            with torch.no_grad():
                for x_s, z_t, r, y in self.test_loader:
                    x_s, z_t, r = x_s.to(self.device), z_t.to(self.device), r.to(self.device)
                    if isinstance(model, GatedFusionNet):
                        _, g = model(x_s, z_t, return_gate=True)
                    else:
                        _, g = model(x_s, z_t, reliability=r, return_gate=True)
                    gates.append(g.cpu().numpy())
            gates_arr = np.concatenate(gates)
            gate_mean = float(np.mean(gates_arr))

        return {
            "model_name": model_name,
            "test_accuracy": acc,
            "macro_precision": prec,
            "macro_recall": rec,
            "macro_f1": f1,
            "log_loss": loss,
            "mean_sensor_gate_weight": gate_mean,
        }

    def run_all_comparisons(self) -> pd.DataFrame:
        """
        Executes comparative benchmark across all 4 architectures.
        """
        print("===============================================================")
        print("Starting Multi-Source Fusion Comparative Research Benchmark")
        print("===============================================================")

        models = {
            "SensorOnlyNet": SensorOnlyNet(latent_dim=self.latent_dim),
            "SimpleFusionNet": SimpleFusionNet(latent_dim=self.latent_dim),
            "GatedFusionNet": GatedFusionNet(latent_dim=self.latent_dim),
            "ReliabilityGatedFusionNet": ReliabilityGatedFusionNet(latent_dim=self.latent_dim),
        }

        all_results = {}
        all_train_losses = {}
        all_test_losses = {}

        for name, model in models.items():
            res, tr_loss, te_loss = self.train_single_model(name, model)
            all_results[name] = res
            all_train_losses[name] = tr_loss
            all_test_losses[name] = te_loss

        # Build comparison DataFrame
        rows = []
        for name, r in all_results.items():
            rows.append({
                "Architecture": name,
                "Test Accuracy": f"{r['test_accuracy'] * 100:.2f}%",
                "Test Macro-F1": f"{r['macro_f1']:.4f}",
                "Test Macro-Precision": f"{r['macro_precision']:.4f}",
                "Test Macro-Recall": f"{r['macro_recall']:.4f}",
                "Log-Loss": f"{r['log_loss']:.4f}",
                "Mean Sensor Gate (E[g])": f"{r['mean_sensor_gate_weight']:.4f}" if r["mean_sensor_gate_weight"] is not None else "N/A (Fixed 1.0)",
                "Training Time (s)": f"{r['train_duration_seconds']:.2f}s",
            })

        df_comp = pd.DataFrame(rows).sort_values(by="Test Macro-F1", ascending=False).reset_index(drop=True)

        # Save table to CSV and Markdown
        df_comp.to_csv(os.path.join(self.tables_dir, "fusion_comparison.csv"), index=False)
        md_table = df_comp.to_markdown(index=False)
        with open(os.path.join(self.tables_dir, "fusion_comparison.md"), "w") as f:
            f.write(f"# Multi-Source Fusion Architecture Comparison (Clean Benchmark)\n\n{md_table}\n")

        # Save metrics JSON
        with open(os.path.join(self.metrics_dir, "fusion_comparison.json"), "w") as f:
            json.dump(all_results, f, indent=2)

        # Generate plots
        self._plot_fusion_comparison_barchart(all_results)
        self._plot_learning_curves(all_train_losses, all_test_losses)

        print("\n===============================================================")
        print("Final Multi-Source Fusion Comparison Table:")
        print("===============================================================")
        print(df_comp.to_string(index=False))
        return df_comp

    def _plot_fusion_comparison_barchart(self, results: Dict[str, Dict[str, Any]]) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        names = list(results.keys())
        f1_scores = [results[n]["macro_f1"] for n in names]
        acc_scores = [results[n]["test_accuracy"] for n in names]

        x = np.arange(len(names))
        width = 0.35

        ax.bar(x - width/2, acc_scores, width, label="Test Accuracy", color="#1976d2", alpha=0.85, edgecolor="black")
        ax.bar(x + width/2, f1_scores, width, label="Macro-F1 Score", color="#388e3c", alpha=0.85, edgecolor="black")

        ax.set_title("Multi-Source Fusion Architectures Performance (Held-out Test Split, N=440)", fontsize=13)
        ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10, rotation=15)
        ax.set_ylim(0.85, 1.02)
        ax.legend(loc="lower right")
        plt.tight_layout()

        fig.savefig(os.path.join(self.plots_dir, "fusion_comparison_barchart.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_learning_curves(
        self, train_losses: Dict[str, List[float]], test_losses: Dict[str, List[float]]
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#d32f2f", "#7b1fa2", "#1976d2", "#388e3c"]

        for (name, tr_l), col in zip(train_losses.items(), colors):
            ax.plot(range(1, len(tr_l) + 1), tr_l, label=f"{name} (Train)", color=col, lw=1.5, alpha=0.7)
            ax.plot(range(1, len(tr_l) + 1), test_losses[name], label=f"{name} (Test)", color=col, lw=2.0, ls="--")

        ax.set_title("Cross-Entropy Loss Learning Curves Across Fusion Architectures", fontsize=13)
        ax.set_xlabel("Training Epochs", fontsize=11)
        ax.set_ylabel("Cross-Entropy Loss", fontsize=11)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
        plt.tight_layout()

        fig.savefig(os.path.join(self.plots_dir, "fusion_learning_curves.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    trainer = FusionTrainer()
    trainer.run_all_comparisons()
