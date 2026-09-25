"""
Baseline Models Trainer Orchestrator.

Trains baseline models using 5-fold stratified cross-validation on training data,
evaluates on held-out test data, saves model checkpoints, and logs metrics.
"""

import os
import json
import pickle
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from models.baselines import get_baseline_models
from models.evaluator import ModelEvaluator


class BaselineTrainer:
    """
    Orchestrates baseline model training, cross-validation, persistence, and evaluation.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        checkpoints_dir: Optional[str] = None,
        random_state: int = 42,
    ):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.data_dir = data_dir or os.path.join(base_dir, "data", "processed")
        self.checkpoints_dir = checkpoints_dir or os.path.join(base_dir, "models", "checkpoints")
        self.metrics_dir = os.path.join(base_dir, "results", "metrics")
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

        self.random_state = random_state

        # Load processed data
        self.X_train = np.load(os.path.join(self.data_dir, "X_train.npy"))
        self.X_test = np.load(os.path.join(self.data_dir, "X_test.npy"))
        self.y_train = np.load(os.path.join(self.data_dir, "y_train.npy"))
        self.y_test = np.load(os.path.join(self.data_dir, "y_test.npy"))

        with open(os.path.join(self.data_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)

        self.class_names = self.metadata["class_names"]
        self.feature_names = self.metadata["feature_names"]

        self.evaluator = ModelEvaluator(
            class_names=self.class_names,
            feature_names=self.feature_names,
        )

    def train_and_evaluate_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Trains and cross-validates all baseline models, evaluates on held-out test data,
        and saves checkpoints.
        """
        print(f"Starting Baseline Training Pipeline on {len(self.X_train)} training samples...")
        models = get_baseline_models(random_state=self.random_state)
        all_results = {}
        best_f1 = -1.0
        best_model_name = ""
        best_model_obj = None

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

        for name, model in models.items():
            print(f"\n--- Training & Cross-Validating {name} ---")

            # 5-fold Stratified Cross-Validation on X_train
            cv_res = cross_validate(model, self.X_train, self.y_train, cv=cv, scoring=scoring, n_jobs=-1)
            cv_summary = {
                "cv_accuracy_mean": float(np.mean(cv_res["test_accuracy"])),
                "cv_accuracy_std": float(np.std(cv_res["test_accuracy"])),
                "cv_f1_macro_mean": float(np.mean(cv_res["test_f1_macro"])),
                "cv_f1_macro_std": float(np.std(cv_res["test_f1_macro"])),
                "cv_precision_macro_mean": float(np.mean(cv_res["test_precision_macro"])),
                "cv_recall_macro_mean": float(np.mean(cv_res["test_recall_macro"])),
            }
            print(f"  5-Fold CV Macro-F1: {cv_summary['cv_f1_macro_mean']:.4f} ± {cv_summary['cv_f1_macro_std']:.4f}")

            # Fit on entire X_train
            model.fit(self.X_train, self.y_train)

            # Evaluate on held-out X_test
            test_metrics = self.evaluator.evaluate_model(
                model_name=name,
                model=model,
                X_test=self.X_test,
                y_test=self.y_test,
                cv_scores=cv_summary,
            )
            print(f"  Test Accuracy: {test_metrics['test_accuracy']:.4f} | Test Macro-F1: {test_metrics['macro_f1']:.4f}")

            # Persist individual model checkpoint
            ckpt_path = os.path.join(self.checkpoints_dir, f"{name.lower()}.pkl")
            with open(ckpt_path, "wb") as f:
                pickle.dump(model, f)

            all_results[name] = test_metrics

            if test_metrics["macro_f1"] > best_f1:
                best_f1 = test_metrics["macro_f1"]
                best_model_name = name
                best_model_obj = model

        # Persist best model
        print(f"\n=======================================================")
        print(f"Best Performing Model: {best_model_name} (Test Macro-F1: {best_f1:.4f})")
        print(f"=======================================================")
        best_path = os.path.join(self.checkpoints_dir, "best_model.pkl")
        with open(best_path, "wb") as f:
            pickle.dump(best_model_obj, f)

        # Generate comparative summary tables and charts
        df_comp = self.evaluator.generate_model_comparison(all_results)
        print("\nSummary Comparison Table:")
        print(df_comp.to_string(index=False))

        # Save all results JSON
        results_json_path = os.path.join(self.metrics_dir, "baseline_models.json")
        with open(results_json_path, "w") as f:
            json.dump(all_results, f, indent=2)

        return all_results


if __name__ == "__main__":
    trainer = BaselineTrainer()
    trainer.train_and_evaluate_all()
