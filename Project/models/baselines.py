"""
Machine Learning Baselines Module for Explainable Multi-Source Soil Health Intelligence System.

Implements standard supervised classifiers:
1. Logistic Regression (L2 regularized, L-BFGS solver)
2. Random Forest (100 estimators, Gini criterion)
3. XGBoost Classifier (Gradient boosted trees)
4. Multi-Layer Perceptron (2-layer neural baseline)
"""

import os
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


def get_baseline_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns a dictionary of configured supervised classification baseline models.
    All models enforce deterministic random seeds.
    """
    models = {
        "LogisticRegression": LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric="mlogloss",
            n_jobs=-1,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            max_iter=500,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
        ),
    }
    return models
