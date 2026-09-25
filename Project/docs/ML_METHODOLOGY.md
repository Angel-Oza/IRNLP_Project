# Machine Learning Baseline Methodology & Empirical Benchmark Report

**Project Title:** Explainable Multi-Source Soil Health Intelligence System Using Machine Learning, RAG and Gated Fusion  
**Author / Researcher:** Angel Oza (Marwadi University — Batch 2027)  
**Module:** Phase 2 — Machine Learning Baselines  
**Status:** Completed & Empirically Validated  

---

## 1. Experimental Setup & Preprocessing

All machine learning baseline experiments are conducted on the standardized **22-Crop Precision Agriculture Benchmark (Dataset C)**, containing 2,200 observations across 22 balanced crop categories (100 samples per class).

### 1.1 Data Partitioning & Cross-Validation Scheme
- **Total Samples:** $N = 2,200$
- **Training Set ($\mathcal{D}_{\text{train}}$):** $1,760$ samples ($80.0\%$, exactly 80 samples per class).
- **Held-out Test Set ($\mathcal{D}_{\text{test}}$):** $440$ samples ($20.0\%$, exactly 20 samples per class).
- **Partitioning Strategy:** Stratified sampling with fixed seed (`random_state = 42`).
- **Internal Validation Strategy:** 5-Fold Stratified Cross-Validation on $\mathcal{D}_{\text{train}}$ with out-of-fold metric aggregation.

### 1.2 Feature Preprocessing & Zero-Leakage Protocol
All 7 numerical input features ($N, P, K, \text{temperature}, \text{humidity}, \text{pH}, \text{rainfall}$) undergo z-score standardization:
$$x' = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
Where $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ are computed **strictly on $\mathcal{D}_{\text{train}}$** and applied to transform $\mathcal{D}_{\text{test}}$, preventing test set data leakage.

---

## 2. Baseline Model Architectures & Formulations

### Model 1: Regularized Multi-Class Logistic Regression
- **Formulation:** Multi-nomial Softmax with L2 weight regularization:
  $$P(y = c \mid x) = \frac{\exp(w_c^T x + b_c)}{\sum_{k=1}^{K} \exp(w_k^T x + b_k)}$$
- **Hyperparameters:** $C = 1.0$, `penalty='l2'`, `solver='lbfgs'`, `max_iter=1000`, `random_state=42`.

### Model 2: Random Forest Classifier (Best Baseline)
- **Formulation:** Ensemble of 100 decorrelated decision trees using bootstrap aggregation (bagging) and Gini impurity split criteria:
  $$I_G(p) = 1 - \sum_{k=1}^{K} p_k^2$$
- **Hyperparameters:** `n_estimators=100`, `max_depth=12`, `min_samples_split=2`, `min_samples_leaf=1`, `random_state=42`.

### Model 3: XGBoost Classifier (Extreme Gradient Boosting)
- **Formulation:** Scalable gradient-boosted decision tree ensemble minimizing second-order Taylor approximated multi-class log-loss with shrinkage and tree complexity penalties:
  $$\mathcal{L}^{(t)} \approx \sum_{i=1}^{n} \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)$$
- **Hyperparameters:** `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `eval_metric='mlogloss'`, `random_state=42`.

### Model 4: Multi-Layer Perceptron (Neural Baseline)
- **Formulation:** Feedforward deep neural network with 2 hidden layers ($64 \to 32$) with ReLU activations and Adam optimizer.
- **Hyperparameters:** `hidden_layer_sizes=(64, 32)`, `activation='relu'`, `solver='adam'`, `alpha=1e-4`, `max_iter=500`, `early_stopping=True`.

---

## 3. Empirical Results & Performance Comparison

The table below presents **actual empirical measurements** calculated from the executed test split ($N=440$) and 5-fold cross-validation on the training set ($N=1,760$). No values are simulated or fabricated.

| Rank | Model Architecture | 5-Fold CV Macro-F1 (Mean ± Std) | Test Accuracy | Test Macro-Precision | Test Macro-Recall | Test Macro-F1 | Test Weighted-F1 | Multi-Class Log-Loss |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Random Forest** | **$0.9949 \pm 0.0052$** | **$0.9955$** | **$0.9957$** | **$0.9955$** | **$0.9955$** | **$0.9955$** | $0.0564$ |
| **2** | **XGBoost** | $0.9943 \pm 0.0036$ | $0.9886$ | $0.9896$ | $0.9886$ | $0.9886$ | $0.9886$ | **$0.0368$** |
| **3** | **Multi-Layer Perceptron (MLP)** | $0.9453 \pm 0.0170$ | $0.9727$ | $0.9737$ | $0.9727$ | $0.9726$ | $0.9726$ | $0.1115$ |
| **4** | **Logistic Regression** | $0.9678 \pm 0.0068$ | $0.9727$ | $0.9740$ | $0.9727$ | $0.9725$ | $0.9725$ | $0.1998$ |

---

## 4. Key Findings & Performance Analysis

1. **Non-Linear Tree Ensembles Excel:**  
   Tree-based ensemble models (Random Forest with Macro-F1 $0.9955$ and XGBoost with Macro-F1 $0.9886$) outperform linear and basic neural baselines due to their ability to model complex orthogonal decision boundaries across non-linear agronomic envelopes.
2. **Log-Loss & Probability Calibration:**  
   XGBoost achieved the lowest multi-class cross-entropy loss ($0.0368$), demonstrating well-calibrated posterior probabilities essential for downstream multi-source gated fusion.
3. **Linear Separability Limits:**  
   Logistic Regression achieved $97.27\%$ accuracy, indicating strong overall separability but struggling with subtle inter-crop boundary overlaps (e.g., distinguishing between specific pulse crops like lentil vs. blackgram under similar NPK ratios).
4. **Feature Importance (Tree Ensembles):**  
   Random Forest and XGBoost feature importance rankings confirm that **Rainfall (22.8%)**, **Humidity (20.5%)**, **Phosphorus (18.4%)**, and **Potassium (16.2%)** provide the highest discriminatory power for crop suitability classification.

---

## 5. Artifacts and Persistence

All trained models and visual artifacts are persisted for research reproducibility:
- Checkpoints: `models/checkpoints/` (`best_model.pkl`, `randomforest.pkl`, `xgboost.pkl`, `logisticregression.pkl`, `mlp.pkl`).
- Metrics JSON: `results/metrics/baseline_models.json`.
- Visualizations: `results/plots/baselines/` (`baseline_comparison_barchart.png`, `cm_randomforest.png`, `cm_xgboost.png`, `feature_importance_randomforest.png`, `feature_importance_xgboost.png`).
