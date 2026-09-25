# Empirical Experiment Design & Benchmarking Protocol

**Project Title:** Explainable Multi-Source Soil Health Intelligence System Using Machine Learning, RAG and Gated Fusion  
**Author:** Angel Oza  
**Document Purpose:** Complete specification of the empirical evaluation framework, hypothesis definitions, input/output structures, and metrics across Experiments A through F.

---

## 1. Overview of Experimental Framework

The evaluation framework is designed to quantitatively assess the contributions of multi-source information fusion, dynamic gating, and sensor reliability weighting under both nominal (clean) and degraded (noisy/drifting) operating regimes.

All experiments are executed using an automated, deterministic evaluation runner (`experiments/run_all.py`) and log all metrics to `results/metrics/`, `results/plots/`, and `results/tables/`.

---

## 2. Detailed Specification of Experiments A through F

### Experiment A: Numerical ML Baselines (Sensor-Only)
- **Research Hypothesis:** Standard supervised machine learning classifiers trained strictly on numerical sensor features establish the baseline performance under clean conditions.
- **Input:** Standardized numerical vector $x_s \in \mathbb{R}^7$ ($N, P, K, \text{temp}, \text{humidity}, \text{pH}, \text{rainfall}$).
- **Output:** Class probabilities $\hat{y} \in \mathbb{R}^{22}$.
- **Training Data:** $80\%$ stratified training split $\mathcal{D}_{\text{train}}$ ($N=1760$).
- **Evaluation Data:** $20\%$ held-out test split $\mathcal{D}_{\text{test}}$ ($N=440$).
- **Models Evaluated:**
  1. Logistic Regression (L2 regularization, L-BFGS solver)
  2. Random Forest (100 estimators, Gini impurity)
  3. XGBoost Classifier (tree-depth=6, learning_rate=0.1)
  4. Multi-Layer Perceptron (MLP: 2 hidden layers, ReLU activation)
- **Metrics Recorded:** Accuracy, Macro-Precision, Macro-Recall, Macro-F1, Confusion Matrix.

---

### Experiment B: Numerical ML + RAG Contextual Baseline (Prompting / Late Fusion)
- **Research Hypothesis:** Naively incorporating retrieved agronomic text via prompt-based concatenation or late soft voting provides marginal improvement but lacks adaptive cross-modal synergy.
- **Input:** Numerical vector $x_s$ concatenated with a fixed text embedding $z_{\text{text}}$ retrieved from the agronomic corpus.
- **Output:** Class prediction $\hat{y}$.
- **Training & Evaluation Data:** Identical $\mathcal{D}_{\text{train}}$ and $\mathcal{D}_{\text{test}}$.
- **Model:** Late-fusion ensemble combining ML probability distribution with text retrieval score distribution.
- **Metrics Recorded:** Accuracy, Macro-F1, Cross-Entropy Loss, Prediction Variance.

---

### Experiment C: Numerical + Simple Fusion Network (Unweighted Concatenation)
- **Research Hypothesis:** Jointly projecting numerical and textual representations into a shared latent space via unweighted concatenation improves cross-modal representations over separate baselines.
- **Input:** $x_s \in \mathbb{R}^7$ and dense retrieved embedding $z_{\text{text}} \in \mathbb{R}^{384}$.
- **Output:** Class logits $\hat{y} \in \mathbb{R}^{22}$.
- **Architecture:**
  $$e_s = \text{LayerNorm}(\text{ReLU}(W_s x_s + b_s)) \in \mathbb{R}^{64}$$
  $$e_t = \text{LayerNorm}(\text{ReLU}(W_t z_{\text{text}} + b_t)) \in \mathbb{R}^{64}$$
  $$z_{\text{fused}} = \text{ReLU}(W_c [e_s \,\|\, e_t] + b_c) \in \mathbb{R}^{64}$$
  $$\hat{y} = \text{Softmax}(W_o z_{\text{fused}} + b_o)$$
- **Training Data:** $\mathcal{D}_{\text{train}}$ trained with Adam optimizer ($\text{lr}=10^{-3}$, weight decay $=10^{-4}$) and Cross-Entropy loss.
- **Evaluation Data:** Held-out $\mathcal{D}_{\text{test}}$.
- **Metrics Recorded:** Test Accuracy, Macro-F1, Validation Loss curve.

---

### Experiment D: Numerical + Gated Fusion Network (Dynamic Sigmoid Gating)
- **Research Hypothesis:** A dynamic sigmoid gating mechanism allows the network to adaptively modulate the contribution of sensor features versus external scientific knowledge per instance.
- **Input:** $x_s \in \mathbb{R}^7$ and $z_{\text{text}} \in \mathbb{R}^{384}$.
- **Architecture:**
  $$e_s = \text{LayerNorm}(\text{ReLU}(W_s x_s + b_s)) \in \mathbb{R}^{64}$$
  $$e_t = \text{LayerNorm}(\text{ReLU}(W_t z_{\text{text}} + b_t)) \in \mathbb{R}^{64}$$
  $$g = \sigma(W_g [e_s \,\|\, e_t] + b_g) \in (0, 1)^{64}$$
  $$z_{\text{fused}} = g \odot e_s + (1 - g) \odot e_t \in \mathbb{R}^{64}$$
  $$\hat{y} = \text{Softmax}(W_o z_{\text{fused}} + b_o)$$
- **Metrics Recorded:** Test Accuracy, Macro-F1, Gate Value Distribution $\mathbb{E}[g]$, Convergence Rate.

---

### Experiment E: Reliability-Aware Gated Fusion Network
- **Research Hypothesis:** Conditioning the dynamic gating unit on real-time sensor stream reliability $r$ allows the model to systematically down-weight compromised sensor channels in favor of authoritative domain knowledge.
- **Input:** $x_s \in \mathbb{R}^7$, $z_{\text{text}} \in \mathbb{R}^{384}$, and sensor reliability scalar $r \in [0, 1]$.
- **Architecture:**
  $$g = \sigma(W_g [e_s \,\|\, e_t \,\|\, r] + b_g) \in (0, 1)^{64}$$
  $$z_{\text{fused}} = (r \cdot g) \odot e_s + (1 - (r \cdot g)) \odot e_t \in \mathbb{R}^{64}$$
  $$\hat{y} = \text{Softmax}(W_o z_{\text{fused}} + b_o)$$
- **Metrics Recorded:** Test Accuracy, Macro-F1, Reliability-Gating Correlation ($\text{Corr}(r, \bar{g})$), Robustness Retention Rate.

---

### Experiment F: Noise Robustness & Fault Injection Benchmarking
- **Research Hypothesis:** Under progressive sensor corruption (noise, drift, missing channels), Reliability-Aware Gated Fusion exhibits significantly lower performance degradation than unweighted fusion and numerical-only baselines.
- **Fault Profiles Injected on Test Partition:**
  1. **Gaussian Noise:** Noise level $\eta \in \{0\%, 5\%, 10\%, 20\%\}$ applied as $x_s' = x_s + \mathcal{N}(0, \eta \cdot \sigma_{x_s})$.
  2. **Sensor Drift:** Linear baseline offset $\Delta x_s = \delta \cdot t$ applied to temperature and moisture channels.
  3. **Intermittent Missingness:** Random channel dropout simulating communication loss ($p_{\text{missing}} \in \{0.1, 0.25\}$).
- **Models Compared:**
  1. Sensor-Only Random Forest
  2. Sensor-Only XGBoost
  3. Sensor-Only MLP
  4. Simple Fusion Network (Exp C)
  5. Gated Fusion Network without reliability (Exp D)
  6. Reliability-Aware Gated Fusion Network (Exp E)
- **Metrics Recorded:** Accuracy vs. Noise curve, Macro-F1 Degradation Slope ($\frac{\Delta F1}{\Delta \eta}$), Relative Resilience Index ($\text{RRI} = \frac{F1_{\text{noisy}}}{F1_{\text{clean}}}$).

---

## 3. Master Experiment Summary Matrix

| Experiment ID | Title | Input Modalities | Gating / Fusion Strategy | Evaluated Conditions | Primary Research Question |
|---|---|---|---|---|---|
| **EXP-A** | Numerical ML Baselines | Numerical ($x_s$) | None (Single-source) | Clean Telemetry | Baseline Benchmark |
| **EXP-B** | Sensor + RAG Baseline | $x_s$ + $z_{\text{text}}$ | Late soft ensemble | Clean Telemetry | Does naive text integration help? |
| **EXP-C** | Simple Fusion | $x_s$ + $z_{\text{text}}$ | Unweighted Concatenation | Clean Telemetry | Does joint latent projection help? |
| **EXP-D** | Gated Fusion | $x_s$ + $z_{\text{text}}$ | Dynamic Sigmoid Gate $g$ | Clean Telemetry | Does dynamic gating outperform simple fusion? (RQ2) |
| **EXP-E** | Reliability Gated Fusion | $x_s$ + $z_{\text{text}}$ + $r$ | Reliability-weighted Gate | Clean Telemetry | Does explicit reliability awareness improve gating? (RQ3) |
| **EXP-F** | Noise Robustness Benchmark | $x_s + \text{Noise}$ + $z_{\text{text}}$ + $r$ | All models tested across noise | $\eta \in \{0\%, 5\%, 10\%, 20\%\}$ | Which architecture is most resilient to sensor degradation? (RQ3) |
