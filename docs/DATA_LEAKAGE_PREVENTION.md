# Data Leakage Prevention Protocol & Academic Safeguards

**Project Title:** Explainable Multi-Source Soil Health Intelligence System Using Machine Learning, RAG and Gated Fusion  
**Author:** Angel Oza  
**Document Purpose:** Mandatory rules and architectural constraints ensuring zero data leakage across preprocessing, RAG retrieval, model training, and experimental evaluation.

---

## 1. Principles of Data Leakage Prevention

In multi-source systems combining Machine Learning with Retrieval-Augmented Generation (RAG), data leakage can occur through three distinct failure modes:
1. **Classical Feature Leakage:** Fitting preprocessing transformers (scalers, encoders, imputers) on the full dataset rather than the training partition alone.
2. **Retrieval / Corpus Leakage:** Including test sample IDs, exact tabular feature rows, or explicit classification lookup rules in the RAG vector knowledge base.
3. **Prompt / Context Leakage:** Injecting target label hints or ground-truth classifications into the retrieval query or LLM context during inference.

This document formalizes the protocols implemented to eliminate all three failure modes.

---

## 2. Six Mandatory Anti-Leakage Protocols

### Protocol 1: Strict Train/Test Partitioning
- A single, immutable stratified split is established before any model training or feature transformation:
  $$\mathcal{D}_{\text{train}} \; (80\%, \; N=1760), \quad \mathcal{D}_{\text{test}} \; (20\%, \; N=440)$$
- Fixed random seed (`seed = 42`) is enforced across all experiments.
- The test set $\mathcal{D}_{\text{test}}$ is held out in an isolated partition and is evaluated **only once** per trained checkpoint.
- All cross-validation ($k=5$ stratified folds) is conducted strictly inside $\mathcal{D}_{\text{train}}$.

### Protocol 2: Isolated Feature Preprocessing Pipeline
- All numerical scaling (e.g., `StandardScaler`, `MinMaxScaler`) and categorical encoders are fitted **strictly on $\mathcal{D}_{\text{train}}$**:
  $$\mu_{\text{train}}, \, \sigma_{\text{train}} = \text{Fit}(\mathcal{D}_{\text{train}})$$
  $$X_{\text{train}}^{\text{scaled}} = \frac{X_{\text{train}} - \mu_{\text{train}}}{\sigma_{\text{train}}}, \quad X_{\text{test}}^{\text{scaled}} = \frac{X_{\text{test}} - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
- No statistics (mean, variance, min, max) from $\mathcal{D}_{\text{test}}$ are ever computed during the fitting stage.

### Protocol 3: Decoupled RAG Knowledge Base Construction
- The agricultural scientific corpus is constructed exclusively from **external agronomic literature** (FAO, USDA, ICAR publications).
- **Corpus Ban List:** The RAG corpus is strictly prohibited from containing:
  - Tabular rows or CSV records from Dataset C.
  - Hardcoded if-else decision rules mapped to specific sample numbers (e.g., *"Sample 42 with N=90 is Rice"*).
  - Benchmark sample identifiers or dataset metadata.
- Text chunks contain general agronomic descriptions (e.g., physiological temperature tolerances, nitrogen mineralization dynamics, transpiration ratios).

### Protocol 4: Query Formulation Without Label Clues
- Retrieval queries are synthesized dynamically from the **observed continuous soil parameters** and general agronomic inquiry templates:
  - *Compliant Query:* `"Agronomic requirements for soil with pH 6.2, nitrogen 85 kg/ha, high moisture 82%, warm temperature 24C. Nutrient uptake and crop management guidelines."`
  - *Prohibited Query:* `"Which crop is suitable for sample 123?"` or `"Is this sample rice?"`
- Query generators do not receive or query against the true ground-truth label $y$.

### Protocol 5: Independent Evaluation of Baselines vs. Fusion
- When evaluating the Gated Fusion model against baselines, all models are evaluated on the exact same held-out test split $\mathcal{D}_{\text{test}}$.
- The text representation $e_t$ for a test sample is generated solely by retrieving passages matching its observed sensor query from the external corpus, ensuring zero exposure to target labels.

### Protocol 6: Automated Leakage Auditing Tests
- Automated unit tests (`tests/test_leakage.py`) run before every benchmark execution:
  1. `test_scaler_train_only()`: Verifies that test set mean and standard deviation differ from the fitted scaler parameters.
  2. `test_rag_corpus_no_test_labels()`: Verifies that no test dataset rows or sample IDs exist in the FAISS vector database.
  3. `test_train_test_disjointness()`: Verifies that the intersection of training indices and testing indices is empty.

---

## 3. Data Leakage Compliance Checklist

| Safeguard Item | Implementation Mechanism | Verification Test | Status |
|---|---|---|---|
| **Train/Test Isolation** | Stratified $80/20$ split (`random_state=42`) | `tests/test_leakage.py::test_train_test_disjointness` | Enforced |
| **Transformer Fit Scope** | `Pipeline(steps=[('scaler', StandardScaler())])` fit on train | `tests/test_leakage.py::test_scaler_train_only` | Enforced |
| **RAG Corpus Integrity** | Curated FAO/USDA/ICAR texts only | `tests/test_leakage.py::test_rag_corpus_no_test_labels` | Enforced |
| **Query Neutrality** | Parameter-based template without label variables | `tests/test_leakage.py::test_query_neutrality` | Enforced |
| **Held-Out Test Freeze** | Test split loaded read-only during inference | `tests/test_models.py::test_reproducibility` | Enforced |
