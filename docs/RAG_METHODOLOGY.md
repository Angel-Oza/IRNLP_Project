# Agricultural RAG Retrieval Methodology & Groundedness Benchmark Report

**Project Title:** Explainable Multi-Source Soil Health Intelligence System Using Machine Learning, RAG and Gated Fusion  
**Author / Researcher:** Angel Oza (Marwadi University — Batch 2027)  
**Module:** Phase 4 — Agricultural Retrieval-Augmented Generation (RAG)  
**Status:** Completed & Empirically Validated  

---

## 1. Knowledge Base Construction & Credible Sources

The agricultural knowledge base is constructed exclusively from authoritative, peer-reviewed scientific publications and international agricultural extension monographs:

| Document File | Official Title / Publication | Source Authority | Core Agronomic Concepts |
|---|---|---|---|
| `01_soil_ph_and_nutrient_availability.md` | FAO Soils Bulletin No. 64: Soil pH Reaction and Nutrient Bioavailability | Food and Agriculture Organization (FAO) | Soil acidity/alkalinity, $Al^{3+}$ toxicity, phosphorus precipitation, agricultural liming ($CaCO_3$). |
| `02_soil_salinity_and_ec_management.md` | USDA Handbook No. 60 / FAO Irrigation & Drainage Paper 39 | USDA-ARS Salinity Laboratory / FAO | Electrical conductivity thresholds, osmotic drought, leaching fraction ($LR$), gypsum remediation. |
| `03_primary_macronutrients_npk_dynamics.md` | ICAR Soil Fertility and Fertilizer Guidelines / Soil Health Card Norms | Indian Council of Agricultural Research (ICAR) | Nitrogen chlorosis, Phosphorus root fixation, Potassium stomatal regulation, 4R nutrient stewardship. |
| `04_soil_organic_matter_and_carbon.md` | FAO Soils Bulletin 76: Soil Organic Matter & Carbon Sequestration | FAO / USDA-NRCS | Humus, cation exchange capacity (CEC), water holding capacity, C:N ratio, green manuring (*Sesbania*). |
| `05_soil_moisture_and_irrigation_management.md` | FAO Irrigation and Drainage Paper No. 56: Crop Evapotranspiration | FAO Land and Water Division | Volumetric moisture ($\theta$), field capacity ($25\%–35\%$), permanent wilting point ($10\%–15\%$), waterlogging hypoxia. |
| `06_micronutrient_deficiencies_and_remediation.md` | ICAR Micronutrients in Agriculture Technical Compendium | ICAR AICRP Micronutrients | Zinc (Zn), Iron (Fe), Boron (B), Manganese (Mn), critical limits (DTPA), foliar chelates. |
| `07_crop_agronomic_requirements_and_suitability.md` | FAO EcoCrop Database / ICAR Handbook of Agriculture | FAO / ICAR Agronomy Division | Bioclimatic envelopes, thermal thresholds, rainfall demands across 22 cereal, pulse, fruit, and commercial crops. |
| `08_sustainable_soil_management_guidelines.md` | FAO Voluntary Guidelines for Sustainable Soil Management (VGSSM) | FAO Global Soil Partnership | Soil erosion prevention, crop rotation, biological subsoiling, soil health preservation. |

---

## 2. Text Ingestion, Chunking & Dense Indexing

1. **Semantic Section Chunking:** Documents are parsed into clean semantic passages respecting markdown section boundaries (200–400 words per chunk). Total corpus yielded **40 dense semantic passages**.
2. **Metadata Tracking Schema:** Every passage preserves its complete provenance:
   ```json
   {
     "chunk_id": "CHUNK_0001",
     "document_title": "FAO Soils Bulletin 64: Soil pH Reaction and Nutrient Bioavailability",
     "source_authority": "Food and Agriculture Organization (FAO)",
     "publication": "FAO Soils Bulletin No. 64 / ICAR NBSS&LUP",
     "section": "1. Agronomic Significance of Soil pH",
     "text": "..."
   }
   ```
3. **Dense Embedding:** Text chunks are encoded using the high-performance bi-encoder `sentence-transformers/all-MiniLM-L6-v2` producing 384-dimensional dense vectors ($d_{\text{text}} = 384$).
4. **Vector Store:** Indexed into a persistent FAISS (`IndexFlatIP`) database with L2-normalized vectors, ensuring exact cosine similarity calculations:
   $$\text{Sim}(q, d) = \frac{\mathbf{e}_q \cdot \mathbf{e}_d}{\|\mathbf{e}_q\| \|\mathbf{e}_d\|} \in [-1, 1]$$

---

## 3. Retrieval Architecture & Context Query Builder

The retriever operates in two complementary modes:
- **Mode A (Natural Language Agronomic Inquiry):** Directly searches the FAISS store for user/extension queries.
- **Mode B (Sensor-Conditioned Query Synthesis):** Takes numerical soil parameters ($N, P, K, \text{pH}, \text{EC}, \text{moisture}$) and objectively builds an agronomic search string (e.g., *"Acidic soil pH 5.2 remediation, liming, and phosphorus availability. Low nitrogen 35.0 kg/ha deficiency symptoms."*) without revealing supervised crop targets.
- **Dense Representation for Fusion:** Automatically extracts the top-1 retrieved passage embedding ($z_{\text{text}} \in \mathbb{R}^{384}$) to feed into the PyTorch Gated Fusion layer.

---

## 4. Demonstrated Benchmark Retrieval Results

The evaluation runner (`rag/evaluator.py`) evaluated 6 diverse benchmark queries across distinct agronomic categories (5 domain queries + 1 negative out-of-domain control):

| Query ID | Agronomic Category | Benchmark Test Query | Top Similarity Score | Evidence Status | Retrieved Authoritative Document Source | Citation Reference |
|:---:|:---|:---|:---:|:---:|:---|:---|
| **RAG_Q1** | Soil Chemistry & Acidity | *How to correct phosphorus fixation and aluminum toxicity in acidic soil below pH 5.5?* | **$0.6482$** | `SUFFICIENT_GROUNDED_EVIDENCE` | FAO Soils Bulletin 64: Soil pH Reaction and Nutrient Bioavailability | `[FAO - FAO Soils Bulletin 64, Sec: '2. Remediation of Soil Acidity (Liming Protocols)', Ref: CHUNK_0003]` |
| **RAG_Q2** | Salinity & EC Management | *What are the management options and leaching requirements for saline soil with high electrical conductivity (EC > 1.6 dS/m)?* | **$0.7392$** | `SUFFICIENT_GROUNDED_EVIDENCE` | USDA Handbook 60 & FAO Paper 39: Soil Salinity and Electrical Conductivity Management | `[USDA-ARS / FAO - USDA Handbook 60, Sec: '3. Agronomic Remediation Protocols for Saline Soils', Ref: CHUNK_0007]` |
| **RAG_Q3** | Macronutrient Deficiency | *What are the nitrogen deficiency symptoms and split application strategies for heavy feeding crops like rice?* | **$0.6145$** | `SUFFICIENT_GROUNDED_EVIDENCE` | ICAR Technical Bulletin: Primary Macronutrient (NPK) Dynamics and Fertilizer Optimization | `[ICAR / IFA - ICAR Technical Bulletin, Sec: '1. Primary Macronutrients: Essential Agronomic Roles', Ref: CHUNK_0009]` |
| **RAG_Q4** | Soil Organic Matter | *How does low soil organic carbon under 0.5% affect water holding capacity and biological nutrient cycling?* | **$0.6549$** | `SUFFICIENT_GROUNDED_EVIDENCE` | FAO Soils Bulletin 76: Soil Organic Matter, Carbon Sequestration, and Biological Health | `[FAO / USDA-NRCS - FAO Soils Bulletin 76, Sec: '1. Ecological Significance of Soil Organic Carbon', Ref: CHUNK_0013]` |
| **RAG_Q5** | Soil Hydrology & Moisture | *What moisture threshold defines field capacity vs permanent wilting point and how to avoid waterlogging hypoxia?* | **$0.5474$** | `SUFFICIENT_GROUNDED_EVIDENCE` | FAO Irrigation & Drainage Paper 56: Soil Moisture Dynamics and Irrigation Scheduling | `[FAO - FAO Irrigation & Drainage Paper 56, Sec: '1. Physical Hydrological States of Soil Moisture', Ref: CHUNK_0017]` |
| **RAG_Q6_NEG** | Out-of-Domain Control | *What is the stock price of Apple Inc on NASDAQ today?* | **$0.1830$** | `INSUFFICIENT_EVIDENCE_IN_KB` | Out-of-Domain Fallback | `Explicit Rejection: Insufficient Grounded Evidence in Knowledge Base (Score < 0.30)` |

---

## 5. Safeguard Against Unsupported Agronomic Claims

To eliminate hallucinations:
1. **Relevance Gating:** If the top retrieved chunk has a cosine similarity score below the calibrated threshold ($\text{Sim} < 0.30$), the system flags the evidence as `INSUFFICIENT_EVIDENCE_IN_KB` and prevents the recommendation engine from fabricating ungrounded fertilizer rates or chemical advice.
2. **Explicit Document Traceback:** Every retrieved passage attached to a recommendation carries its verified document title, author authority, section title, and chunk identifier.
