# RAG Retrieval Benchmark & Groundedness Evaluation

| Query ID   | Agronomic Category             |   Top Similarity Score | Evidence Status              | Retrieved Document Source                                                                 | Benchmark Status   |
|:-----------|:-------------------------------|-----------------------:|:-----------------------------|:------------------------------------------------------------------------------------------|:-------------------|
| RAG_Q1     | Soil Chemistry & Acidity       |                 0.6482 | SUFFICIENT_GROUNDED_EVIDENCE | FAO Soils Bulletin 64: Soil pH Reaction and Nutrient Bioavailability                      | PASSED             |
| RAG_Q2     | Salinity & EC Management       |                 0.7392 | SUFFICIENT_GROUNDED_EVIDENCE | USDA Handbook 60 & FAO Paper 39: Soil Salinity and Electrical Conductivity Management     | PASSED             |
| RAG_Q3     | Macronutrient Deficiency       |                 0.6145 | SUFFICIENT_GROUNDED_EVIDENCE | ICAR Technical Bulletin: Primary Macronutrient (NPK) Dynamics and Fertilizer Optimization | PASSED             |
| RAG_Q4     | Soil Organic Matter            |                 0.6549 | SUFFICIENT_GROUNDED_EVIDENCE | FAO Soils Bulletin 76: Soil Organic Matter, Carbon Sequestration, and Biological Health   | PASSED             |
| RAG_Q5     | Soil Hydrology & Moisture      |                 0.5474 | SUFFICIENT_GROUNDED_EVIDENCE | FAO Irrigation & Drainage Paper 56: Soil Moisture Dynamics and Irrigation Scheduling      | PASSED             |
| RAG_Q6_NEG | Out-of-Domain Negative Control |                 0.183  | INSUFFICIENT_EVIDENCE_IN_KB  | FAO EcoCrop & ICAR Compendium: Crop Ecological Growth Envelopes and Soil Compatibility    | PASSED             |
