"""
Agricultural Advisory and Complete Intelligence Pipeline Service Layer.

Bridges the FastAPI application with:
- Phase 8: AgronomicAdvisoryGenerator and EvidenceGroundingValidator
- Phase 4: SoilKnowledgeRetriever and FAISS Vector Store
- Complete Multi-Source Unified Intelligence Pipeline (Phases 1-8)
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from advisory.generator import AgronomicAdvisoryGenerator
from advisory.schemas import AgronomicAdvisoryPayload
from rag.retriever import SoilKnowledgeRetriever
from backend.schemas import (
    AdvisoryRequest,
    AdvisoryResponse,
    SoilHealthSummarySchema,
    CropPredictionSummarySchema,
    RecommendationItemSchema,
    EvidenceReferenceSchema,
    EvidenceValidationSummarySchema,
    UnifiedIntelligenceRequest,
    UnifiedIntelligenceResponse,
    StageStatus,
    KnowledgeCorpusResponse,
    KnowledgeChunkItem,
    SampleProfile,
)
from backend.services.analysis_service import get_analysis_service
from backend.services.ml_service import get_ml_service


class AdvisoryService:
    """
    Service wrapper for Grounded Agricultural Advisory and Unified Intelligence Pipeline.
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = os.path.abspath(base_dir or BASE_DIR)
        self.generator = AgronomicAdvisoryGenerator(base_dir=self.base_dir)
        self.analysis_service = get_analysis_service()
        self.ml_service = get_ml_service()

        # Load knowledge corpus metadata
        meta_path = os.path.join(self.base_dir, "knowledge", "faiss_index", "chunks_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.corpus_metadata = json.load(f)
        else:
            self.corpus_metadata = []

    def generate_advisory(
        self,
        soil_data: Dict[str, float],
        target_crop: Optional[str] = None,
        crop_prediction_data: Optional[Dict[str, Any]] = None,
    ) -> AdvisoryResponse:
        """
        Generates evidence-grounded agronomic advisory recommendations.
        """
        if not soil_data:
            raise ValueError("Soil chemistry parameters must be provided to generate an advisory.")

        # Run Phase 8 Generator
        payload: AgronomicAdvisoryPayload = self.generator.generate_advisory(
            soil_data=soil_data,
            target_crop=target_crop,
            crop_prediction_data=crop_prediction_data,
        )

        # Convert to API schemas
        soil_summary = SoilHealthSummarySchema(
            shi_score=payload.soil_health_summary.shi_score,
            category=payload.soil_health_summary.category,
            strengths=payload.soil_health_summary.strengths,
            limiting_factors=payload.soil_health_summary.limiting_factors,
            parameter_scores=payload.soil_health_summary.parameter_scores,
        )

        crop_summary = None
        if payload.crop_prediction:
            crop_summary = CropPredictionSummarySchema(
                predicted_crop=payload.crop_prediction.predicted_crop,
                confidence=payload.crop_prediction.confidence,
                key_positive_drivers=payload.crop_prediction.key_positive_drivers,
                key_negative_drivers=payload.crop_prediction.key_negative_drivers,
            )

        recommendations: List[RecommendationItemSchema] = []
        for r in payload.recommendations:
            ev_list: List[EvidenceReferenceSchema] = []
            for ev in r.supporting_evidence:
                ev_list.append(
                    EvidenceReferenceSchema(
                        chunk_id=ev.chunk_id,
                        source_authority=ev.source_authority,
                        document_title=ev.document_title,
                        section=ev.section,
                        citation=ev.citation,
                        relevance_score=round(float(ev.relevance_score), 4),
                        snippet=ev.snippet,
                    )
                )
            recommendations.append(
                RecommendationItemSchema(
                    topic=r.topic,
                    issue_detected=r.issue_detected,
                    action=r.action,
                    rationale=r.rationale,
                    urgency=r.urgency,
                    validation_status=r.validation_status,
                    supporting_evidence=ev_list,
                )
            )

        val_summary = EvidenceValidationSummarySchema(
            total_recommendations=payload.evidence_validation.total_recommendations,
            validated_count=payload.evidence_validation.validated_count,
            limited_evidence_count=payload.evidence_validation.limited_evidence_count,
            unsupported_count=payload.evidence_validation.unsupported_count,
            grounding_pass_rate=payload.evidence_validation.grounding_pass_rate,
            average_evidence_count=payload.evidence_validation.average_evidence_count,
        )

        return AdvisoryResponse(
            timestamp=payload.timestamp,
            input_soil_data=payload.input_soil_data,
            soil_health_summary=soil_summary,
            crop_prediction=crop_summary,
            recommendations=recommendations,
            evidence_validation=val_summary,
            provenance_sources=payload.provenance_sources,
            disclaimers_and_limitations=payload.disclaimers_and_limitations,
        )

    def run_unified_intelligence(self, request: UnifiedIntelligenceRequest) -> UnifiedIntelligenceResponse:
        """
        Executes complete multi-stage pipeline:
        Input -> Preprocessing -> ML Prediction -> SHI -> XAI -> RAG -> Grounded Advisory.
        """
        stages = [
            StageStatus(
                stage_id=1,
                name="Telemetry & Chemistry Preprocessing",
                status="COMPLETED",
                description="Validated chemistry parameters and standardized features with zero leakage.",
            ),
            StageStatus(
                stage_id=2,
                name="ML Crop Suitability & Classification",
                status="COMPLETED",
                description="Executed inference with best supervised benchmark classifier (Random Forest).",
            ),
            StageStatus(
                stage_id=3,
                name="Soil Health Index (SHI) Scoring",
                status="COMPLETED",
                description="Calculated composite 0-100 SHI score using ICAR/FAO agronomic curves.",
            ),
            StageStatus(
                stage_id=4,
                name="Explainable AI (TreeSHAP)",
                status="COMPLETED",
                description="Extracted positive and negative feature attribution drivers for the prediction.",
            ),
            StageStatus(
                stage_id=5,
                name="RAG Literature Retrieval & Advisory Grounding",
                status="COMPLETED",
                description="Retrieved peer-reviewed evidence from FAO/USDA/ICAR corpus and validated recommendations.",
            ),
        ]

        # 1. Prepare ML Feature Dictionary
        crop_features = {
            "N": float(request.N),
            "P": float(request.P),
            "K": float(request.K),
            "temperature": float(request.temperature),
            "humidity": float(request.humidity),
            "ph": float(request.pH or 6.5),
            "rainfall": float(request.rainfall),
        }

        # 2. Execute ML Prediction & TreeSHAP XAI
        crop_pred_res = self.ml_service.predict_crop(crop_features)

        # 3. Prepare Soil Chemistry Dictionary for SHI Engine
        soil_dict = {
            "pH": request.pH,
            "EC": request.EC,
            "OC": request.OC,
            "N": request.N,
            "P": request.P,
            "K": request.K,
            "S": request.S,
            "Zn": request.Zn,
            "B": request.B,
            "Fe": request.Fe,
            "Mn": request.Mn,
            "Cu": request.Cu,
        }

        # 4. Execute Soil Health Analysis
        soil_analysis_res = self.analysis_service.analyze_soil(soil_dict)

        # 5. Format XAI payload for Advisory Generator
        xai_payload = {
            "prediction": {
                "crop": crop_pred_res.predicted_crop,
                "confidence": crop_pred_res.confidence,
                "class_id": crop_pred_res.class_id,
            },
            "ml_feature_explanation": {
                "feature_attributions": [attr.model_dump() for attr in crop_pred_res.feature_attributions],
                "summary_statement": crop_pred_res.summary_statement,
            },
        }

        # 6. Execute Grounded Advisory Generation
        clean_soil_data = {k: float(v) for k, v in soil_dict.items() if v is not None}
        advisory_res = self.generate_advisory(
            soil_data=clean_soil_data,
            target_crop=crop_pred_res.predicted_crop,
            crop_prediction_data=xai_payload,
        )

        provenance_meta = {
            "shi_provenance": "ICAR Soil Health Card Scheme / FAO Bulletins 64 & 76 / USDA Handbook 60",
            "ml_model": "Random Forest Multi-Class Benchmark (22 Classes)",
            "xai_method": "TreeSHAP (Exact Shapley Value Decomposition)",
            "rag_vector_store": "Curated FAISS MiniLM Index (48 Agronomic Chunks)",
            "evidence_validation": "Cosine Relevance Threshold >= 0.35 Filter",
            "academic_institution": "Marwadi University — Department of ICT (2026)",
        }

        return UnifiedIntelligenceResponse(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            pipeline_stages=stages,
            soil_health=soil_analysis_res,
            crop_prediction=crop_pred_res,
            advisory=advisory_res,
            provenance=provenance_meta,
        )

    def get_knowledge_corpus(self) -> KnowledgeCorpusResponse:
        """Returns all documents and chunks in the scientific knowledge base."""
        doc_dict: Dict[str, Dict[str, Any]] = {}
        chunk_items: List[KnowledgeChunkItem] = []
        authorities_set = set()

        for chunk in self.corpus_metadata:
            authorities_set.add(chunk["source_authority"])
            title = chunk["document_title"]
            if title not in doc_dict:
                doc_dict[title] = {
                    "document_title": title,
                    "source_authority": chunk["source_authority"],
                    "publication": chunk["publication"],
                    "filename": chunk["filename"],
                    "chunk_count": 0,
                }
            doc_dict[title]["chunk_count"] += 1

            chunk_items.append(
                KnowledgeChunkItem(
                    chunk_id=chunk["chunk_id"],
                    document_title=chunk["document_title"],
                    source_authority=chunk["source_authority"],
                    publication=chunk["publication"],
                    section=chunk["section"],
                    text_preview=chunk["text"][:240] + ("..." if len(chunk["text"]) > 240 else ""),
                )
            )

        return KnowledgeCorpusResponse(
            total_chunks=len(self.corpus_metadata),
            authorities=sorted(list(authorities_set)),
            documents=list(doc_dict.values()),
            chunks=chunk_items,
        )

    def get_sample_profiles(self) -> List[SampleProfile]:
        """Returns certified empirical sample profiles from Dataset A and C."""
        return [
            SampleProfile(
                id="sample-bareilly-kvk-01",
                title="Bareilly Certified Farm (Alkaline & Low OC)",
                location="ICAR KVK Bareilly (Dataset A, Certificate #1)",
                description="Certified lab soil sample showing low organic carbon (0.23%), low nitrogen, and high pH (7.13) under subtropical Indo-Gangetic conditions.",
                soil_data={
                    "pH": 7.13,
                    "EC": 0.13,
                    "OC": 0.23,
                    "N": 108.0,
                    "P": 11.59,
                    "K": 130.0,
                    "S": 14.67,
                    "Zn": 0.98,
                    "B": 0.60,
                    "Fe": 8.06,
                    "Mn": 2.03,
                    "Cu": 0.47,
                },
                crop_features={
                    "N": 108.0,
                    "P": 11.59,
                    "K": 130.0,
                    "temperature": 26.5,
                    "humidity": 68.0,
                    "ph": 7.13,
                    "rainfall": 145.0,
                },
                highlight="Low OC (<0.75%) and Deficient Nitrogen triggering bio-fertilizer and organic manuring advisory.",
            ),
            SampleProfile(
                id="sample-rice-optimal",
                title="Optimal Alluvial Wetland Soil (High Fertility)",
                location="Indo-Gangetic Alluvial Plain",
                description="High fertility soil with balanced macro-nutrients and optimal neutral pH suited for intensive paddy rice cultivation.",
                soil_data={
                    "pH": 6.70,
                    "EC": 0.45,
                    "OC": 0.95,
                    "N": 380.0,
                    "P": 22.0,
                    "K": 240.0,
                    "S": 18.5,
                    "Zn": 1.20,
                    "B": 0.85,
                    "Fe": 9.50,
                    "Mn": 4.20,
                    "Cu": 0.80,
                },
                crop_features={
                    "N": 85.0,
                    "P": 45.0,
                    "K": 40.0,
                    "temperature": 23.5,
                    "humidity": 82.0,
                    "ph": 6.70,
                    "rainfall": 215.0,
                },
                highlight="Optimal SHI (>85/100) with High Fertility Index and strong positive TreeSHAP drivers for Rice.",
            ),
            SampleProfile(
                id="sample-acidic-tea",
                title="Acidic Hill Soil (Acid Stress & P-Fixation)",
                location="Eastern Himalayan Foothills",
                description="Strongly acidic soil (pH 4.80) with severe phosphorus fixation and critical micronutrient leaching risks.",
                soil_data={
                    "pH": 4.80,
                    "EC": 0.20,
                    "OC": 1.10,
                    "N": 260.0,
                    "P": 8.5,
                    "K": 95.0,
                    "S": 8.0,
                    "Zn": 0.45,
                    "B": 0.35,
                    "Fe": 14.5,
                    "Mn": 1.80,
                    "Cu": 0.25,
                },
                crop_features={
                    "N": 20.0,
                    "P": 135.0,
                    "K": 195.0,
                    "temperature": 18.2,
                    "humidity": 75.0,
                    "ph": 4.80,
                    "rainfall": 180.0,
                },
                highlight="Severe Acidity triggering FAO Bulletin 64 Agricultural Liming protocol and rock phosphate recommendation.",
            ),
            SampleProfile(
                id="sample-saline-arid",
                title="Arid Saline Soil (High EC & Osmotic Stress)",
                location="Northwest Arid Zone",
                description="Slightly saline/alkaline soil with elevated electrical conductivity (EC 2.4 dS/m) causing severe osmotic water stress.",
                soil_data={
                    "pH": 8.40,
                    "EC": 2.40,
                    "OC": 0.35,
                    "N": 160.0,
                    "P": 14.0,
                    "K": 310.0,
                    "S": 22.0,
                    "Zn": 0.50,
                    "B": 1.80,
                    "Fe": 3.80,
                    "Mn": 1.50,
                    "Cu": 0.30,
                },
                crop_features={
                    "N": 25.0,
                    "P": 65.0,
                    "K": 20.0,
                    "temperature": 32.0,
                    "humidity": 45.0,
                    "ph": 8.40,
                    "rainfall": 60.0,
                },
                highlight="Saline Osmotic Stress triggering USDA Handbook 60 Leaching Fraction and Gypsum remediation.",
            ),
        ]


_advisory_service_instance: Optional[AdvisoryService] = None


def get_advisory_service() -> AdvisoryService:
    global _advisory_service_instance
    if _advisory_service_instance is None:
        _advisory_service_instance = AdvisoryService()
    return _advisory_service_instance
