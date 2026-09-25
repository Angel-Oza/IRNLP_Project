"""
Data Models and Schemas for Agricultural Advisory and Evidence Grounding Layer.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class EvidenceReference:
    """Represents a verified scientific passage retrieved from the knowledge base."""
    chunk_id: str
    source_authority: str
    document_title: str
    section: str
    citation: str
    relevance_score: float
    snippet: str


@dataclass
class RecommendationItem:
    """Represents an actionable agronomic advisory item backed by scientific evidence."""
    topic: str
    issue_detected: str
    action: str
    rationale: str
    urgency: str  # "HIGH", "MEDIUM", "LOW"
    validation_status: str  # "VALIDATED", "LIMITED_EVIDENCE", "UNSUPPORTED"
    supporting_evidence: List[EvidenceReference] = field(default_factory=list)


@dataclass
class SoilHealthSummary:
    """Summary of Soil Health Index (SHI) status."""
    shi_score: float
    category: str
    strengths: List[str]
    limiting_factors: List[str]
    parameter_scores: Dict[str, float]


@dataclass
class CropPredictionSummary:
    """Summary of ML / Gated Fusion Crop Suitability Prediction."""
    predicted_crop: str
    confidence: float
    key_positive_drivers: List[str]
    key_negative_drivers: List[str]


@dataclass
class EvidenceValidationSummary:
    """Summary of evidence grounding validation metrics."""
    total_recommendations: int
    validated_count: int
    limited_evidence_count: int
    unsupported_count: int
    grounding_pass_rate: float
    average_evidence_count: float


@dataclass
class AgronomicAdvisoryPayload:
    """Comprehensive, explainable advisory output payload."""
    timestamp: str
    input_soil_data: Dict[str, float]
    soil_health_summary: SoilHealthSummary
    crop_prediction: Optional[CropPredictionSummary]
    recommendations: List[RecommendationItem]
    evidence_validation: EvidenceValidationSummary
    provenance_sources: List[str]
    disclaimers_and_limitations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Converts the payload to a JSON-serializable dictionary."""
        return asdict(self)
