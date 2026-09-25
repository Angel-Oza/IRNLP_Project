"""
Agricultural Advisory and Evidence-Grounded Recommendation Package.
"""

from advisory.schemas import (
    EvidenceReference,
    RecommendationItem,
    SoilHealthSummary,
    CropPredictionSummary,
    EvidenceValidationSummary,
    AgronomicAdvisoryPayload,
)
from advisory.evidence import EvidenceGroundingValidator, get_evidence_validator
from advisory.templates import AgronomicRemediationTemplates
from advisory.generator import AgronomicAdvisoryGenerator, get_advisory_generator

__all__ = [
    "EvidenceReference",
    "RecommendationItem",
    "SoilHealthSummary",
    "CropPredictionSummary",
    "EvidenceValidationSummary",
    "AgronomicAdvisoryPayload",
    "EvidenceGroundingValidator",
    "get_evidence_validator",
    "AgronomicRemediationTemplates",
    "AgronomicAdvisoryGenerator",
    "get_advisory_generator",
]
