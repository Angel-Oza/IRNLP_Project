"""
Pydantic Request & Response Schemas for Soil Health Intelligence API.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


# ------------------------------------------------------------------------------
# 1. Health & Status Schemas
# ------------------------------------------------------------------------------
class ComponentStatus(BaseModel):
    name: str
    status: str
    details: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    timestamp: str
    components: Dict[str, str]


# ------------------------------------------------------------------------------
# 2. Soil Analysis Schemas (Phase 7 SHI Engine)
# ------------------------------------------------------------------------------
class SoilAnalysisRequest(BaseModel):
    pH: Optional[float] = Field(None, description="Soil pH Reaction (0-14, optimal 6.0-7.5)", ge=0.0, le=14.0)
    EC: Optional[float] = Field(None, description="Electrical Conductivity in dS/m (optimal < 0.8)", ge=0.0)
    OC: Optional[float] = Field(None, description="Soil Organic Carbon in % (optimal >= 0.75%)", ge=0.0)
    N: Optional[float] = Field(None, description="Available Nitrogen in kg/ha (optimal 280-560)", ge=0.0)
    P: Optional[float] = Field(None, description="Available Phosphorus in kg/ha (optimal 10-25)", ge=0.0)
    K: Optional[float] = Field(None, description="Available Potassium in kg/ha (optimal 110-280)", ge=0.0)
    S: Optional[float] = Field(None, description="Available Sulphur in ppm (critical >= 10.0)", ge=0.0)
    Zn: Optional[float] = Field(None, description="Available Zinc in ppm (critical >= 0.60)", ge=0.0)
    B: Optional[float] = Field(None, description="Available Boron in ppm (critical >= 0.50)", ge=0.0)
    Fe: Optional[float] = Field(None, description="Available Iron in ppm (critical >= 4.50)", ge=0.0)
    Mn: Optional[float] = Field(None, description="Available Manganese in ppm (critical >= 2.00)", ge=0.0)
    Cu: Optional[float] = Field(None, description="Available Copper in ppm (critical >= 0.20)", ge=0.0)


class ParameterDetail(BaseModel):
    raw_value: float
    score: float
    status: str
    is_deficient: bool
    name: str
    unit: str
    desirable_range: str
    provenance: str


class SoilAnalysisResponse(BaseModel):
    shi_score: float
    category: str
    parameter_scores: Dict[str, float]
    normalized_weights: Dict[str, float]
    weighted_contributions: Dict[str, float]
    parameter_details: Dict[str, ParameterDetail]
    deficiencies: List[str]
    strengths: List[str]
    limiting_factors: List[str]
    agronomic_summary: str
    warnings: List[str]
    provenance_summary: str


# ------------------------------------------------------------------------------
# 3. Crop Prediction & XAI Schemas (Phase 2 & 3)
# ------------------------------------------------------------------------------
class CropPredictionRequest(BaseModel):
    N: float = Field(..., description="Nitrogen content (kg/ha)", ge=0.0)
    P: float = Field(..., description="Phosphorus content (kg/ha)", ge=0.0)
    K: float = Field(..., description="Potassium content (kg/ha)", ge=0.0)
    temperature: float = Field(..., description="Ambient temperature (°C)", ge=-20.0, le=60.0)
    humidity: float = Field(..., description="Relative humidity (%)", ge=0.0, le=100.0)
    ph: float = Field(..., description="Soil pH", ge=0.0, le=14.0)
    rainfall: float = Field(..., description="Precipitation / Rainfall (mm)", ge=0.0)


class FeatureAttributionItem(BaseModel):
    feature: str
    raw_value: float
    shap_value: float
    contribution: str  # "positive" or "negative"
    abs_importance: float


class ClassProbability(BaseModel):
    crop: str
    probability: float


class CropPredictionResponse(BaseModel):
    predicted_crop: str
    confidence: float
    class_id: int
    top_probabilities: List[ClassProbability]
    base_expected_value: float
    feature_attributions: List[FeatureAttributionItem]
    summary_statement: str
    scope_note: str


# ------------------------------------------------------------------------------
# 4. Grounded Advisory & Evidence Schemas (Phase 4 & 8)
# ------------------------------------------------------------------------------
class EvidenceReferenceSchema(BaseModel):
    chunk_id: str
    source_authority: str
    document_title: str
    section: str
    citation: str
    relevance_score: float
    snippet: str


class RecommendationItemSchema(BaseModel):
    topic: str
    issue_detected: str
    action: str
    rationale: str
    urgency: str  # "HIGH", "MEDIUM", "LOW"
    validation_status: str  # "VALIDATED", "LIMITED_EVIDENCE", "UNSUPPORTED"
    supporting_evidence: List[EvidenceReferenceSchema]


class SoilHealthSummarySchema(BaseModel):
    shi_score: float
    category: str
    strengths: List[str]
    limiting_factors: List[str]
    parameter_scores: Dict[str, float]


class CropPredictionSummarySchema(BaseModel):
    predicted_crop: str
    confidence: float
    key_positive_drivers: List[str]
    key_negative_drivers: List[str]


class EvidenceValidationSummarySchema(BaseModel):
    total_recommendations: int
    validated_count: int
    limited_evidence_count: int
    unsupported_count: int
    grounding_pass_rate: float
    average_evidence_count: float


class AdvisoryRequest(BaseModel):
    soil_data: Dict[str, float] = Field(..., description="Key-value mapping of soil chemistry parameters")
    target_crop: Optional[str] = Field(None, description="Optional target crop name")
    crop_prediction_data: Optional[Dict[str, Any]] = Field(None, description="Optional ML prediction and XAI output")


class AdvisoryResponse(BaseModel):
    timestamp: str
    input_soil_data: Dict[str, float]
    soil_health_summary: SoilHealthSummarySchema
    crop_prediction: Optional[CropPredictionSummarySchema]
    recommendations: List[RecommendationItemSchema]
    evidence_validation: EvidenceValidationSummarySchema
    provenance_sources: List[str]
    disclaimers_and_limitations: List[str]


# ------------------------------------------------------------------------------
# 5. Complete Unified Intelligence Workflow Schema
# ------------------------------------------------------------------------------
class UnifiedIntelligenceRequest(BaseModel):
    # Soil chemistry parameters for SHI & Advisory
    pH: Optional[float] = Field(6.5, ge=0.0, le=14.0)
    EC: Optional[float] = Field(0.5, ge=0.0)
    OC: Optional[float] = Field(0.8, ge=0.0)
    N: float = Field(..., ge=0.0, description="Nitrogen (kg/ha)")
    P: float = Field(..., ge=0.0, description="Phosphorus (kg/ha)")
    K: float = Field(..., ge=0.0, description="Potassium (kg/ha)")
    S: Optional[float] = Field(15.0, ge=0.0)
    Zn: Optional[float] = Field(1.0, ge=0.0)
    B: Optional[float] = Field(0.6, ge=0.0)
    Fe: Optional[float] = Field(6.0, ge=0.0)
    Mn: Optional[float] = Field(3.0, ge=0.0)
    Cu: Optional[float] = Field(0.5, ge=0.0)
    
    # Environmental & weather parameters for ML Crop Suitability
    temperature: float = Field(..., ge=-20.0, le=60.0, description="Temperature (°C)")
    humidity: float = Field(..., ge=0.0, le=100.0, description="Humidity (%)")
    rainfall: float = Field(..., ge=0.0, description="Rainfall (mm)")


class StageStatus(BaseModel):
    stage_id: int
    name: str
    status: str
    description: str


class UnifiedIntelligenceResponse(BaseModel):
    timestamp: str
    pipeline_stages: List[StageStatus]
    soil_health: SoilAnalysisResponse
    crop_prediction: CropPredictionResponse
    advisory: AdvisoryResponse
    provenance: Dict[str, Any]


# ------------------------------------------------------------------------------
# 6. Software Sensor Simulation Schemas (Phase 6)
# ------------------------------------------------------------------------------
class SensorFrameSchema(BaseModel):
    timestamp: str
    index: int
    Temp_5cm: Optional[float] = None
    SM_5cm: Optional[float] = None
    EC_5cm: Optional[float] = None
    Temp_50cm: Optional[float] = None
    SM_50cm: Optional[float] = None
    EC_50cm: Optional[float] = None
    Precipitation: Optional[float] = None


class ReliabilityDetailSchema(BaseModel):
    composite_reliability: float
    s_range: float
    s_dist: float
    s_avail: float
    s_step: float
    status_label: str


class SensorStreamResponse(BaseModel):
    frames: List[SensorFrameSchema]
    reliability: ReliabilityDetailSchema
    total_records: int
    current_index: int
    year: int
    simulation_notice: str = "Software Sensor Stream Simulation (Dataset B Berambadi Replay) — No Physical Hardware Connected"


# ------------------------------------------------------------------------------
# 7. Knowledge Corpus & Benchmark Sample Schemas
# ------------------------------------------------------------------------------
class KnowledgeChunkItem(BaseModel):
    chunk_id: str
    document_title: str
    source_authority: str
    publication: str
    section: str
    text_preview: str


class KnowledgeCorpusResponse(BaseModel):
    total_chunks: int
    authorities: List[str]
    documents: List[Dict[str, Any]]
    chunks: List[KnowledgeChunkItem]


class SampleProfile(BaseModel):
    id: str
    title: str
    location: str
    description: str
    soil_data: Dict[str, float]
    crop_features: Dict[str, float]
    highlight: str
