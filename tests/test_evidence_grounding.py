"""
Unit and Integration Tests for RAG Evidence Grounding and Validation Layer.
"""

import pytest
from advisory.evidence import EvidenceGroundingValidator, get_evidence_validator


@pytest.fixture
def validator():
    return get_evidence_validator()


def test_evidence_retrieval_and_validation_valid_topic(validator):
    topic = "Soil Organic Carbon"
    query = "Soil organic carbon restoration and farmyard manure application."
    action = "Incorporate 5-10 t/ha farmyard manure and green manure."

    evidence, status = validator.validate_and_retrieve_evidence(topic, query, action, top_k=2)

    assert status == "VALIDATED"
    assert len(evidence) == 2
    assert evidence[0].relevance_score >= 0.35
    assert "FAO" in evidence[0].source_authority or "USDA" in evidence[0].source_authority or "ICAR" in evidence[0].source_authority
    assert evidence[0].chunk_id != ""
    assert len(evidence[0].snippet) > 20


def test_evidence_validation_unsupported_out_of_domain(validator):
    topic = "Quantum Cryptography"
    query = "Quantum key distribution protocols and photonic lattice cryptography."
    action = "Install superconducting qubit detectors."

    evidence, status = validator.validate_and_retrieve_evidence(topic, query, action, top_k=2)

    # Must be rejected as unsupported or limited due to low similarity
    if len(evidence) > 0:
        assert evidence[0].relevance_score < 0.35
    assert status in ["UNSUPPORTED", "LIMITED_EVIDENCE"]


def test_evidence_citation_formatting(validator):
    topic = "Soil Salinity & EC"
    query = "Soil salinity management, electrical conductivity, and leaching requirement."
    action = "Apply leaching irrigation to flush soluble salts below root zone."

    evidence, status = validator.validate_and_retrieve_evidence(topic, query, action, top_k=1)

    assert len(evidence) == 1
    top_ev = evidence[0]
    assert "USDA" in top_ev.citation or "FAO" in top_ev.citation
    assert top_ev.document_title != ""
    assert top_ev.section != ""
