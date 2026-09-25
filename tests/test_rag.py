"""
Unit and Integration Tests for Agricultural RAG Retrieval Engine.
"""

import os
import pytest
import numpy as np
from rag.retriever import SoilKnowledgeRetriever, get_soil_retriever


@pytest.fixture
def retriever():
    return get_soil_retriever()


def test_retriever_initialization(retriever):
    assert retriever.index is not None
    assert retriever.index.ntotal >= 40
    assert len(retriever.metadata) == retriever.index.ntotal
    assert retriever.model is not None


def test_chunk_metadata_schema(retriever):
    for chunk in retriever.metadata:
        assert "chunk_id" in chunk
        assert "document_title" in chunk
        assert "source_authority" in chunk
        assert "section" in chunk
        assert "text" in chunk
        assert len(chunk["text"]) > 20


def test_text_query_retrieval(retriever):
    query = "How does soil pH affect phosphorus availability and what liming material is used?"
    res = retriever.retrieve_by_text(query, top_k=3)

    assert res["total_retrieved"] == 3
    assert res["sufficient_evidence"] is True
    assert res["evidence_status"] == "SUFFICIENT_GROUNDED_EVIDENCE"

    top = res["results"][0]
    assert top["similarity_score"] > 0.50
    assert "citation" in top
    assert "FAO" in top["source_authority"] or "ICAR" in top["source_authority"]


def test_context_conditioned_retrieval(retriever):
    soil_state = {
        "ph": 5.2,
        "N": 35.0,
        "P": 8.0,
        "K": 120.0,
        "moisture": 14.5,
        "EC": 0.25,
    }
    res = retriever.retrieve_by_soil_state(soil_state, target_crop="Maize", top_k=2)

    assert len(res["results"]) == 2
    assert "input_soil_state" in res
    assert res["sufficient_evidence"] is True


def test_dense_embedding_for_fusion(retriever):
    soil_state = {
        "ph": 6.5,
        "N": 90.0,
        "P": 45.0,
        "K": 50.0,
        "moisture": 32.0,
        "EC": 0.30,
    }
    emb = retriever.get_text_embedding_for_soil_state(soil_state, target_crop="Rice")

    assert isinstance(emb, np.ndarray)
    assert emb.shape == (384,)
    # Verify vector is L2 normalized
    np.testing.assert_allclose(np.linalg.norm(emb), 1.0, atol=1e-3)


def test_out_of_domain_query_rejection(retriever):
    out_of_domain = "What is the capital city of France?"
    res = retriever.retrieve_by_text(out_of_domain, top_k=3, min_score=0.40)

    # Score should be low and marked as insufficient
    top_score = res["results"][0]["similarity_score"] if res["results"] else 0.0
    assert top_score < 0.40
    assert res["sufficient_evidence"] is False
    assert res["evidence_status"] == "INSUFFICIENT_EVIDENCE_IN_KB"
