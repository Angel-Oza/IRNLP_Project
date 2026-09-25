"""
Evidence Validation and Grounding Engine.

Reuses the existing Agricultural RAG Retrieval Engine (rag/retriever.py & FAISS vector store)
to validate every proposed agronomic recommendation against authoritative scientific literature
(FAO Soils Bulletins, USDA Handbooks, ICAR Guidelines).
"""

import os
import sys
from typing import Dict, List, Any, Optional, Tuple

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from rag.retriever import SoilKnowledgeRetriever, get_soil_retriever
from advisory.schemas import EvidenceReference


class EvidenceGroundingValidator:
    """
    Validates proposed agricultural recommendations against indexed scientific literature.
    """

    RELEVANCE_THRESHOLD_VALIDATED = 0.35
    RELEVANCE_THRESHOLD_LIMITED = 0.25

    def __init__(self, retriever: Optional[SoilKnowledgeRetriever] = None):
        self.retriever = retriever or get_soil_retriever()

    def validate_and_retrieve_evidence(
        self,
        topic: str,
        issue_query: str,
        action_statement: str,
        top_k: int = 2,
    ) -> Tuple[List[EvidenceReference], str]:
        """
        Retrieves supporting scientific evidence from existing FAISS index and determines validation status.
        
        Args:
            topic: Agronomic topic category (e.g. "Nitrogen Management", "Organic Carbon Restoration").
            issue_query: Natural language query describing the soil issue / deficiency.
            action_statement: The proposed remediation action.
            top_k: Number of supporting evidence passages to retrieve.
            
        Returns:
            Tuple of (List[EvidenceReference], validation_status).
            validation_status is one of: "VALIDATED", "LIMITED_EVIDENCE", "UNSUPPORTED".
        """
        combined_query = f"{topic}: {issue_query} Remediation: {action_statement}"
        rag_res = self.retriever.retrieve_by_text(combined_query, top_k=top_k, min_score=self.RELEVANCE_THRESHOLD_LIMITED)

        evidence_list = []
        top_score = 0.0

        for r in rag_res["results"]:
            score = float(r["similarity_score"])
            if score > top_score:
                top_score = score

            evidence_list.append(
                EvidenceReference(
                    chunk_id=r["chunk_id"],
                    source_authority=r["source_authority"],
                    document_title=r["document_title"],
                    section=r["section"],
                    citation=r["citation"],
                    relevance_score=round(score, 4),
                    snippet=r["text"][:300].strip() + ("..." if len(r["text"]) > 300 else ""),
                )
            )

        # Determine validation status
        if top_score >= self.RELEVANCE_THRESHOLD_VALIDATED and len(evidence_list) > 0:
            status = "VALIDATED"
        elif top_score >= self.RELEVANCE_THRESHOLD_LIMITED and len(evidence_list) > 0:
            status = "LIMITED_EVIDENCE"
        else:
            status = "UNSUPPORTED"

        return evidence_list, status


def get_evidence_validator() -> EvidenceGroundingValidator:
    """Helper factory for EvidenceGroundingValidator."""
    return EvidenceGroundingValidator()


if __name__ == "__main__":
    validator = get_evidence_validator()
    topic = "Soil Organic Carbon"
    query = "Critically low organic carbon (0.24%) and soil biological degradation."
    action = "Incorporate 5-10 tonnes/ha of farmyard manure or vermicompost and cultivate leguminous green manure."

    evidence, status = validator.validate_and_retrieve_evidence(topic, query, action)
    print(f"Validation Status: {status}")
    for ev in evidence:
        print(f"  Citation: {ev.citation} (Score: {ev.relevance_score})")
