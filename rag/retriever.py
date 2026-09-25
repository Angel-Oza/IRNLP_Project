"""
Soil Context-Aware RAG Retriever Module.

Retrieves authoritative agronomic scientific literature (FAO, USDA, ICAR)
based on natural language queries or current numerical soil/sensor states.
Extracts dense text embeddings for Gated Fusion and formats verifiable source citations.
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class SoilKnowledgeRetriever:
    """
    Dense Passage Retriever over Curated Agricultural Science FAISS Vector Store.
    """

    DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    RELEVANCE_THRESHOLD = 0.30

    def __init__(
        self,
        index_dir: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.index_dir = index_dir or os.path.join(base_dir, "knowledge", "faiss_index")

        faiss_path = os.path.join(self.index_dir, "index.faiss")
        meta_path = os.path.join(self.index_dir, "chunks_metadata.json")

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            raise FileNotFoundError(
                f"FAISS index or metadata missing in {self.index_dir}. "
                f"Please run `python -m knowledge.indexer` first."
            )

        print(f"Loading FAISS index from: {faiss_path}")
        self.index = faiss.read_index(faiss_path)

        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata: List[Dict[str, Any]] = json.load(f)

        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        print(f"Loading embedding model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)

    def retrieve_by_text(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = RELEVANCE_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Retrieves top-k passages matching natural language query text.
        """
        query_clean = query.strip()
        if not query_clean:
            return {"query": query, "total_retrieved": 0, "results": [], "sufficient_evidence": False}

        # Embed query (normalized for cosine similarity)
        q_emb = self.model.encode([query_clean], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype=np.float32)

        # Search FAISS index
        scores, indices = self.index.search(q_emb, k=top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            sim_score = float(score)

            citation = (
                f"[{meta['source_authority']} - {meta['document_title']}, "
                f"Sec: '{meta['section']}', Ref: {meta['chunk_id']}]"
            )

            results.append({
                "chunk_id": meta["chunk_id"],
                "similarity_score": round(sim_score, 4),
                "is_relevant": sim_score >= min_score,
                "document_title": meta["document_title"],
                "source_authority": meta["source_authority"],
                "publication": meta["publication"],
                "section": meta["section"],
                "citation": citation,
                "text": meta["text"],
            })

        has_sufficient = len(results) > 0 and results[0]["similarity_score"] >= min_score

        return {
            "query": query_clean,
            "top_k": top_k,
            "total_retrieved": len(results),
            "sufficient_evidence": has_sufficient,
            "evidence_status": "SUFFICIENT_GROUNDED_EVIDENCE" if has_sufficient else "INSUFFICIENT_EVIDENCE_IN_KB",
            "results": results,
        }

    def retrieve_by_soil_state(
        self,
        soil_features: Dict[str, float],
        target_crop: Optional[str] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Synthesizes a context-aware agronomic query from numerical soil parameters and retrieves evidence.
        """
        query = self.build_query_from_soil_state(soil_features, target_crop=target_crop)
        response = self.retrieve_by_text(query, top_k=top_k)
        response["input_soil_state"] = soil_features
        response["target_crop"] = target_crop
        return response

    def get_text_embedding_for_soil_state(
        self,
        soil_features: Dict[str, float],
        target_crop: Optional[str] = None,
    ) -> np.ndarray:
        """
        Returns the 384-dimensional dense representation z_text for the top retrieved passage.
        Used as the text modality input to the PyTorch Gated Fusion network.
        """
        res = self.retrieve_by_soil_state(soil_features, target_crop=target_crop, top_k=1)
        if res["results"]:
            top_text = res["results"][0]["text"]
        else:
            top_text = self.build_query_from_soil_state(soil_features, target_crop=target_crop)

        emb = self.model.encode([top_text], normalize_embeddings=True)
        return np.array(emb[0], dtype=np.float32)

    def build_query_from_soil_state(
        self,
        soil_dict: Dict[str, float],
        target_crop: Optional[str] = None,
    ) -> str:
        """
        Constructs an objective agronomic inquiry without label-revealing keywords.
        """
        parts = []
        if target_crop:
            parts.append(f"Agronomic requirements and management for {target_crop.lower()}.")

        ph = soil_dict.get("ph")
        if ph is not None:
            if ph < 5.8:
                parts.append(f"Acidic soil pH {ph:.2f} remediation, liming, and phosphorus availability.")
            elif ph > 7.6:
                parts.append(f"Alkaline soil pH {ph:.2f} management, gypsum, and micronutrient availability.")
            else:
                parts.append(f"Optimal neutral soil pH {ph:.2f} nutrient bioavailability.")

        n = soil_dict.get("N")
        if n is not None:
            if n < 50:
                parts.append(f"Low nitrogen {n:.1f} kg/ha deficiency symptoms and fertilizer application.")
            elif n > 100:
                parts.append(f"High nitrogen {n:.1f} kg/ha vegetative demand and split application.")

        p = soil_dict.get("P")
        if p is not None and p < 20:
            parts.append(f"Low phosphorus {p:.1f} kg/ha root development and phosphatic placement.")

        k = soil_dict.get("K")
        if k is not None and k > 100:
            parts.append(f"High potassium {k:.1f} kg/ha fruit development and moisture stress tolerance.")

        sm = soil_dict.get("moisture") or soil_dict.get("SM_5cm")
        if sm is not None:
            if sm < 18:
                parts.append(f"Drought moisture stress {sm:.1f}% volumetric content and irrigation scheduling.")
            elif sm > 38:
                parts.append(f"Excess waterlogging {sm:.1f}% drainage and hypoxia prevention.")

        ec = soil_dict.get("EC") or soil_dict.get("EC_5cm")
        if ec is not None and ec > 1.2:
            parts.append(f"Soil salinity electrical conductivity {ec:.2f} dS/m and leaching requirement.")

        if not parts:
            return "Sustainable soil fertility management and balanced nutrient stewardship."

        return " ".join(parts)


def get_soil_retriever(index_dir: Optional[str] = None) -> SoilKnowledgeRetriever:
    """Helper factory for SoilKnowledgeRetriever."""
    return SoilKnowledgeRetriever(index_dir=index_dir)


if __name__ == "__main__":
    retriever = get_soil_retriever()
    test_q = "How to treat phosphorus fixation in acidic soils?"
    print(f"\n--- Testing Query: '{test_q}' ---")
    out = retriever.retrieve_by_text(test_q, top_k=2)
    print(f"Evidence Status: {out['evidence_status']}")
    for r in out["results"]:
        print(f"\nScore: {r['similarity_score']} | Citation: {r['citation']}")
        print(f"Snippet: {r['text'][:200]}...")
