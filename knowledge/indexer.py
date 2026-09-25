"""
Knowledge Base Indexer Module for Explainable Multi-Source Soil Health Intelligence System.

Processes curated agricultural scientific documents (FAO, USDA, ICAR),
performs semantic section-aware chunking, generates dense embeddings via
Sentence-Transformers (all-MiniLM-L6-v2), and builds a persistent FAISS index.
"""

import os
import glob
import json
import re
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class AgriculturalKnowledgeIndexer:
    """
    Indexes curated scientific agronomic literature into a FAISS dense vector store.
    """

    DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    def __init__(
        self,
        corpus_dir: Optional[str] = None,
        index_dir: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.corpus_dir = corpus_dir or os.path.join(base_dir, "knowledge", "corpus")
        self.index_dir = index_dir or os.path.join(base_dir, "knowledge", "faiss_index")
        os.makedirs(self.index_dir, exist_ok=True)

        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        print(f"Loading SentenceTransformer model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        self.chunks_metadata: List[Dict[str, Any]] = []
        self.faiss_index: Optional[faiss.IndexFlatIP] = None

    def build_and_save_index(self) -> Dict[str, Any]:
        """
        Ingests all documents from corpus_dir, chunks them, computes embeddings,
        builds FAISS index, and saves all artifacts.
        """
        print(f"Ingesting scientific documents from: {self.corpus_dir}")
        doc_files = sorted(glob.glob(os.path.join(self.corpus_dir, "*.md")))
        if not doc_files:
            raise FileNotFoundError(f"No markdown documents found in corpus directory: {self.corpus_dir}")

        all_chunks = []
        chunk_id_counter = 1

        for doc_path in doc_files:
            filename = os.path.basename(doc_path)
            with open(doc_path, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = self._chunk_document(content, filename, start_id=chunk_id_counter)
            all_chunks.extend(chunks)
            chunk_id_counter += len(chunks)

        print(f"Extracted {len(all_chunks)} semantic chunks across {len(doc_files)} scientific documents.")

        # Compute dense embeddings
        texts_to_embed = [c["text"] for c in all_chunks]
        print(f"Computing {self.EMBEDDING_DIM}-dimensional dense embeddings for {len(texts_to_embed)} chunks...")
        embeddings = self.model.encode(texts_to_embed, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Build FAISS Index (Inner Product on normalized vectors = Cosine Similarity)
        self.faiss_index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
        self.faiss_index.add(embeddings)
        print(f"FAISS index populated with {self.faiss_index.ntotal} vectors.")

        # Save artifacts
        self.chunks_metadata = all_chunks
        self._save_artifacts(embeddings)

        summary = {
            "num_documents": len(doc_files),
            "num_chunks": len(all_chunks),
            "embedding_dimension": self.EMBEDDING_DIM,
            "embedding_model": self.model_name,
            "index_path": os.path.join(self.index_dir, "index.faiss"),
            "metadata_path": os.path.join(self.index_dir, "chunks_metadata.json"),
        }
        return summary

    def _chunk_document(self, content: str, filename: str, start_id: int = 1) -> List[Dict[str, Any]]:
        """
        Splits markdown document into clean semantic chunks preserving section metadata.
        """
        # Extract title and source metadata from header
        lines = content.strip().split("\n")
        title = lines[0].replace("#", "").strip() if lines else filename
        authority = "FAO / ICAR / USDA"
        publication = "Agronomic Scientific Monograph"

        for line in lines[:10]:
            if line.startswith("**Source Authority:**"):
                authority = line.replace("**Source Authority:**", "").strip()
            elif line.startswith("**Publication:**"):
                publication = line.replace("**Publication:**", "").strip()

        # Split document by markdown headings (## or ###)
        sections = re.split(r"\n(?=##+\s+)", content)
        chunks = []
        c_id = start_id

        for sec in sections:
            sec_clean = sec.strip()
            if not sec_clean:
                continue

            # Extract section header
            sec_header_match = re.match(r"^##+\s+(.+)$", sec_clean, re.MULTILINE)
            sec_header = sec_header_match.group(1).strip() if sec_header_match else "General Agronomy"

            # If section is excessively long (>2000 chars), split into sub-paragraphs
            if len(sec_clean) > 1800:
                paragraphs = sec_clean.split("\n\n")
                buf = ""
                for p in paragraphs:
                    if len(buf) + len(p) < 1200:
                        buf += p + "\n\n"
                    else:
                        if buf.strip():
                            chunks.append({
                                "chunk_id": f"CHUNK_{c_id:04d}",
                                "document_title": title,
                                "source_authority": authority,
                                "publication": publication,
                                "section": sec_header,
                                "filename": filename,
                                "text": buf.strip(),
                            })
                            c_id += 1
                        buf = p + "\n\n"
                if buf.strip():
                    chunks.append({
                        "chunk_id": f"CHUNK_{c_id:04d}",
                        "document_title": title,
                        "source_authority": authority,
                        "publication": publication,
                        "section": sec_header,
                        "filename": filename,
                        "text": buf.strip(),
                    })
                    c_id += 1
            else:
                chunks.append({
                    "chunk_id": f"CHUNK_{c_id:04d}",
                    "document_title": title,
                    "source_authority": authority,
                    "publication": publication,
                    "section": sec_header,
                    "filename": filename,
                    "text": sec_clean,
                })
                c_id += 1

        return chunks

    def _save_artifacts(self, embeddings: np.ndarray) -> None:
        """Saves FAISS index, metadata JSON, and raw embeddings numpy array."""
        faiss_path = os.path.join(self.index_dir, "index.faiss")
        faiss.write_index(self.faiss_index, faiss_path)

        meta_path = os.path.join(self.index_dir, "chunks_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks_metadata, f, indent=2)

        np_path = os.path.join(self.index_dir, "embeddings.npy")
        np.save(np_path, embeddings)

        print(f"Artifacts successfully saved to {self.index_dir}")


def build_knowledge_index() -> Dict[str, Any]:
    """Helper entry point to build knowledge index."""
    indexer = AgriculturalKnowledgeIndexer()
    return indexer.build_and_save_index()


if __name__ == "__main__":
    build_knowledge_index()
