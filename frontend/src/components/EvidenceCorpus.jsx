import React, { useEffect, useState } from "react";
import {
  BookOpen,
  Search,
  Filter,
} from "lucide-react";
import api from "../api/client";

export default function EvidenceCorpus() {
  const [corpusData, setCorpusData] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedAuthority, setSelectedAuthority] = useState("ALL");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadCorpus() {
      try {
        const res = await api.getKnowledgeCorpus();
        setCorpusData(res);
      } catch (err) {
        console.error("Failed to load knowledge corpus:", err);
      } finally {
        setLoading(false);
      }
    }
    loadCorpus();
  }, []);

  if (loading) {
    return (
      <div className="academic-card" style={{ textAlign: "center", padding: "48px" }}>
        <div className="spinner" style={{ margin: "0 auto 12px auto" }} />
        <div style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
          Loading Curated Agricultural Knowledge Base...
        </div>
      </div>
    );
  }

  const { total_chunks = 48, authorities = ["FAO", "ICAR", "USDA"], documents = [], chunks = [] } = corpusData || {};

  const filteredChunks = chunks.filter((ch) => {
    const matchesSearch =
      ch.document_title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      ch.section.toLowerCase().includes(searchTerm.toLowerCase()) ||
      ch.text_preview.toLowerCase().includes(searchTerm.toLowerCase()) ||
      ch.chunk_id.toLowerCase().includes(searchTerm.toLowerCase());

    const matchesAuth =
      selectedAuthority === "ALL" || ch.source_authority === selectedAuthority;

    return matchesSearch && matchesAuth;
  });

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title">
          <BookOpen size={22} color="var(--color-slate)" />
          Curated Scientific Knowledge Base &amp; Literature Corpus
        </h2>
        <p className="page-desc">
          Authoritative agricultural literature indexed in the dense FAISS vector store (MiniLM 384-dimensional embeddings) powering semantic RAG retrieval for agronomic advisory grounding.
        </p>
      </div>

      {/* Corpus Summary Cards */}
      <div className="stat-card-grid" style={{ marginBottom: "20px" }}>
        <div className="stat-card">
          <div className="stat-label">Indexed Passages</div>
          <div className="stat-value">
            {total_chunks} <span style={{ fontSize: "0.9rem", color: "var(--text-muted)", fontWeight: "normal" }}>Chunks</span>
          </div>
          <div className="stat-meta">
            MiniLM 384-dim Dense Vectors
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Authority Bodies</div>
          <div className="stat-value">
            {authorities.length} <span style={{ fontSize: "0.9rem", color: "var(--text-muted)", fontWeight: "normal" }}>Organizations</span>
          </div>
          <div className="stat-meta">
            FAO &bull; ICAR &bull; USDA
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Source Publications</div>
          <div className="stat-value">
            {documents.length || 6} <span style={{ fontSize: "0.9rem", color: "var(--text-muted)", fontWeight: "normal" }}>Bulletins</span>
          </div>
          <div className="stat-meta">
            Handbooks, Schemes &amp; Bulletins
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Grounding Threshold</div>
          <div className="stat-value" style={{ color: "var(--color-green-dark)" }}>
            &ge; 0.35
          </div>
          <div className="stat-meta">
            Cosine Similarity Minimum
          </div>
        </div>
      </div>

      {/* Search & Filter Bar */}
      <div className="academic-card" style={{ padding: "14px 18px", marginBottom: "16px" }}>
        <div style={{ display: "flex", gap: "12px", alignItems: "center", flexWrap: "wrap" }}>
          <div style={{ flex: 1, minWidth: "240px", position: "relative" }}>
            <Search
              size={15}
              color="var(--text-muted)"
              style={{ position: "absolute", left: "10px", top: "50%", transform: "translateY(-50%)" }}
            />
            <input
              type="text"
              className="form-input"
              style={{ width: "100%", paddingLeft: "32px" }}
              placeholder="Search literature by topic, nutrient, pH, or chunk ID..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <Filter size={14} color="var(--text-muted)" />
            <select
              className="form-input"
              value={selectedAuthority}
              onChange={(e) => setSelectedAuthority(e.target.value)}
              style={{ width: "auto" }}
            >
              <option value="ALL">All Authorities ({authorities.length})</option>
              {authorities.map((auth) => (
                <option key={auth} value={auth}>
                  {auth} Only
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Literature Chunks Table/List */}
      <div className="academic-card">
        <div className="card-header-clean">
          <div>
            <h3 className="card-title-clean">Indexed Literature Passages</h3>
            <div className="card-subtitle-clean">
              Showing {filteredChunks.length} of {chunks.length} passages
            </div>
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
          {filteredChunks.map((chunk) => (
            <div
              key={chunk.chunk_id}
              style={{
                border: "1px solid var(--border-color)",
                borderRadius: "var(--radius-sm)",
                padding: "12px 14px",
                backgroundColor: "#ffffff",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", flexWrap: "gap", gap: "6px" }}>
                <div style={{ fontWeight: 600, fontSize: "0.85rem", color: "var(--text-primary)" }}>
                  <span
                    className={`status-pill ${
                      chunk.source_authority === "ICAR"
                        ? "status-pill-green"
                        : chunk.source_authority === "FAO"
                        ? "status-pill-blue"
                        : "status-pill-gray"
                    }`}
                    style={{ marginRight: "6px" }}
                  >
                    {chunk.source_authority}
                  </span>
                  {chunk.document_title}
                </div>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.72rem", color: "var(--text-dim)" }}>
                  ID: {chunk.chunk_id}
                </span>
              </div>

              <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: "6px", lineHeight: "1.5" }}>
                {chunk.text_preview}
              </div>

              <div style={{ display: "flex", justifyContent: "space-between", marginTop: "8px", fontSize: "0.72rem", color: "var(--text-muted)" }}>
                <span><strong>Section:</strong> {chunk.section}</span>
                <span><strong>Citation:</strong> {chunk.citation}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
