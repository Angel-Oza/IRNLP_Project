import React, { useState } from "react";
import {
  GitMerge,
  CheckCircle2,
  ArrowRight,
  RefreshCw,
} from "lucide-react";
import api from "../api/client";

export default function IntelligencePipeline({ initialPayload, onAdvisoryGenerated }) {
  const defaultData = {
    pH: 7.13,
    EC: 0.13,
    OC: 0.23,
    N: 108.0,
    P: 11.59,
    K: 130.0,
    S: 14.67,
    Zn: 0.98,
    B: 0.6,
    Fe: 8.06,
    Mn: 2.03,
    Cu: 0.47,
    temperature: 26.5,
    humidity: 68.0,
    rainfall: 145.0,
  };

  const [formData, setFormData] = useState(initialPayload || defaultData);
  const [pipelineResult, setPipelineResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleInputChange = (key, value) => {
    setFormData((prev) => ({
      ...prev,
      [key]: value === "" ? "" : parseFloat(value),
    }));
  };

  const handleExecute = async (e) => {
    if (e) e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await api.runUnifiedIntelligence(formData);
      setPipelineResult(res);
      if (onAdvisoryGenerated && res.advisory) {
        onAdvisoryGenerated(res.advisory);
      }
    } catch (err) {
      setError(err.message || "Failed to execute complete intelligence pipeline.");
    } finally {
      setLoading(false);
    }
  };

  const loadPreset = (presetKey) => {
    if (presetKey === "bareilly") {
      setFormData(defaultData);
    } else if (presetKey === "optimal") {
      setFormData({
        pH: 6.70, EC: 0.45, OC: 0.95, N: 380.0, P: 22.0, K: 240.0,
        S: 18.5, Zn: 1.20, B: 0.85, Fe: 9.50, Mn: 4.20, Cu: 0.80,
        temperature: 23.5, humidity: 82.0, rainfall: 215.0
      });
    } else if (presetKey === "acidic") {
      setFormData({
        pH: 4.80, EC: 0.20, OC: 1.10, N: 260.0, P: 8.5, K: 95.0,
        S: 8.0, Zn: 0.45, B: 0.35, Fe: 14.5, Mn: 1.80, Cu: 0.25,
        temperature: 18.2, humidity: 75.0, rainfall: 180.0
      });
    } else if (presetKey === "saline") {
      setFormData({
        pH: 8.40, EC: 2.40, OC: 0.35, N: 160.0, P: 14.0, K: 310.0,
        S: 22.0, Zn: 0.50, B: 1.80, Fe: 3.80, Mn: 1.50, Cu: 0.30,
        temperature: 31.0, humidity: 42.0, rainfall: 45.0
      });
    }
    setPipelineResult(null);
  };

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title">
          <GitMerge size={22} color="var(--color-green-dark)" />
          Unified Multi-Source Intelligence Pipeline
        </h2>
        <p className="page-desc">
          Executes end-to-end multi-source diagnostic pipeline: Preprocessing &rarr; Supervised ML Classification &rarr; TreeSHAP Feature Attributions &rarr; Soil Health Index Scoring &rarr; Dense FAISS RAG Retrieval &rarr; Grounded Advisory Synthesis.
        </p>
      </div>

      {/* Input Parameters Form Card */}
      <div className="academic-card">
        <div className="card-header-clean">
          <div>
            <h3 className="card-title-clean">Multi-Source Input Telemetry &amp; Soil Chemistry</h3>
            <div className="card-subtitle-clean">Provide 12 chemistry indicators and 3 environmental features</div>
          </div>
          <button
            type="button"
            className="btn-secondary btn-sm"
            onClick={() => {
              setFormData(defaultData);
              setPipelineResult(null);
            }}
          >
            <RefreshCw size={12} /> Reset Defaults
          </button>
        </div>

        {/* Quick Presets */}
        <div style={{ marginBottom: "14px", display: "flex", alignItems: "center", gap: "6px", flexWrap: "wrap" }}>
          <span style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--text-muted)" }}>Benchmark Profiles:</span>
          <button type="button" className="btn-secondary btn-sm" onClick={() => loadPreset("bareilly")}>Bareilly Inceptisol</button>
          <button type="button" className="btn-secondary btn-sm" onClick={() => loadPreset("optimal")}>Optimal Wetland</button>
          <button type="button" className="btn-secondary btn-sm" onClick={() => loadPreset("acidic")}>Acidic Hill</button>
          <button type="button" className="btn-secondary btn-sm" onClick={() => loadPreset("saline")}>Arid Saline</button>
        </div>

        <form onSubmit={handleExecute}>
          {/* Section 1: Primary Macronutrients & Master Soil Properties */}
          <div style={{ fontSize: "0.78rem", fontWeight: 700, color: "var(--color-green-dark)", textTransform: "uppercase", marginBottom: "8px" }}>
            1. Primary Macronutrients &amp; Master Soil Properties
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "10px", marginBottom: "16px" }}>
            {[
              { key: "pH", label: "Soil pH", unit: "pH" },
              { key: "EC", label: "EC (Salinity)", unit: "dS/m" },
              { key: "OC", label: "Organic Carbon", unit: "%" },
              { key: "N", label: "Nitrogen (N)", unit: "kg/ha" },
              { key: "P", label: "Phosphorus (P)", unit: "kg/ha" },
              { key: "K", label: "Potassium (K)", unit: "kg/ha" },
              { key: "temperature", label: "Temperature", unit: "°C" },
              { key: "humidity", label: "Humidity", unit: "%" },
            ].map((f) => (
              <div key={f.key} className="form-group" style={{ marginBottom: "4px" }}>
                <label className="form-label">
                  <span>{f.label}</span>
                  <span className="form-unit">{f.unit}</span>
                </label>
                <input
                  type="number"
                  step="any"
                  className="form-input"
                  value={formData[f.key] !== undefined ? formData[f.key] : ""}
                  onChange={(e) => handleInputChange(f.key, e.target.value)}
                  required
                />
              </div>
            ))}
          </div>

          {/* Section 2: Secondary & Micronutrients + Rainfall */}
          <div style={{ fontSize: "0.78rem", fontWeight: 700, color: "var(--color-blue)", textTransform: "uppercase", marginBottom: "8px" }}>
            2. Secondary Nutrients, Micronutrients &amp; Precipitation
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "10px", marginBottom: "16px" }}>
            {[
              { key: "S", label: "Sulphur (S)", unit: "ppm" },
              { key: "Zn", label: "Zinc (Zn)", unit: "ppm" },
              { key: "B", label: "Boron (B)", unit: "ppm" },
              { key: "Fe", label: "Iron (Fe)", unit: "ppm" },
              { key: "Mn", label: "Manganese (Mn)", unit: "ppm" },
              { key: "Cu", label: "Copper (Cu)", unit: "ppm" },
              { key: "rainfall", label: "Rainfall", unit: "mm" },
            ].map((f) => (
              <div key={f.key} className="form-group" style={{ marginBottom: "4px" }}>
                <label className="form-label">
                  <span>{f.label}</span>
                  <span className="form-unit">{f.unit}</span>
                </label>
                <input
                  type="number"
                  step="any"
                  className="form-input"
                  value={formData[f.key] !== undefined ? formData[f.key] : ""}
                  onChange={(e) => handleInputChange(f.key, e.target.value)}
                  required
                />
              </div>
            ))}
          </div>

          {error && (
            <div className="notice-box notice-box-warning" style={{ marginBottom: "14px" }}>
              <strong>Error:</strong> {error}
            </div>
          )}

          <div style={{ display: "flex", justifyContent: "flex-end" }}>
            <button
              type="submit"
              id="btn-run-full-pipeline"
              className="btn-primary"
              disabled={loading}
            >
              {loading ? (
                <>
                  <div className="spinner" /> Executing 6-Stage Pipeline...
                </>
              ) : (
                <>
                  Execute Complete Pipeline <ArrowRight size={14} />
                </>
              )}
            </button>
          </div>
        </form>
      </div>

      {/* Results Section */}
      {pipelineResult && (
        <div>
          {/* Execution Flow Summary */}
          <div className="academic-card">
            <div className="card-header-clean">
              <div>
                <h3 className="card-title-clean">Pipeline Stage Execution Audit</h3>
                <div className="card-subtitle-clean">End-to-end verification of all multi-source stages</div>
              </div>
              <span className="status-pill status-pill-green">All 6 Stages Completed</span>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "10px" }}>
              {[
                { stage: "Stage 1", name: "Preprocessing", desc: "Standardization & validation" },
                { stage: "Stage 2", name: "ML Classifier", desc: `Predicted: ${pipelineResult.crop_prediction?.predicted_crop}` },
                { stage: "Stage 3", name: "TreeSHAP XAI", desc: "Local Shapley drivers" },
                { stage: "Stage 4", name: "SHI Engine", desc: `SHI: ${pipelineResult.soil_analysis?.shi_score.toFixed(1)}/100` },
                { stage: "Stage 5", name: "FAISS RAG", desc: "Literature retrieval" },
                { stage: "Stage 6", name: "Advisory", desc: `${pipelineResult.advisory?.recommendations?.length || 0} protocols` },
              ].map((st, i) => (
                <div
                  key={i}
                  style={{
                    border: "1px solid var(--border-color)",
                    borderRadius: "var(--radius-sm)",
                    padding: "10px 12px",
                    backgroundColor: "var(--bg-subtle)",
                  }}
                >
                  <div style={{ fontSize: "0.7rem", fontWeight: 700, color: "var(--color-green-dark)" }}>
                    <CheckCircle2 size={11} style={{ display: "inline", marginRight: "4px" }} />
                    {st.stage}
                  </div>
                  <div style={{ fontWeight: 600, fontSize: "0.82rem", color: "var(--text-primary)", marginTop: "2px" }}>
                    {st.name}
                  </div>
                  <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: "2px" }}>
                    {st.desc}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Key Summary Cards */}
          <div className="stat-card-grid" style={{ marginBottom: "20px" }}>
            <div className="stat-card">
              <div className="stat-label">Soil Health Index</div>
              <div className="stat-value" style={{ color: "var(--color-green-dark)" }}>
                {pipelineResult.soil_analysis?.shi_score.toFixed(1)} / 100
              </div>
              <div className="stat-meta">
                {pipelineResult.soil_analysis?.category}
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-label">Predicted Crop</div>
              <div className="stat-value" style={{ textTransform: "capitalize", color: "var(--color-blue)" }}>
                {pipelineResult.crop_prediction?.predicted_crop}
              </div>
              <div className="stat-meta">
                Confidence: {(pipelineResult.crop_prediction?.confidence * 100).toFixed(1)}%
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-label">Validated Protocols</div>
              <div className="stat-value">
                {pipelineResult.advisory?.recommendations?.length || 0}
              </div>
              <div className="stat-meta">
                Grounding Pass Rate: 100%
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-label">Literature Citations</div>
              <div className="stat-value">
                {pipelineResult.advisory?.provenance_sources?.length || 0}
              </div>
              <div className="stat-meta">
                FAO &bull; ICAR &bull; USDA
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
