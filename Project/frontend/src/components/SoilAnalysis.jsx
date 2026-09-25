import React, { useState } from "react";
import {
  Sprout,
  RefreshCw,
  Scale,
  ArrowRight,
} from "lucide-react";
import api from "../api/client";

export default function SoilAnalysis({ initialSoilData }) {
  const defaultParams = {
    pH: 7.13,
    EC: 0.13,
    OC: 0.23,
    N: 108.0,
    P: 11.59,
    K: 130.0,
    S: 14.67,
    Zn: 0.98,
    B: 0.60,
    Fe: 8.06,
    Mn: 2.03,
    Cu: 0.47,
  };

  const [formData, setFormData] = useState(initialSoilData || defaultParams);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const parameterMeta = [
    { key: "pH", name: "Soil Reaction (pH)", unit: "pH", range: "6.0 - 7.5", prov: "FAO Bull. 64" },
    { key: "EC", name: "Electrical Conductivity", unit: "dS/m", range: "< 0.80 dS/m", prov: "USDA Hdbk 60" },
    { key: "OC", name: "Organic Carbon", unit: "%", range: ">= 0.75%", prov: "FAO Bull. 76" },
    { key: "N", name: "Available Nitrogen", unit: "kg/ha", range: "280 - 560", prov: "ICAR Card" },
    { key: "P", name: "Available Phosphorus", unit: "kg/ha", range: "10 - 25", prov: "ICAR Card" },
    { key: "K", name: "Available Potassium", unit: "kg/ha", range: "110 - 280", prov: "ICAR Card" },
    { key: "S", name: "Available Sulphur", unit: "ppm", range: ">= 10.0", prov: "ICAR AICRP" },
    { key: "Zn", name: "Available Zinc", unit: "ppm", range: ">= 0.60", prov: "ICAR AICRP" },
    { key: "B", name: "Available Boron", unit: "ppm", range: ">= 0.50", prov: "ICAR AICRP" },
    { key: "Fe", name: "Available Iron", unit: "ppm", range: ">= 4.50", prov: "ICAR AICRP" },
    { key: "Mn", name: "Available Manganese", unit: "ppm", range: ">= 2.00", prov: "ICAR AICRP" },
    { key: "Cu", name: "Available Copper", unit: "ppm", range: ">= 0.20", prov: "ICAR AICRP" },
  ];

  const handleInputChange = (key, value) => {
    setFormData((prev) => ({
      ...prev,
      [key]: value === "" ? null : parseFloat(value),
    }));
  };

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await api.analyzeSoil(formData);
      setResult(res);
    } catch (err) {
      setError(err.message || "Failed to calculate Soil Health Index.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFormData(defaultParams);
    setResult(null);
    setError(null);
  };

  const loadPreset = (presetName) => {
    if (presetName === "bareilly") {
      setFormData(defaultParams);
    } else if (presetName === "optimal") {
      setFormData({
        pH: 6.70, EC: 0.45, OC: 0.95, N: 380.0, P: 22.0, K: 240.0,
        S: 18.5, Zn: 1.20, B: 0.85, Fe: 9.50, Mn: 4.20, Cu: 0.80
      });
    } else if (presetName === "acidic") {
      setFormData({
        pH: 4.80, EC: 0.20, OC: 1.10, N: 260.0, P: 8.5, K: 95.0,
        S: 8.0, Zn: 0.45, B: 0.35, Fe: 14.5, Mn: 1.80, Cu: 0.25
      });
    } else if (presetName === "saline") {
      setFormData({
        pH: 8.40, EC: 2.40, OC: 0.35, N: 160.0, P: 14.0, K: 310.0,
        S: 22.0, Zn: 0.50, B: 1.80, Fe: 3.80, Mn: 1.50, Cu: 0.30
      });
    }
    setResult(null);
  };

  const getShiBarColor = (score) => {
    if (score >= 80) return "var(--color-green)";
    if (score >= 60) return "#0284c7";
    if (score >= 40) return "var(--color-amber)";
    return "var(--color-red)";
  };

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title">
          <Sprout size={22} color="var(--color-green-dark)" />
          Soil Analysis &amp; Health Index (SHI)
        </h2>
        <p className="page-desc">
          Rigorous agronomic evaluation of 12 chemical &amp; physical soil indicators against ICAR Soil Health Card norms, FAO Soils Bulletins 64/76, and USDA Handbook 60.
        </p>
      </div>

      <div className="grid-2" style={{ alignItems: "flex-start" }}>
        {/* LEFT COLUMN: Parameter Input Form */}
        <div className="academic-card">
          <div className="card-header-clean">
            <div>
              <h3 className="card-title-clean">Soil Chemistry Measurements</h3>
              <div className="card-subtitle-clean">Enter 12 laboratory measurements or load standard benchmarks</div>
            </div>
            <button
              type="button"
              className="btn-secondary btn-sm"
              onClick={handleReset}
            >
              <RefreshCw size={12} /> Reset
            </button>
          </div>

          {/* Quick Presets */}
          <div style={{ marginBottom: "16px", display: "flex", alignItems: "center", gap: "6px", flexWrap: "wrap" }}>
            <span style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--text-muted)" }}>Benchmark Presets:</span>
            <button type="button" className="btn-secondary btn-sm" onClick={() => loadPreset("bareilly")}>Bareilly Inceptisol</button>
            <button type="button" className="btn-secondary btn-sm" onClick={() => loadPreset("optimal")}>Optimal Wetland</button>
            <button type="button" className="btn-secondary btn-sm" onClick={() => loadPreset("acidic")}>Acidic Hill</button>
            <button type="button" className="btn-secondary btn-sm" onClick={() => loadPreset("saline")}>Arid Saline</button>
          </div>

          <form onSubmit={handleSubmit}>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "10px" }}>
              {parameterMeta.map((p) => (
                <div key={p.key} className="form-group" style={{ marginBottom: "6px" }}>
                  <label className="form-label" htmlFor={`input-${p.key}`}>
                    <span>{p.name}</span>
                    <span className="form-unit">{p.unit}</span>
                  </label>
                  <input
                    id={`input-${p.key}`}
                    type="number"
                    step="any"
                    className="form-input"
                    value={formData[p.key] !== null && formData[p.key] !== undefined ? formData[p.key] : ""}
                    onChange={(e) => handleInputChange(p.key, e.target.value)}
                    placeholder={`Opt: ${p.range}`}
                  />
                  <div style={{ fontSize: "0.68rem", color: "var(--text-dim)", display: "flex", justifyContent: "space-between" }}>
                    <span>Range: {p.range}</span>
                    <span>{p.prov}</span>
                  </div>
                </div>
              ))}
            </div>

            {error && (
              <div className="notice-box notice-box-warning" style={{ marginTop: "14px" }}>
                <strong>Error:</strong> {error}
              </div>
            )}

            <div style={{ marginTop: "16px", display: "flex", justifyContent: "flex-end" }}>
              <button
                type="submit"
                id="btn-calculate-shi"
                className="btn-primary"
                disabled={loading}
              >
                {loading ? (
                  <>
                    <div className="spinner" /> Calculating...
                  </>
                ) : (
                  <>
                    Calculate Soil Health Index <ArrowRight size={14} />
                  </>
                )}
              </button>
            </div>
          </form>
        </div>

        {/* RIGHT COLUMN: Evaluation Results */}
        <div>
          {result ? (
            <div className="academic-card">
              <div className="card-header-clean">
                <div>
                  <h3 className="card-title-clean">SHI Evaluation Result</h3>
                  <div className="card-subtitle-clean">Composite Soil Health Index score and limiting factors</div>
                </div>
                <span className="status-pill status-pill-green">Computed (ICAR Standards)</span>
              </div>

              {/* SHI Score Summary */}
              <div style={{ marginBottom: "16px", padding: "14px", backgroundColor: "var(--bg-subtle)", borderRadius: "var(--radius-sm)", border: "1px solid var(--border-color)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: "6px" }}>
                  <div>
                    <span style={{ fontSize: "1.4rem", fontWeight: 700, color: getShiBarColor(result.shi_score) }}>
                      {result.shi_score.toFixed(1)} / 100
                    </span>
                    <span style={{ marginLeft: "10px", fontSize: "0.85rem", fontWeight: 600, color: "var(--text-secondary)" }}>
                      {result.category}
                    </span>
                  </div>
                </div>

                <div className="shi-progress-bar">
                  <div
                    className="shi-progress-fill"
                    style={{
                      width: `${Math.min(Math.max(result.shi_score, 0), 100)}%`,
                      backgroundColor: getShiBarColor(result.shi_score),
                    }}
                  />
                </div>
              </div>

              {/* Limiting Factors */}
              <div style={{ marginBottom: "16px" }}>
                <div style={{ fontSize: "0.82rem", fontWeight: 600, color: "var(--text-secondary)", marginBottom: "8px" }}>
                  Identified Limiting Factors:
                </div>
                {result.limiting_factors && result.limiting_factors.length > 0 ? (
                  <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                    {result.limiting_factors.map((lf, i) => (
                      <div
                        key={i}
                        style={{
                          border: "1px solid var(--border-color)",
                          borderLeft: "3px solid var(--color-amber)",
                          padding: "8px 12px",
                          borderRadius: "var(--radius-sm)",
                          backgroundColor: "#ffffff",
                          fontSize: "0.8rem",
                        }}
                      >
                        <div style={{ display: "flex", justifyContent: "space-between", fontWeight: 600 }}>
                          <span>{lf.parameter}: {lf.status}</span>
                          <span style={{ color: "var(--color-red)" }}>Penalty: -{lf.penalty.toFixed(1)} pts</span>
                        </div>
                        <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "2px" }}>
                          Measured Value: <strong>{lf.measured_value}</strong> &bull; Optimal Range: {lf.reference_range}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="notice-box notice-box-success" style={{ margin: 0 }}>
                    All evaluated soil parameters meet or exceed minimum agronomic health thresholds.
                  </div>
                )}
              </div>

              {/* Parameter Status Table */}
              <div>
                <div style={{ fontSize: "0.82rem", fontWeight: 600, color: "var(--text-secondary)", marginBottom: "8px" }}>
                  Parameter Score Breakdown:
                </div>
                <div className="academic-table-container">
                  <table className="academic-table">
                    <thead>
                      <tr>
                        <th>Parameter</th>
                        <th>Measured</th>
                        <th>Score</th>
                        <th>Evaluation</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.parameter_evaluations &&
                        result.parameter_evaluations.map((p, idx) => (
                          <tr key={idx}>
                            <td style={{ fontWeight: 600, fontSize: "0.8rem" }}>{p.parameter_name}</td>
                            <td className="num-cell">{p.measured_value} {p.unit}</td>
                            <td className="num-cell" style={{ fontWeight: 600, color: p.score >= 70 ? "var(--color-green)" : p.score >= 40 ? "var(--color-amber)" : "var(--color-red)" }}>
                              {p.score.toFixed(0)} / 100
                            </td>
                            <td>
                              <span className={`status-pill ${p.status === "Optimal" || p.status === "Sufficient" ? "status-pill-green" : p.status === "Moderate" ? "status-pill-amber" : "status-pill-red"}`}>
                                {p.status}
                              </span>
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ) : (
            <div className="academic-card" style={{ textAlign: "center", padding: "48px 24px" }}>
              <Scale size={36} color="var(--text-muted)" style={{ margin: "0 auto 12px auto" }} />
              <h3 style={{ fontSize: "1rem", fontWeight: 600, color: "var(--text-primary)", marginBottom: "6px" }}>
                Awaiting Parameter Input
              </h3>
              <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", maxWidth: "380px", margin: "0 auto" }}>
                Enter soil test values or select a benchmark preset on the left, then click &ldquo;Calculate Soil Health Index&rdquo;.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
