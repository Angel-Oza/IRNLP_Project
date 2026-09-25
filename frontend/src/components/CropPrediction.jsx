import React, { useState } from "react";
import {
  Layers,
  Zap,
  RefreshCw,
  ArrowRight,
} from "lucide-react";
import api from "../api/client";

export default function CropPrediction({ initialCropFeatures }) {
  const defaultFeatures = {
    N: 90.0,
    P: 42.0,
    K: 43.0,
    temperature: 20.87,
    humidity: 82.0,
    ph: 6.5,
    rainfall: 202.93,
  };

  const [features, setFeatures] = useState(initialCropFeatures || defaultFeatures);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const featureMeta = [
    { key: "N", name: "Available Nitrogen (N)", unit: "kg/ha", hint: "0 - 140 kg/ha" },
    { key: "P", name: "Available Phosphorus (P)", unit: "kg/ha", hint: "5 - 145 kg/ha" },
    { key: "K", name: "Available Potassium (K)", unit: "kg/ha", hint: "5 - 205 kg/ha" },
    { key: "temperature", name: "Ambient Temperature", unit: "°C", hint: "8.8 - 43.7 °C" },
    { key: "humidity", name: "Relative Humidity", unit: "%", hint: "14 - 100 %" },
    { key: "ph", name: "Soil Reaction (pH)", unit: "pH", hint: "3.5 - 9.9" },
    { key: "rainfall", name: "Precipitation / Rainfall", unit: "mm", hint: "20 - 298 mm" },
  ];

  const handleInputChange = (key, value) => {
    setFeatures((prev) => ({
      ...prev,
      [key]: value === "" ? "" : parseFloat(value),
    }));
  };

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await api.predictCrop(features);
      setPrediction(res);
    } catch (err) {
      setError(err.message || "Failed to predict crop suitability.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFeatures(defaultFeatures);
    setPrediction(null);
    setError(null);
  };

  const loadPreset = (cropType) => {
    if (cropType === "rice") {
      setFeatures({ N: 90.0, P: 42.0, K: 43.0, temperature: 20.87, humidity: 82.0, ph: 6.5, rainfall: 202.93 });
    } else if (cropType === "maize") {
      setFeatures({ N: 78.0, P: 45.0, K: 20.0, temperature: 22.5, humidity: 65.0, ph: 6.2, rainfall: 85.0 });
    } else if (cropType === "cotton") {
      setFeatures({ N: 120.0, P: 40.0, K: 25.0, temperature: 28.0, humidity: 55.0, ph: 7.2, rainfall: 75.0 });
    } else if (cropType === "coffee") {
      setFeatures({ N: 100.0, P: 25.0, K: 30.0, temperature: 24.0, humidity: 60.0, ph: 6.8, rainfall: 160.0 });
    }
    setPrediction(null);
  };

  // Compute max absolute SHAP value for scaling bars
  const maxShap = prediction
    ? Math.max(...prediction.feature_attributions.map((a) => Math.abs(a.shap_value)), 0.05)
    : 0.1;

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title">
          <Layers size={22} color="var(--color-blue)" />
          Crop Prediction &amp; TreeSHAP Explainable AI
        </h2>
        <p className="page-desc">
          Predicts optimal crop suitability using the trained Random Forest classifier (N=22 crop classes) and decomposes exact local feature attributions via TreeSHAP.
        </p>
      </div>

      <div className="grid-2" style={{ alignItems: "flex-start" }}>
        {/* LEFT COLUMN: Input Form */}
        <div className="academic-card">
          <div className="card-header-clean">
            <div>
              <h3 className="card-title-clean">Agronomic Feature Inputs</h3>
              <div className="card-subtitle-clean">Provide 7 environmental and chemical features for inference</div>
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
          <div style={{ marginBottom: "14px", display: "flex", alignItems: "center", gap: "6px", flexWrap: "wrap" }}>
            <span style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--text-muted)" }}>Benchmark Profiles:</span>
            <button type="button" className="btn-secondary btn-sm" onClick={() => loadPreset("rice")}>Rice Profile</button>
            <button type="button" className="btn-secondary btn-sm" onClick={() => loadPreset("maize")}>Maize Profile</button>
            <button type="button" className="btn-secondary btn-sm" onClick={() => loadPreset("cotton")}>Cotton Profile</button>
            <button type="button" className="btn-secondary btn-sm" onClick={() => loadPreset("coffee")}>Coffee Profile</button>
          </div>

          <form onSubmit={handleSubmit}>
            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
              {featureMeta.map((f) => (
                <div key={f.key} className="form-group" style={{ marginBottom: "6px" }}>
                  <label className="form-label" htmlFor={`crop-input-${f.key}`}>
                    <span>{f.name}</span>
                    <span className="form-unit">{f.unit}</span>
                  </label>
                  <input
                    id={`crop-input-${f.key}`}
                    type="number"
                    step="any"
                    className="form-input"
                    value={features[f.key] !== undefined ? features[f.key] : ""}
                    onChange={(e) => handleInputChange(f.key, e.target.value)}
                    placeholder={`e.g. ${f.hint}`}
                    required
                  />
                </div>
              ))}
            </div>

            {error && (
              <div className="notice-box notice-box-warning" style={{ marginTop: "12px" }}>
                <strong>Error:</strong> {error}
              </div>
            )}

            <div style={{ marginTop: "16px", display: "flex", justifyContent: "flex-end" }}>
              <button
                type="submit"
                id="btn-predict-crop"
                className="btn-primary"
                disabled={loading}
              >
                {loading ? (
                  <>
                    <div className="spinner" /> Running Classifier...
                  </>
                ) : (
                  <>
                    Predict Crop Suitability <ArrowRight size={14} />
                  </>
                )}
              </button>
            </div>
          </form>
        </div>

        {/* RIGHT COLUMN: Inference & SHAP Results */}
        <div>
          {prediction ? (
            <div className="academic-card">
              <div className="card-header-clean">
                <div>
                  <h3 className="card-title-clean">Suitability &amp; Attributions</h3>
                  <div className="card-subtitle-clean">Random Forest inference and TreeSHAP decomposition</div>
                </div>
                <span className="status-pill status-pill-green">Inference Complete</span>
              </div>

              {/* Prediction Banner */}
              <div style={{ padding: "14px", backgroundColor: "var(--color-green-light)", border: "1px solid var(--color-green-border)", borderRadius: "var(--radius-sm)", marginBottom: "16px" }}>
                <div style={{ fontSize: "0.75rem", textTransform: "uppercase", fontWeight: 700, color: "var(--color-green-dark)" }}>
                  Optimal Recommended Crop
                </div>
                <div style={{ fontSize: "1.6rem", fontWeight: 700, color: "var(--color-green-dark)", textTransform: "capitalize", marginTop: "2px" }}>
                  {prediction.predicted_crop}
                </div>
                <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: "4px" }}>
                  Model Confidence: <strong>{(prediction.confidence * 100).toFixed(1)}%</strong> &bull; Classifier: Random Forest (N=22 Classes)
                </div>
              </div>

              {/* TreeSHAP Horizontal Bar Chart */}
              <div style={{ marginBottom: "16px" }}>
                <div style={{ fontSize: "0.82rem", fontWeight: 600, color: "var(--text-secondary)", marginBottom: "6px" }}>
                  Local Feature Attributions (TreeSHAP &phi;<sub>i</sub>):
                </div>

                <div className="shap-chart">
                  {prediction.feature_attributions &&
                    prediction.feature_attributions.map((feat) => {
                      const isPos = feat.shap_value >= 0;
                      const barWidthPercent = (Math.abs(feat.shap_value) / maxShap) * 50;

                      return (
                        <div key={feat.feature} className="shap-row">
                          <div className="shap-feat-name">
                            <span>{feat.feature}</span>
                            <span style={{ fontSize: "0.72rem", color: "var(--text-dim)", marginLeft: "4px", fontFamily: "var(--font-mono)" }}>
                              ({feat.feature_value})
                            </span>
                          </div>

                          <div className="shap-track-box">
                            <div className="shap-zero-line" />
                            {isPos ? (
                              <div
                                className="shap-bar-positive"
                                style={{ width: `${Math.max(barWidthPercent, 1)}%` }}
                              />
                            ) : (
                              <div
                                className="shap-bar-negative"
                                style={{ width: `${Math.max(barWidthPercent, 1)}%` }}
                              />
                            )}
                          </div>

                          <div
                            className="shap-score-text"
                            style={{
                              color: isPos ? "var(--color-green)" : "#dc2626",
                              textAlign: "right",
                            }}
                          >
                            {isPos ? `+${feat.shap_value.toFixed(3)}` : feat.shap_value.toFixed(3)}
                          </div>
                        </div>
                      );
                    })}
                </div>

                <div style={{ marginTop: "10px", fontSize: "0.72rem", color: "var(--text-dim)", display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "#dc2626" }}>&larr; Negative Feature Driver</span>
                  <span>Baseline</span>
                  <span style={{ color: "var(--color-green)" }}>Positive Feature Driver &rarr;</span>
                </div>
              </div>

              {/* Class Probability Distribution Table */}
              <div>
                <div style={{ fontSize: "0.82rem", fontWeight: 600, color: "var(--text-secondary)", marginBottom: "8px" }}>
                  Top Class Probability Distribution:
                </div>
                <div className="academic-table-container">
                  <table className="academic-table">
                    <thead>
                      <tr>
                        <th>Crop Class</th>
                        <th>Probability Score</th>
                        <th>Distribution Bar</th>
                      </tr>
                    </thead>
                    <tbody>
                      {prediction.class_probabilities &&
                        prediction.class_probabilities.slice(0, 5).map((cp, idx) => (
                          <tr key={idx}>
                            <td style={{ textTransform: "capitalize", fontWeight: idx === 0 ? 600 : 400 }}>
                              {cp.crop_name} {idx === 0 && <span style={{ fontSize: "0.7rem", color: "var(--color-green-dark)" }}>(Top Match)</span>}
                            </td>
                            <td className="num-cell">
                              {(cp.probability * 100).toFixed(1)}%
                            </td>
                            <td style={{ width: "40%" }}>
                              <div style={{ height: "6px", backgroundColor: "var(--bg-subtle)", borderRadius: "2px", overflow: "hidden" }}>
                                <div
                                  style={{
                                    height: "100%",
                                    width: `${cp.probability * 100}%`,
                                    backgroundColor: idx === 0 ? "var(--color-green)" : "var(--color-slate)",
                                  }}
                                />
                              </div>
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
              <Zap size={36} color="var(--text-muted)" style={{ margin: "0 auto 12px auto" }} />
              <h3 style={{ fontSize: "1rem", fontWeight: 600, color: "var(--text-primary)", marginBottom: "6px" }}>
                Awaiting Feature Inputs
              </h3>
              <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", maxWidth: "380px", margin: "0 auto" }}>
                Enter agronomic values on the left or select a benchmark preset to evaluate crop suitability and compute TreeSHAP values.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
