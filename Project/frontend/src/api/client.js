/**
 * API Client for Soil Health Intelligence System Backend.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function request(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  const headers = {
    "Content-Type": "application/json",
    ...(options.headers || {}),
  };

  try {
    const response = await fetch(url, { ...options, headers });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(errorData.detail || `Request failed with status ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`API Error on ${endpoint}:`, error);
    throw error;
  }
}

export const api = {
  // 1. Health Check
  getHealth: () => request("/health"),

  // 2. Soil Analysis (SHI)
  analyzeSoil: (soilData) =>
    request("/api/analyze", {
      method: "POST",
      body: JSON.stringify(soilData),
    }),

  // 3. Crop Suitability Prediction & XAI
  predictCrop: (features) =>
    request("/api/predict", {
      method: "POST",
      body: JSON.stringify(features),
    }),

  // 4. Grounded Advisory
  getAdvisory: (advisoryPayload) =>
    request("/api/advisory", {
      method: "POST",
      body: JSON.stringify(advisoryPayload),
    }),

  // 5. Complete Unified Intelligence Pipeline
  runUnifiedIntelligence: (payload) =>
    request("/api/intelligence", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  // 6. Sensor Telemetry Simulation Stream
  getSensorStream: (params = {}) => {
    const query = new URLSearchParams();
    if (params.count) query.append("count", params.count);
    if (params.step) query.append("step", params.step);
    if (params.year) query.append("year", params.year);
    if (params.reset) query.append("reset", params.reset);
    return request(`/api/sensors?${query.toString()}`);
  },

  // 7. Scientific Knowledge Base Corpus
  getKnowledgeCorpus: () => request("/api/corpus"),

  // 8. Empirical Benchmark Sample Profiles
  getSampleProfiles: () => request("/api/samples"),
};

export default api;
