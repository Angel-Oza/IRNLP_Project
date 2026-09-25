import React, { useState, useEffect } from "react";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import Dashboard from "./components/Dashboard";
import SoilAnalysis from "./components/SoilAnalysis";
import CropPrediction from "./components/CropPrediction";
import SensorMonitoring from "./components/SensorMonitoring";
import Recommendations from "./components/Recommendations";
import EvidenceCorpus from "./components/EvidenceCorpus";
import IntelligencePipeline from "./components/IntelligencePipeline";
import api from "./api/client";

export default function App() {
  const [activeTab, setActiveTab] = useState("dashboard");
  const [backendStatus, setBackendStatus] = useState("checking");
  const [selectedSample, setSelectedSample] = useState(null);
  const [advisoryData, setAdvisoryData] = useState(null);

  useEffect(() => {
    async function checkBackend() {
      try {
        const res = await api.getHealth();
        setBackendStatus(res.status === "ok" ? "ok" : "error");
      } catch (err) {
        console.warn("Backend health check failed:", err);
        setBackendStatus("error");
      }
    }
    checkBackend();
    const interval = setInterval(checkBackend, 15000);
    return () => clearInterval(interval);
  }, []);

  const handleSelectSample = (sample) => {
    setSelectedSample(sample);
  };

  return (
    <div className="app-container">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        backendStatus={backendStatus}
      />

      <main className="main-content">
        {activeTab === "dashboard" && (
          <Dashboard
            setActiveTab={setActiveTab}
            onSelectSample={handleSelectSample}
          />
        )}

        {activeTab === "intelligence" && (
          <IntelligencePipeline
            initialPayload={
              selectedSample
                ? { ...selectedSample.soil_data, ...selectedSample.crop_features }
                : null
            }
            onAdvisoryGenerated={(adv) => setAdvisoryData(adv)}
          />
        )}

        {activeTab === "soil" && (
          <SoilAnalysis
            initialSoilData={selectedSample ? selectedSample.soil_data : null}
          />
        )}

        {activeTab === "crop" && (
          <CropPrediction
            initialCropFeatures={selectedSample ? selectedSample.crop_features : null}
          />
        )}

        {activeTab === "sensors" && <SensorMonitoring />}

        {activeTab === "advisory" && (
          <Recommendations
            advisoryData={advisoryData}
            onGenerateNew={() => setActiveTab("intelligence")}
          />
        )}

        {activeTab === "evidence" && <EvidenceCorpus />}
      </main>

      <Footer />
    </div>
  );
}
