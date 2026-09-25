import React from "react";
import {
  Activity,
  Sprout,
  Layers,
  Cpu,
  FileCheck2,
  BookOpen,
  GitMerge,
} from "lucide-react";

export default function Navbar({ activeTab, setActiveTab, backendStatus }) {
  const navItems = [
    { id: "dashboard", label: "Dashboard", icon: Activity },
    { id: "soil", label: "Soil Analysis", icon: Sprout },
    { id: "crop", label: "Crop Prediction", icon: Layers },
    { id: "sensors", label: "Sensor Monitoring", icon: Cpu },
    { id: "advisory", label: "Recommendations", icon: FileCheck2 },
    { id: "evidence", label: "Evidence Corpus", icon: BookOpen },
    { id: "intelligence", label: "Unified Pipeline", icon: GitMerge, badge: "Full Flow" },
  ];

  return (
    <header className="navbar">
      {/* Top Academic Header */}
      <div className="navbar-top">
        <div className="navbar-top-inner">
          <div
            className="nav-brand"
            role="button"
            tabIndex={0}
            onClick={() => setActiveTab("dashboard")}
            onKeyDown={(e) => e.key === "Enter" && setActiveTab("dashboard")}
          >
            <div className="brand-badge">
              <Sprout size={18} />
            </div>
            <div>
              <h1 className="brand-title">
                Explainable Multi-Source Soil Health Intelligence System
              </h1>
              <div className="brand-subtitle">
                Soil Health Intelligence &amp; Agricultural Decision Support &bull; University Research Prototype
              </div>
            </div>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <div className="nav-status-badge">
              {backendStatus === "ok" ? (
                <>
                  <div className="status-dot-active" />
                  <span>Backend API Online</span>
                </>
              ) : (
                <>
                  <div className="status-dot-error" />
                  <span>Checking Backend...</span>
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tab Bar */}
      <nav className="navbar-nav" aria-label="Main Navigation">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              id={`nav-${item.id}`}
              className={`nav-tab-btn ${isActive ? "active" : ""}`}
              onClick={() => setActiveTab(item.id)}
            >
              <Icon size={15} />
              <span>{item.label}</span>
              {item.badge && <span className="nav-badge-pill">{item.badge}</span>}
            </button>
          );
        })}
      </nav>
    </header>
  );
}
