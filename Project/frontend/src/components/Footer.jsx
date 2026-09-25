import React from "react";
import { Award, GraduationCap } from "lucide-react";

export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-inner">
        <div style={{ display: "flex", gap: "16px", alignItems: "center", flexWrap: "wrap", justifyContent: "center" }}>
          <span style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
            <GraduationCap size={15} color="var(--color-green-dark)" />
            <strong>Marwadi University</strong> &mdash; Department of Information &amp; Communication Technology
          </span>
          <span>&bull;</span>
          <span style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
            <Award size={15} color="var(--color-blue)" />
            Explainable Multi-Source Soil Health Intelligence System
          </span>
          <span>&bull;</span>
          <span style={{ color: "var(--text-muted)", fontSize: "0.78rem" }}>
            University Research Prototype
          </span>
        </div>
      </div>
    </footer>
  );
}
