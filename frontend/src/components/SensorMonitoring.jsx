import React, { useEffect, useState, useRef, useCallback } from "react";
import {
  Cpu,
  Play,
  Pause,
  RotateCcw,
  FastForward,
} from "lucide-react";
import api from "../api/client";

// Clean Academic Engineering Sensor Line Chart Component
function EngineeringSensorChart({
  title,
  unit,
  data = [],
  getValue,
  color = "#15803d",
  decimals = 1,
  minSuggested,
  maxSuggested,
}) {
  const points = data
    .map((d, i) => {
      const rawVal = getValue ? getValue(d) : undefined;
      const numVal = typeof rawVal === "number" && !isNaN(rawVal) ? rawVal : null;
      return {
        val: numVal,
        timestamp: d.timestamp || `Frame ${d.index ?? i}`,
        index: d.index ?? i,
      };
    })
    .filter((p) => p.val !== null);

  const currentPoint = points.length > 0 ? points[points.length - 1] : null;
  const currentVal = currentPoint ? currentPoint.val : null;

  let yMin = minSuggested;
  let yMax = maxSuggested;

  if (points.length > 0) {
    const dataMin = Math.min(...points.map((p) => p.val));
    const dataMax = Math.max(...points.map((p) => p.val));

    if (yMin === undefined) {
      const pad = (dataMax - dataMin) * 0.15 || (Math.abs(dataMin) * 0.1) || 1.0;
      yMin = Math.max(0, dataMin - pad);
    }
    if (yMax === undefined) {
      const pad = (dataMax - dataMin) * 0.15 || (Math.abs(dataMax) * 0.1) || 1.0;
      yMax = dataMax + pad;
    }
    if (yMin === yMax) {
      yMin = Math.max(0, yMin - 1);
      yMax += 1;
    }
  } else {
    yMin = yMin ?? 0;
    yMax = yMax ?? 100;
  }

  const range = yMax - yMin || 1;
  const midVal = (yMin + yMax) / 2;

  const width = 300;
  const height = 130;
  const padLeft = 46;
  const padRight = 14;
  const padTop = 14;
  const padBottom = 26;

  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;

  const coords = points.map((p, i) => {
    const x = padLeft + (i / Math.max(points.length - 1, 1)) * chartW;
    const clampedVal = Math.min(Math.max(p.val, yMin), yMax);
    const y = padTop + chartH - ((clampedVal - yMin) / range) * chartH;
    return { x, y, val: p.val, time: p.timestamp };
  });

  const pathLine =
    coords.length > 0
      ? `M ${coords.map((c) => `${c.x.toFixed(1)},${c.y.toFixed(1)}`).join(" L ")}`
      : "";
  const pathArea =
    coords.length > 0
      ? `${pathLine} L ${coords[coords.length - 1].x.toFixed(1)},${(padTop + chartH).toFixed(1)} L ${coords[0].x.toFixed(1)},${(padTop + chartH).toFixed(1)} Z`
      : "";

  const formatTime = (ts) => {
    if (!ts) return "";
    if (ts.includes(" ")) {
      const timePart = ts.split(" ")[1];
      if (timePart) {
        return timePart.substring(0, 5);
      }
    }
    return ts;
  };

  const startTimeLabel = points.length > 0 ? formatTime(points[0].timestamp) : "";
  const midTimeLabel =
    points.length > 2 ? formatTime(points[Math.floor(points.length / 2)].timestamp) : "";
  const latestTimeLabel =
    points.length > 0 ? formatTime(points[points.length - 1].timestamp) : "";

  return (
    <div className="sensor-chart-card">
      <div className="sensor-chart-header">
        <span className="sensor-chart-title">{title}</span>
        <span className="sensor-chart-current">
          {currentVal !== null ? currentVal.toFixed(decimals) : "—"}{" "}
          <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: "normal" }}>
            {unit}
          </span>
        </span>
      </div>

      <svg
        width="100%"
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        style={{ overflow: "visible" }}
      >
        <line
          x1={padLeft}
          y1={padTop}
          x2={width - padRight}
          y2={padTop}
          stroke="#e2e8f0"
          strokeDasharray="2 2"
        />
        <text
          x={padLeft - 6}
          y={padTop + 4}
          textAnchor="end"
          fontSize="9"
          fill="#64748b"
          fontFamily="monospace"
        >
          {yMax.toFixed(decimals)}
        </text>

        <line
          x1={padLeft}
          y1={padTop + chartH / 2}
          x2={width - padRight}
          y2={padTop + chartH / 2}
          stroke="#e2e8f0"
          strokeDasharray="2 2"
        />
        <text
          x={padLeft - 6}
          y={padTop + chartH / 2 + 3}
          textAnchor="end"
          fontSize="9"
          fill="#64748b"
          fontFamily="monospace"
        >
          {midVal.toFixed(decimals)}
        </text>

        <line
          x1={padLeft}
          y1={padTop + chartH}
          x2={width - padRight}
          y2={padTop + chartH}
          stroke="#cbd5e1"
        />
        <text
          x={padLeft - 6}
          y={padTop + chartH + 3}
          textAnchor="end"
          fontSize="9"
          fill="#64748b"
          fontFamily="monospace"
        >
          {yMin.toFixed(decimals)}
        </text>

        <line x1={padLeft} y1={padTop} x2={padLeft} y2={padTop + chartH} stroke="#cbd5e1" />

        {pathArea && <path d={pathArea} fill={color} fillOpacity="0.08" />}

        {pathLine && (
          <path
            d={pathLine}
            fill="none"
            stroke={color}
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        )}

        {coords.map((c, idx) => {
          const isLatest = idx === coords.length - 1;
          return (
            <circle
              key={idx}
              cx={c.x}
              cy={c.y}
              r={isLatest ? "4" : "2"}
              fill={isLatest ? color : "#ffffff"}
              stroke={color}
              strokeWidth={isLatest ? "2" : "1.5"}
            >
              <title>{`${c.time}: ${c.val.toFixed(decimals)} ${unit}`}</title>
            </circle>
          );
        })}

        {startTimeLabel && (
          <text x={padLeft} y={height - 6} fontSize="9" fill="#64748b" fontFamily="monospace">
            {startTimeLabel}
          </text>
        )}
        {midTimeLabel && (
          <text
            x={padLeft + chartW / 2}
            y={height - 6}
            textAnchor="middle"
            fontSize="9"
            fill="#64748b"
            fontFamily="monospace"
          >
            {midTimeLabel}
          </text>
        )}
        {latestTimeLabel && (
          <text
            x={width - padRight}
            y={height - 6}
            textAnchor="end"
            fontSize="9"
            fill="#64748b"
            fontFamily="monospace"
          >
            {latestTimeLabel}
          </text>
        )}
      </svg>
    </div>
  );
}

export default function SensorMonitoring() {
  const [streamData, setStreamData] = useState(null);
  const [history, setHistory] = useState([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [stepSize, setStepSize] = useState(1);
  const [selectedYear, setSelectedYear] = useState(2016);
  const [loading, setLoading] = useState(false);
  const timerRef = useRef(null);

  const fetchNextFrames = useCallback(
    async (reset = false) => {
      setLoading(true);
      try {
        const res = await api.getSensorStream({
          count: 1,
          step: stepSize,
          year: selectedYear,
          reset: reset,
        });
        setStreamData(res);
        if (res.frames && res.frames.length > 0) {
          setHistory((prev) => {
            const updated = [res.frames[0], ...prev];
            return updated.slice(0, 15);
          });
        }
      } catch (err) {
        console.error("Sensor stream error:", err);
      } finally {
        setLoading(false);
      }
    },
    [selectedYear, stepSize]
  );

  useEffect(() => {
    fetchNextFrames(true);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [fetchNextFrames]);

  useEffect(() => {
    if (isPlaying) {
      timerRef.current = setInterval(() => {
        fetchNextFrames(false);
      }, 1500);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isPlaying, fetchNextFrames]);

  const reliability = streamData?.reliability || {
    composite_reliability: 0.854,
    s_range: 0.997,
    s_dist: 0.857,
    s_avail: 1.0,
    s_step: 1.0,
    status_label: "Nominal",
  };

  // chronological order for line charts
  const chartHistory = history.slice().reverse();

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title">
          <Cpu size={22} color="var(--color-amber)" />
          Sensor Telemetry &amp; Dynamic Reliability Stream
        </h2>
        <p className="page-desc">
          Chronological software replay of historical in-situ environmental sensors from Dataset B (Berambadi Observatory 2016–2025) with dynamic reliability estimation r(t).
        </p>
      </div>

      {/* Dataset & Simulation Notice */}
      <div className="notice-box notice-box-info" style={{ marginBottom: "16px" }}>
        <strong>Software Replay Specification:</strong> Replaying historical in-situ telemetry from the 10-year Berambadi Hydrological Observatory dataset for offline algorithmic robustness and gated fusion evaluation.
      </div>

      {/* Playback Controls & Status Bar */}
      <div className="academic-card" style={{ marginBottom: "16px", padding: "14px 18px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "12px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
            <button
              id="btn-stream-toggle"
              className="btn-primary"
              style={{ padding: "6px 14px", fontSize: "0.82rem" }}
              onClick={() => setIsPlaying(!isPlaying)}
            >
              {isPlaying ? (
                <>
                  <Pause size={14} /> Pause Stream
                </>
              ) : (
                <>
                  <Play size={14} /> Play Stream
                </>
              )}
            </button>

            <button
              id="btn-stream-step"
              className="btn-secondary btn-sm"
              onClick={() => fetchNextFrames(false)}
              disabled={isPlaying || loading}
            >
              <FastForward size={13} /> Step Frame (+{stepSize})
            </button>

            <button
              id="btn-stream-reset"
              className="btn-secondary btn-sm"
              onClick={() => fetchNextFrames(true)}
              disabled={isPlaying}
            >
              <RotateCcw size={13} /> Reset Stream
            </button>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "12px", flexWrap: "wrap" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
              <label style={{ fontSize: "0.78rem", fontWeight: 600, color: "var(--text-secondary)" }}>
                Telemetry Year:
              </label>
              <select
                className="form-input"
                style={{ padding: "4px 8px", fontSize: "0.78rem", width: "auto" }}
                value={selectedYear}
                onChange={(e) => setSelectedYear(parseInt(e.target.value))}
              >
                {Array.from({ length: 10 }, (_, i) => 2016 + i).map((yr) => (
                  <option key={yr} value={yr}>
                    {yr} Records
                  </option>
                ))}
              </select>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
              <label style={{ fontSize: "0.78rem", fontWeight: 600, color: "var(--text-secondary)" }}>
                Step Size:
              </label>
              <select
                className="form-input"
                style={{ padding: "4px 8px", fontSize: "0.78rem", width: "auto" }}
                value={stepSize}
                onChange={(e) => setStepSize(parseInt(e.target.value))}
              >
                <option value={1}>1 Frame (15 min)</option>
                <option value={4}>4 Frames (1 hour)</option>
                <option value={12}>12 Frames (3 hours)</option>
                <option value={24}>24 Frames (6 hours)</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* 4 Sensor Engineering Charts with Correct Field Mapping */}
      <div className="stat-card-grid" style={{ marginBottom: "20px" }}>
        {/* Chart 1: Soil Moisture */}
        <EngineeringSensorChart
          title="Soil Moisture (5cm Depth)"
          unit="vol %"
          data={chartHistory}
          getValue={(f) =>
            f.SM_5cm !== undefined && f.SM_5cm !== null
              ? f.SM_5cm
              : f.SM_50cm !== undefined && f.SM_50cm !== null
              ? f.SM_50cm
              : f.soil_moisture
          }
          decimals={1}
          color="#0284c7"
        />

        {/* Chart 2: Soil Temperature */}
        <EngineeringSensorChart
          title="Soil Temperature (5cm Depth)"
          unit="°C"
          data={chartHistory}
          getValue={(f) =>
            f.Temp_5cm !== undefined && f.Temp_5cm !== null
              ? f.Temp_5cm
              : f.Temp_50cm !== undefined && f.Temp_50cm !== null
              ? f.Temp_50cm
              : f.soil_temperature
          }
          decimals={1}
          color="#d97706"
        />

        {/* Chart 3: Electrical Conductivity */}
        <EngineeringSensorChart
          title="Electrical Conductivity (5cm)"
          unit="dS/m"
          data={chartHistory}
          getValue={(f) =>
            f.EC_5cm !== undefined && f.EC_5cm !== null
              ? f.EC_5cm
              : f.EC_50cm !== undefined && f.EC_50cm !== null
              ? f.EC_50cm
              : f.ec
          }
          decimals={2}
          color="#15803d"
        />

        {/* Chart 4: Rainfall */}
        <EngineeringSensorChart
          title="Precipitation / Rainfall"
          unit="mm"
          data={chartHistory}
          getValue={(f) =>
            f.Precipitation !== undefined && f.Precipitation !== null
              ? f.Precipitation
              : f.rainfall !== undefined && f.rainfall !== null
              ? f.rainfall
              : 0
          }
          decimals={1}
          minSuggested={0}
          maxSuggested={10}
          color="#4f46e5"
        />
      </div>

      {/* Two-Column Grid: Reliability Breakdown + Telemetry History Table */}
      <div className="grid-2">
        {/* Dynamic Reliability Evaluation Card */}
        <div className="academic-card">
          <div className="card-header-clean">
            <div>
              <h3 className="card-title-clean">Dynamic Reliability Score r(t)</h3>
              <div className="card-subtitle-clean">
                Composite online sensor trust metric decomposed across 4 health sub-scores
              </div>
            </div>
            <span className="status-pill status-pill-green">{reliability.status_label}</span>
          </div>

          <div
            style={{
              marginBottom: "16px",
              padding: "12px",
              backgroundColor: "var(--bg-subtle)",
              borderRadius: "var(--radius-sm)",
              border: "1px solid var(--border-color)",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "baseline",
                marginBottom: "4px",
              }}
            >
              <span
                style={{ fontSize: "1.3rem", fontWeight: 700, color: "var(--color-green-dark)" }}
              >
                r(t) = {(reliability.composite_reliability * 100).toFixed(1)}%
              </span>
              <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                Threshold for Fusion: &ge; 70.0%
              </span>
            </div>
            <div className="shi-progress-bar">
              <div
                className="shi-progress-fill"
                style={{
                  width: `${reliability.composite_reliability * 100}%`,
                  backgroundColor:
                    reliability.composite_reliability >= 0.85
                      ? "var(--color-green)"
                      : reliability.composite_reliability >= 0.6
                      ? "var(--color-amber)"
                      : "var(--color-red)",
                }}
              />
            </div>
          </div>

          <div className="academic-table-container">
            <table className="academic-table">
              <thead>
                <tr>
                  <th>Component Health Metric</th>
                  <th>Formula / Meaning</th>
                  <th>Score</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ fontWeight: 600 }}>s<sub>range</sub> (Physical Range)</td>
                  <td style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                    Valid physical sensor bounds
                  </td>
                  <td className="num-cell">{(reliability.s_range * 100).toFixed(0)}%</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>s<sub>dist</sub> (Distributional Z-Score)</td>
                  <td style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                    Seasonal mean deviation
                  </td>
                  <td className="num-cell">{(reliability.s_dist * 100).toFixed(0)}%</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>s<sub>avail</sub> (Availability &amp; Latency)</td>
                  <td style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                    Data freshness &amp; completeness
                  </td>
                  <td className="num-cell">{(reliability.s_avail * 100).toFixed(0)}%</td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>s<sub>step</sub> (Rate-of-Change)</td>
                  <td style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                    Physically plausibility of gradient
                  </td>
                  <td className="num-cell">{(reliability.s_step * 100).toFixed(0)}%</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Chronological Telemetry Frame History Table */}
        <div className="academic-card">
          <div className="card-header-clean">
            <div>
              <h3 className="card-title-clean">Recent Telemetry Frames</h3>
              <div className="card-subtitle-clean">
                Last chronological frames retrieved from Dataset B
              </div>
            </div>
          </div>

          <div className="academic-table-container">
            <table className="academic-table">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Moisture (5cm)</th>
                  <th>Temp (5cm)</th>
                  <th>EC (5cm)</th>
                  <th>Precipitation</th>
                </tr>
              </thead>
              <tbody>
                {history.slice(0, 8).map((frame, idx) => {
                  const sm =
                    frame.SM_5cm !== undefined && frame.SM_5cm !== null
                      ? frame.SM_5cm
                      : frame.SM_50cm !== undefined && frame.SM_50cm !== null
                      ? frame.SM_50cm
                      : frame.soil_moisture;
                  const temp =
                    frame.Temp_5cm !== undefined && frame.Temp_5cm !== null
                      ? frame.Temp_5cm
                      : frame.Temp_50cm !== undefined && frame.Temp_50cm !== null
                      ? frame.Temp_50cm
                      : frame.soil_temperature;
                  const ec =
                    frame.EC_5cm !== undefined && frame.EC_5cm !== null
                      ? frame.EC_5cm
                      : frame.EC_50cm !== undefined && frame.EC_50cm !== null
                      ? frame.EC_50cm
                      : frame.ec;
                  const rain =
                    frame.Precipitation !== undefined && frame.Precipitation !== null
                      ? frame.Precipitation
                      : frame.rainfall !== undefined && frame.rainfall !== null
                      ? frame.rainfall
                      : 0;

                  return (
                    <tr
                      key={idx}
                      style={{
                        backgroundColor: idx === 0 ? "var(--color-green-light)" : undefined,
                      }}
                    >
                      <td style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem" }}>
                        {frame.timestamp || `Frame #${frame.index ?? idx}`}
                      </td>
                      <td className="num-cell">{sm !== undefined && sm !== null ? sm.toFixed(1) : "—"}%</td>
                      <td className="num-cell">{temp !== undefined && temp !== null ? temp.toFixed(1) : "—"}°C</td>
                      <td className="num-cell">{ec !== undefined && ec !== null ? ec.toFixed(2) : "—"}</td>
                      <td className="num-cell">{rain !== undefined && rain !== null ? rain.toFixed(1) : "0.0"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
