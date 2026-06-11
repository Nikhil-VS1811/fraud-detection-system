import React, { useEffect, useState } from "react";
import axios from "axios";

const API_BASE =
  (typeof import.meta !== "undefined" && import.meta.env?.VITE_API_URL) || "/api";

export default function ModelInfoCard() {
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    axios.get(`${API_BASE}/model-info`)
      .then((res) => {
        if (!cancelled) setInfo(res.data);
      })
      .catch((err) => {
        if (!cancelled)
          setError(err?.response?.data?.message || err.message || "Request failed");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const rows = info
    ? [
        { label: "Model", value: info.model ?? "—" },
        {
          label: "Dataset Size",
          value:
            info.dataset_size != null
              ? Number(info.dataset_size).toLocaleString()
              : "—",
        },
        { label: "Features", value: info.features ?? "—" },
        {
          label: "Fraud Samples",
          value:
            info.fraud_samples != null
              ? Number(info.fraud_samples).toLocaleString()
              : "—",
        },
        { label: "SMOTE Applied", value: info.smote_applied ? "Yes" : "No" },
        {
          label: "ROC-AUC",
          value: info.roc_auc != null ? `${info.roc_auc}%` : "—",
        },
      ]
    : [];

  return (
    <div className="model-info-card">
      <div className="model-info-header">
        <h3 className="model-info-title">Model Information</h3>
        <span className="model-info-subtitle">Training & dataset overview</span>
      </div>

      {loading && (
        <div className="model-info-grid">
          {Array.from({ length: 6 }).map((_, i) => (
            <div className="model-info-item skeleton" key={i}>
              <div className="skeleton-line short" />
              <div className="skeleton-line long" />
            </div>
          ))}
        </div>
      )}

      {!loading && error && (
        <div className="model-info-error">
          Failed to load model info: {error}
        </div>
      )}

      {!loading && !error && info && (
        <div className="model-info-grid">
          {rows.map((r) => (
            <div className="model-info-item" key={r.label}>
              <span className="model-info-label">{r.label}</span>
              <span className="model-info-value">{r.value}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
