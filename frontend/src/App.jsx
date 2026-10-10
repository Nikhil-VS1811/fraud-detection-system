import { useEffect, useState } from "react";
import axios from "axios";
import AnalyticsCharts from "./components/AnalyticsCharts";
import TopContributors from "./components/TopContributors";
import ModelInfoCard from "./components/ModelInfoCard";
import { SAMPLE_TRANSACTIONS } from "./sampleData";

import "./App.css";

const API_URL =
  import.meta.env.VITE_API_URL ||
  "https://fraud-backend-km6z.onrender.com";

const api = axios.create({
  baseURL: API_URL,
  headers: { "Content-Type": "application/json" },
});

const FEATURE_NAMES = [
  "Time",
  ...Array.from({ length: 28 }, (_, i) => `V${i + 1}`),
  "Amount",
];

const METRIC_KEYS = [
  { key: "precision", label: "Precision" },
  { key: "recall", label: "Recall" },
  { key: "f1", label: "F1 Score" },
  { key: "roc_auc", label: "ROC-AUC" },
];

function MetricCard({ label, value }) {
  const pct =
    typeof value === "number" ? `${value}%` : "—";

  const fill =
    typeof value === "number"
      ? Math.max(0, Math.min(100, value))
      : 0;
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{pct}</div>
      <div className="metric-bar">
        <div className="metric-bar-fill" style={{ width: `${fill}%` }} />
      </div>
    </div>
  );
}

function PredictionForm({ onPredicted, sampleData }) {
  const [features, setFeatures] = useState(Array(30).fill(""));
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [activeSample, setActiveSample] = useState(null);

  const updateFeature = (i, v) => {
    const next = [...features];
    next[i] = v;
    setFeatures(next);
    if (activeSample) {
      setActiveSample(null);
    }
  };

  const loadSample = (type) => {
    const source = (sampleData && sampleData[type]) || SAMPLE_TRANSACTIONS[type];
    if (!source || !Array.isArray(source.features)) {
      setError(`Sample '${type}' is unavailable.`);
      return;
    }

    setFeatures(source.features.map((val) => String(val)));
    setActiveSample({
      key: type,
      label: source.label || (type === "fraud" ? "Fraud Case (Class 1)" : "Legitimate Case (Class 0)"),
      groundTruth: source.ground_truth ?? (type === "fraud" ? 1 : 0),
    });
    setResult(null);
    setError("");
  };

  const clearAll = () => {
    setFeatures(Array(30).fill(""));
    setActiveSample(null);
    setResult(null);
    setError("");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setResult(null);

    const parsed = features.map((f) => parseFloat(f));
    if (parsed.some((n) => Number.isNaN(n))) {
      setError("All 30 features must be valid numbers.");
      return;
    }

    setLoading(true);
    try {
      const { data } = await api.post("/predict", { features: parsed });
      setResult(data);
      onPredicted?.();
    } catch (err) {
      setError(
        err?.response?.data?.detail ||
          err?.message ||
          "Prediction request failed."
      );
    } finally {
      setLoading(false);
    }
  };

  const isFraud = result?.fraud_prediction === 1;

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <h2 className="card-title">Fraud Prediction Demo</h2>
          <p className="card-subtitle">
            Enter all 30 transaction features or load verified test cases from the held-out Kaggle test set.
          </p>
        </div>
        <div className="card-actions">
          <button
            type="button"
            className="btn btn-ghost"
            onClick={() => loadSample("legitimate")}
          >
            Sample Legitimate
          </button>
          <button
            type="button"
            className="btn btn-ghost"
            onClick={() => loadSample("fraud")}
          >
            Sample Fraud
          </button>
          <button type="button" className="btn btn-ghost" onClick={clearAll}>
            Clear
          </button>
        </div>
      </div>

      {activeSample && (
        <div className="sample-loaded-banner">
          <span className="badge badge-info">Verified Test Case</span>
          <span>{activeSample.label}</span>
        </div>
      )}

      <form onSubmit={handleSubmit}>
        <div className="features-grid">
          {FEATURE_NAMES.map((name, i) => (
            <label key={name} className="feature-field">
              <span>{name}</span>
              <input
                type="number"
                step="any"
                value={features[i]}
                onChange={(e) => updateFeature(i, e.target.value)}
                placeholder="0.0"
              />
            </label>
          ))}
        </div>

        <div className="form-footer">
          <button type="submit" className="btn btn-primary" disabled={loading}>
            {loading ? (
              <>
                <span className="spinner" /> Predicting…
              </>
            ) : (
              "Run Prediction"
            )}
          </button>
        </div>

        {error && <div className="alert alert-error">{error}</div>}

        {result && (
          <div className={`result ${isFraud ? "result-fraud" : "result-safe"}`}>
            <div className="result-verdict">
              <span
                className={`badge ${isFraud ? "badge-danger" : "badge-success"}`}
              >
                {isFraud ? "Fraud" : "Not Fraud"}
              </span>
            </div>
            <div className="result-prob">
              <div className="result-prob-label">Fraud probability</div>
              <div className="result-prob-value">
                {(result.fraud_probability * 100).toFixed(2)}%
              </div>
              <div className="prob-bar">
                <div
                  className={`prob-bar-fill ${
                    isFraud ? "fill-danger" : "fill-success"
                  }`}
                  style={{ width: `${Math.min(100, Math.max(0, result.fraud_probability * 100))}%` }}
                />
              </div>

              {activeSample && (
                <div className="ground-truth-match">
                  <span>
                    <strong>Ground Truth:</strong> Class {activeSample.groundTruth} (
                    {activeSample.groundTruth === 1 ? "Fraud" : "Legitimate"})
                  </span>
                  <span>•</span>
                  <span>
                    <strong>Model Match:</strong>{" "}
                    {(isFraud ? 1 : 0) === activeSample.groundTruth
                      ? "✅ Correct"
                      : "❌ Discrepancy"}
                  </span>
                </div>
              )}
            </div>
            <TopContributors
              topContributors={result?.top_contributors || []}
            />
          </div>
        )}
      </form>
    </div>
  );
}

function TransactionsTable({ rows, loading }) {
  const fmtTimestamp = (ts) => {
    if (!ts) return "—";
    const d = new Date(ts);
    return isNaN(d.getTime()) ? String(ts) : d.toLocaleString();
  };

  return (
    <div className="card">
      <div className="card-header">
        <div>
          <h2 className="card-title">Prediction History</h2>
          <p className="card-subtitle">
            Recent transactions scored by the model.
          </p>
        </div>
      </div>

      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Fraud Prediction</th>
              <th>Fraud Probability</th>
              <th>Timestamp</th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr>
                <td colSpan={4} className="table-empty">
                  <span className="spinner" /> Loading…
                </td>
              </tr>
            )}
            {!loading && rows.length === 0 && (
              <tr>
                <td colSpan={4} className="table-empty">
                  No predictions yet.
                </td>
              </tr>
            )}
            {!loading &&
              rows.map((r, idx) => {
                const isFraud = r.fraud_prediction === 1;
                const prob = r.fraud_probability;
                return (
                  <tr key={r.id != null ? `tx-${r.id}` : `tx-idx-${idx}`}>
                    <td className="muted">{r.id ?? idx + 1}</td>
                    <td>
                      <span
                        className={`badge ${
                          isFraud ? "badge-danger" : "badge-success"
                        }`}
                      >
                        {isFraud ? "Fraud" : "Not Fraud"}
                      </span>
                    </td>
                    <td>
                      {typeof prob === "number"
                        ? `${(prob * 100).toFixed(2)}%`
                        : "—"}
                    </td>
                    <td>{fmtTimestamp(r.timestamp)}</td>
                  </tr>
                );
              })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function App() {
  const [metrics, setMetrics] = useState(null);
  const [samples, setSamples] = useState(SAMPLE_TRANSACTIONS);
  const [transactions, setTransactions] = useState([]);
  const [loadingTx, setLoadingTx] = useState(true);
  const [loadError, setLoadError] = useState("");

  const load = async () => {
    setLoadingTx(true);
    setLoadError("");
    try {
      const [m, t, s] = await Promise.all([
        api.get("/metrics"),
        api.get("/transactions"),
        api.get("/samples").catch(() => ({ data: SAMPLE_TRANSACTIONS })),
      ]);
      setMetrics(m.data || {});

      const rawList = Array.isArray(t.data) ? t.data : t.data?.transactions || [];
      const seen = new Set();
      const deduped = [];
      for (const item of rawList) {
        const key = item.id != null ? item.id : `${item.timestamp}-${item.fraud_probability}`;
        if (!seen.has(key)) {
          seen.add(key);
          deduped.push(item);
        }
      }
      setTransactions(deduped);

      if (s?.data && Object.keys(s.data).length > 0) {
        setSamples(s.data);
      }
    } catch (err) {
      setLoadError(
        err?.response?.data?.detail || err?.message || "Failed to load data."
      );
    } finally {
      setLoadingTx(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <div className="brand-mark">FD</div>
          <div>
            <h1 className="brand-title">Fraud Detection</h1>
            <p className="brand-sub">Real-time transaction risk monitoring</p>
          </div>
        </div>
        <button className="btn btn-ghost" onClick={load}>
          Refresh
        </button>
      </header>

      <main className="app-main">
        {loadError && <div className="alert alert-error">{loadError}</div>}

        <section className="metrics-grid">
          {METRIC_KEYS.map(({ key, label }) => (
            <MetricCard key={key} label={label} value={metrics?.[key]} />
          ))}
        </section>
        <ModelInfoCard />
        <AnalyticsCharts
          metrics={metrics}
          transactions={transactions}
        />

        <section>
          <PredictionForm onPredicted={load} sampleData={samples} />
        </section>

        <section>
          <TransactionsTable rows={transactions} loading={loadingTx} />
        </section>
      </main>

      <footer className="app-footer">
        <span>© {new Date().getFullYear()} Fraud Detection</span>
        <span className="muted">Powered by ML risk scoring</span>
      </footer>
    </div>
  );
}
