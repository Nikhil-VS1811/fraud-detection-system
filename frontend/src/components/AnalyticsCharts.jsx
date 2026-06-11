import React, { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts";

const METRIC_LABELS = {
  precision: "Precision",
  recall: "Recall",
  f1: "F1",
  roc_auc: "ROC AUC",
};

const PIE_COLORS = ["#ef4444", "#22c55e"];

export default function AnalyticsCharts({ metrics, transactions }) {
  const metricsData = useMemo(() => {
    if (!metrics) return [];
    return Object.keys(METRIC_LABELS)
      .filter((k) => metrics[k] !== undefined && metrics[k] !== null)
      .map((k) => ({ name: METRIC_LABELS[k], value: Number(metrics[k]) }));
  }, [metrics]);

  const pieData = useMemo(() => {
    if (!Array.isArray(transactions) || transactions.length === 0) return [];
    let fraud = 0;
    let legit = 0;
    transactions.forEach((t) => {
      if (Number(t.fraud_prediction) === 1) fraud += 1;
      else legit += 1;
    });
    return [
      { name: "Fraud", value: fraud },
      { name: "Legitimate", value: legit },
    ];
  }, [transactions]);

  return (
    <div className="analytics-charts">
      <div className="chart-card">
        <h3 className="chart-title">Model Metrics</h3>
        {metricsData.length === 0 ? (
          <div className="chart-empty">No metrics available</div>
        ) : (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={metricsData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#26304a" />
              <XAxis dataKey="name" stroke="#9aa3b2" />
              <YAxis domain={[0, 100]} stroke="#9aa3b2" />
              <Tooltip
                contentStyle={{
                  background: "#121829",
                  border: "1px solid #26304a",
                  color: "#e6e9ef",
                }}
                formatter={(v) => `${v}%`}
              />
              <Bar dataKey="value" fill="#3b82f6" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>

      <div className="chart-card">
        <h3 className="chart-title">Prediction Distribution</h3>
        {pieData.length === 0 ? (
          <div className="chart-empty">No transactions yet</div>
        ) : (
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie
                data={pieData}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={95}
                label
              >
                {pieData.map((_, i) => (
                  <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  background: "#121829",
                  border: "1px solid #26304a",
                  color: "#e6e9ef",
                }}
              />
              <Legend wrapperStyle={{ color: "#e6e9ef" }} />
            </PieChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
