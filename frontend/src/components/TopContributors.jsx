import React from "react";

export default function TopContributors({ topContributors }) {
  const hasData = Array.isArray(topContributors) && topContributors.length > 0;

  return (
    <div className="top-contributors-card">
      <h3 className="top-contributors-title">Top Contributors</h3>
      {!hasData ? (
        <p className="top-contributors-empty">No contributor data available.</p>
      ) : (
        <div className="top-contributors-list">
          {topContributors.map((contributor, index) => {
            const shap = Number(contributor.shap_value);
            const isPositive = shap > 0;

            return (
              <div className="top-contributor-row" key={index}>
                <div className="contributor-feature">{contributor.feature}</div>

                <div className="contributor-metric">
                  <span className="metric-label">Impact</span>
                  <span className="metric-value">{Number(contributor.impact).toFixed(2)}</span>
                </div>

                <div className="contributor-metric">
                  <span className="metric-label">SHAP</span>
                  <span
                    className={`metric-value shap-value ${
  isPositive ? "shap-fraud" : "shap-safe"
}`}
                  >
                    {shap > 0
  ? `+${Number(shap).toFixed(2)}`
  : Number(shap).toFixed(2)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}