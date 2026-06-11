import { useState } from "react";
import { useEffect } from "react";
import axios from "axios";

function App() {

  const [features, setFeatures] = useState(Array(30).fill(0));

  const [result, setResult] = useState(null);

  const [history, setHistory] = useState([]);

  const [loading, setLoading] = useState(false);

  const handleChange = (index, value) => {

    const updated = [...features];

    updated[index] = Number(value);

    setFeatures(updated);
  };

  const handleSubmit = async () => {

    try {

      setLoading(true);

      const response = await axios.post(
        "http://127.0.0.1:8000/predict",
        {
          features: features
        }
      );

      setResult(response.data);
      fetchHistory();

    } catch (error) {

      console.error(error);

      alert("Prediction failed");

    } finally {

      setLoading(false);
    }
  };

  const fetchHistory = async () => {
  try {
    const response = await axios.get(
      "http://127.0.0.1:8000/transactions"
    );

    setHistory(response.data);
  } catch (error) {
    console.error(error);
  }
};

useEffect(() => {
  fetchHistory();
}, []);

  return (
    <div style={{ padding: "30px", fontFamily: "Arial" }}>

      <h1>Fraud Detection Dashboard</h1>

      <h3>Transaction Features</h3>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(5, 1fr)",
          gap: "10px"
        }}
      >
        {features.map((value, index) => (
          <input
            key={index}
            type="number"
            value={value}
            onChange={(e) => handleChange(index, e.target.value)}
            placeholder={`Feature ${index + 1}`}
          />
        ))}
      </div>

      <button
        onClick={handleSubmit}
        style={{
          marginTop: "20px",
          padding: "10px 20px",
          cursor: "pointer"
        }}
      >
        {loading ? "Predicting..." : "Predict Fraud"}
      </button>

      {result && (
        <div style={{ marginTop: "30px" }}>

          <h2>Prediction Result</h2>

          <p>
            Fraud Prediction:
            {" "}
            <strong>
              {result.fraud_prediction === 1 ? "Fraud" : "Not Fraud"}
            </strong>
          </p>

          <p>
            Fraud Probability:
            {" "}
            <strong>
              {(result.fraud_probability * 100).toFixed(2)}%
            </strong>
          </p>

        </div>
      )}
      <h2 style={{ marginTop: "40px" }}>
  Prediction History
</h2>

<table border="1" cellPadding="8">
  <thead>
    <tr>
      <th>ID</th>
      <th>Prediction</th>
      <th>Probability</th>
      <th>Timestamp</th>
    </tr>
  </thead>

  <tbody>
    {history.map((item) => (
      <tr key={item.id}>
        <td>{item.id}</td>

        <td>
          {item.fraud_prediction === 1
            ? "Fraud"
            : "Not Fraud"}
        </td>

        <td>
          {(item.fraud_probability * 100).toFixed(2)}%
        </td>

        <td>{item.timestamp}</td>
      </tr>
    ))}
  </tbody>
</table>
    </div>
  );
}

export default App;