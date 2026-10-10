# Real-Time Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688)
![React](https://img.shields.io/badge/React-18.3-61DAFB)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)
![Render](https://img.shields.io/badge/Render-Deployed-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

A full-stack machine learning application designed to identify fraudulent transactions in real time. The system combines an XGBoost classifier trained with SMOTE oversampling, a FastAPI inference engine, a React monitoring dashboard with Recharts visualizations, SHAP-based feature attribution, and a PostgreSQL database for transaction audit logging.

---

## Live Deployments

| Component | Platform | URL |
| :--- | :--- | :--- |
| **Frontend Dashboard** | Render (Static Site) | [https://fraud-frontend-vxzh.onrender.com](https://fraud-frontend-vxzh.onrender.com) |
| **Backend API** | Render (Web Service) | [https://fraud-backend-km6z.onrender.com](https://fraud-backend-km6z.onrender.com) |
| **API Documentation** | FastAPI Swagger UI | [https://fraud-backend-km6z.onrender.com/docs](https://fraud-backend-km6z.onrender.com/docs) |
| **Cloud Database** | Neon PostgreSQL | Hosted cloud instance |

---

## System Architecture

```mermaid
flowchart TD
    subgraph Client ["Frontend: React 18 and Vite"]
        UI["Analytics and Risk Dashboard"]
        Form["Transaction Input Form"]
        SampleButtons["Sample Test Cases: Legitimate and Fraud"]
    end

    subgraph Backend ["Backend API: FastAPI"]
        API["FastAPI REST Endpoints"]
        Validator["Validation: 30 Finite Features"]
        Pipeline["Canonical Pipeline: StandardScaler and XGBoost"]
        Explainer["SHAP TreeExplainer and Attribution Fallback"]
    end

    subgraph Storage ["Persistence and Artifacts"]
        NeonDB[("Neon PostgreSQL Database")]
        Models["Serialized Models in backend/models/"]
    end

    Form -->|"POST /predict"| API
    SampleButtons -->|"GET /samples or Bundled"| Form
    UI -->|"GET /metrics and /model-info"| API
    UI -->|"GET /transactions"| API

    API --> Validator
    Validator --> Pipeline
    Pipeline --> Models
    Pipeline --> Explainer
    Explainer --> API
    API -->|"Non-blocking write"| NeonDB
    API -->|"JSON: Risk, Probability, SHAP"| UI
```

---

## Machine Learning Methodology

### 1. Dataset & Class Imbalance
The model is trained on the Kaggle Credit Card Fraud Detection benchmark dataset (`creditcard.csv`):
- **Total Transactions:** 284,807
- **Fraudulent Transactions (Class 1):** 492 ($0.1727\%$ prevalence)
- **Legitimate Transactions (Class 0):** 284,315 ($99.8273\%$ prevalence)
- **Input Features (30 total):**
  - `Time`: Elapsed seconds between the transaction and the first transaction in the dataset.
  - `V1`–`V28`: Principal components obtained via PCA transformation (anonymized for confidentiality).
  - `Amount`: Transaction monetary value.

### 2. Leak-Free Preprocessing & Training Flow

To prevent data leakage, splitting occurs on raw features before any statistics are computed:

```mermaid
flowchart LR
    A["Raw Data: 284,807 rows"] --> B["Stratified Split: 80/20, seed 42"]
    B --> C["Train Set: 227,845 rows, 394 frauds"]
    B --> D["Held-Out Test Set: 56,962 rows, 98 frauds"]

    C --> E["Fit Preprocessor on Train: Scale Time and Amount, Passthrough V1-V28"]
    E --> F["Transform Train Data"]
    F --> G["Apply SMOTE to Train: Balance 50/50, seed 42"]
    G --> H["Train XGBoost: 100 estimators, logloss"]

    E -.->|"Apply Fitted Preprocessor"| I["Transform Held-Out Test Data"]
    H --> J["Evaluate on Untouched Test Set"]
    I --> J
    J --> K["Export Canonical Pipeline Artifact"]
```

1. **Stratified Splitting:** The raw dataset is partitioned into an $80\%$ training split ($227,845$ transactions) and a $20\%$ held-out test split ($56,962$ transactions) using `stratify=y` with `random_state=42`.
2. **Train-Only Scaling:** A scikit-learn `ColumnTransformer` fits separate `StandardScaler` instances exclusively on $X_{train}$ for `Time` (column index 0) and `Amount` (column index 29). Features `V1` through `V28` (indices 1–28) are passed through unmodified. Test set statistics are never observed during fitting.
3. **Train-Only SMOTE:** Synthetic Minority Over-sampling Technique (SMOTE) is applied exclusively to the preprocessed training set, balancing the minority fraud class to 50/50 ($227,451$ legitimate vs. $227,451$ synthetic fraud samples). The held-out test set remains strictly un-augmented to preserve natural class prevalence ($0.17\%$).
4. **Model Architecture:** An `XGBClassifier` configured with:
   - `eval_metric="logloss"`
   - `n_estimators=100`
   - `random_state=42`
   - `n_jobs=-1`
5. **Canonical Pipeline Packaging:** The fitted preprocessor and model are bound into an `imblearn.pipeline.Pipeline` (`fraud_detection_pipeline.pkl`). Inference accepts raw unscaled 30-feature DataFrames and transforms them internally exactly once.

---

## Evaluation & Benchmark Metrics

Metrics were calculated strictly on the held-out test set ($N = 56,962$ transactions, containing $98$ verified fraud cases) using a classification decision threshold of $0.5$:

| Metric | Score | Definition & Relevance |
| :--- | :---: | :--- |
| **ROC-AUC** | **97.92%** | Area under ROC curve; measures global discrimination capability across all thresholds. |
| **PR-AUC (Average Precision)** | **87.74%** | Area under Precision-Recall curve; prioritized metric for extreme class imbalance. |
| **Precision (Class 1)** | **73.11%** | $\frac{TP}{TP + FP} = \frac{87}{87 + 32}$; fraction of flagged alerts that were genuine fraud. |
| **Recall (Class 1)** | **88.78%** | $\frac{TP}{TP + FN} = \frac{87}{87 + 11}$; fraction of actual fraud attacks successfully intercepted. |
| **F1-Score (Class 1)** | **80.18%** | Harmonic mean of precision and recall for the fraudulent class. |

### Held-Out Confusion Matrix

$$\begin{array}{c|cc}
& \textbf{Predicted Legitimate (0)} & \textbf{Predicted Fraud (1)} \\
\hline
\textbf{Actual Legitimate (0)} & 56,832 \text{ (TN)} & 32 \text{ (FP)} \\
\textbf{Actual Fraud (1)} & 11 \text{ (FN)} & 87 \text{ (TP)} \\
\end{array}$$

> [!NOTE]
> In highly imbalanced fraud detection ($0.17\%$ fraud rate), overall accuracy ($\approx 99.92\%$) is uninformative due to the large negative majority. PR-AUC ($87.74\%$), Recall ($88.78\%$), and Precision ($73.11\%$) provide a reliable assessment of true positive capture versus false-alarm rates.

---

## Explainability (SHAP) & Limitations

The API exposes feature attributions alongside each prediction:
- **Mechanism:** Initialized via `shap.TreeExplainer` on the underlying XGBoost tree model. The top 5 features with the highest absolute attribution impact are returned with directional sign (+ pushing towards fraud, - pushing towards legitimate).
- **Fallback Guard:** If native SHAP C-bindings (`llvmlite.dylib` / shared objects) are unavailable in the host environment, the system automatically falls back to model feature importances scaled by input feature value.
- **Limitations:**
  - `V1`–`V28` are anonymized PCA transformations, meaning explanations indicate which mathematical components drove the prediction, but cannot map back directly to human merchant categories or geographical markers without the original PCA eigenvectors.
  - SHAP TreeExplainer values reflect local feature contributions for the individual transaction rather than causal explanations.

---

## Application Features

- **Real-Time Risk Scoring:** Submits 30 transaction features to the `/predict` endpoint and receives the fraud verdict ($0$ or $1$), fraud probability ($0.0\%$ to $100.0\%$), and top 5 contributing features.
- **Verified Sample Demo Buttons:**
  - **Sample Legitimate:** Loads a verified Class 0 test case ($Time=160760.0, Amount=\$23.00, \dots$) producing $0.00\%$ risk ($1.7 \times 10^{-6}$).
  - **Sample Fraud:** Loads a verified Class 1 test case ($Time=57007.0, Amount=\$0.01, V_{14}=-6.17, V_{17}=-6.54, \dots$) producing $99.99\%$ risk ($0.99998$).
  - **Ground-Truth Comparison Banner:** Displays whether the model output matches the verified test sample label.
- **Input Validation:** Enforces strict numeric validation; rejects requests with missing features, NaN, or infinite values with HTTP 400.
- **Analytics Dashboard:** Visualizes model benchmark metrics and prediction distributions using Recharts.
- **Transaction History Audit Log:** Persists prediction logs (verdict, probability, timestamp) to PostgreSQL with deduplicated client-side rendering.
- **Resilient Persistence:** Database writes in `/predict` are isolated in guarded blocks so database cold-starts or Neon connection limits do not fail inference.

---

## Visual Walkthrough

### Analytics Dashboard
Interactive visualization of model metrics and aggregate prediction distributions.
![Analytics Dashboard](screenshots/analytics-dashboard.png)

### Model Overview & Metrics
Summary cards displaying dataset statistics, SMOTE configuration, and benchmark performance.
![Dashboard](screenshots/dashboard.png)

### Real-Time Prediction Form
Interface for submitting 30-feature vectors with instant scoring.
![Prediction](screenshots/prediction.png)

### SHAP Feature Impact
Top contributors ranking feature influence on the fraud risk score.
![SHAP](screenshots/shap.png)

### Historical Transaction Log
Audit log of transactions scored by the API stored in PostgreSQL.
![History](screenshots/history.png)

---

## API Reference

The FastAPI backend exposes the following REST endpoints:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | API status check (`{"message": "Fraud Detection API is running"}`) |
| `GET` | `/health` | Service health probe (`{"status": "healthy"}`) |
| `GET` | `/metrics` | Returns held-out evaluation metrics (Precision, Recall, F1, ROC-AUC, PR-AUC) |
| `GET` | `/model-info` | Returns model architecture, dataset dimensions, and SMOTE status |
| `GET` | `/samples` | Returns verified ground-truth test cases (`legitimate` and `fraud`) for demo testing |
| `GET` | `/transactions` | Retrieves historical prediction logs from PostgreSQL |
| `POST` | `/predict` | Evaluates a 30-feature transaction vector and returns prediction, probability, and SHAP impacts |

### Sample Prediction Request

```bash
curl -X POST https://fraud-backend-km6z.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      57007.0, -1.2712, 2.4627, -2.8514, 2.3245, -1.3722, -0.9482, -3.0652,
      1.1669, -2.2688, -4.8811, 2.2551, -4.6864, 0.6524, -6.1743, 0.5944,
      -4.8497, -6.5365, -3.1191, 1.7155, 0.5605, 0.6529, 0.0819, -0.2213,
      -0.5236, 0.2242, 0.7563, 0.6328, 0.2502, 0.01
    ]
  }'
```

### Sample Prediction Response

```json
{
  "fraud_prediction": 1,
  "fraud_probability": 0.999983549118042,
  "top_contributors": [
    {
      "feature": "V14",
      "impact": 6.3209,
      "shap_value": 6.3209
    },
    {
      "feature": "V17",
      "impact": 2.0994,
      "shap_value": 2.0994
    },
    {
      "feature": "V7",
      "impact": 1.7744,
      "shap_value": 1.7744
    },
    {
      "feature": "V4",
      "impact": 1.3272,
      "shap_value": 1.3272
    },
    {
      "feature": "V12",
      "impact": 1.2692,
      "shap_value": 1.2692
    }
  ]
}
```

---

## Repository Structure

```text
fraud-detection-system/
├── backend/
│   ├── Dockerfile                  # Python 3.11 container definition
│   ├── database.py                 # SQLAlchemy engine and session management
│   ├── main.py                     # FastAPI application routes and inference logic
│   ├── models_db.py                # PredictionLog SQLAlchemy table model
│   ├── requirements.txt            # Pinned backend dependencies
│   └── models/
│       ├── fraud_detection_pipeline.pkl  # Canonical fitted imblearn Pipeline
│       ├── fraud_detection_model.pkl     # Legacy standalone model artifact
│       ├── preprocessor.pkl              # Fitted ColumnTransformer scaler
│       ├── metrics.json                  # Verified held-out evaluation metrics
│       └── sample_transactions.json      # Ground-truth test cases for demo
├── frontend/
│   ├── Dockerfile                  # Node 20 container definition
│   ├── package.json                # React 18, Vite, and Recharts dependencies
│   ├── vite.config.js              # Vite configuration
│   └── src/
│       ├── App.jsx                 # Core dashboard, form, and history component
│       ├── App.css                 # Dark-theme application styles
│       ├── sampleData.js           # Client-side fallback ground-truth test samples
│       └── components/
│           ├── AnalyticsCharts.jsx # Recharts metric bars and prediction pie chart
│           ├── ModelInfoCard.jsx   # Training metadata and model specification card
│           └── TopContributors.jsx # SHAP feature importance breakdown list
├── ml/
│   ├── data/                       # Dataset directory (creditcard.csv, gitignored)
│   ├── models/                     # ML training output artifacts
│   ├── notebooks/                  # Exploratory data analysis notebooks
│   ├── src/
│   │   ├── preprocess.py           # Preprocessing and stratified split functions
│   │   ├── train.py                # Model training and SMOTE balancing script
│   │   └── evaluate.py             # Pipeline fit, evaluation, and artifact exporter
│   └── tests/
│       └── test_pipeline.py        # 15 unit tests covering pipeline and isolation
├── screenshots/                    # Application UI screenshots
├── docker-compose.yaml             # Multi-service stack (PostgreSQL, backend, frontend)
├── LICENSE                         # MIT License
└── README.md                       # Documentation
```

---

## Local Setup & Installation

### Option 1: Docker Compose (Recommended)

Requires Docker and Docker Compose:

```bash
# Clone the repository
git clone https://github.com/Nikhil-VS1811/fraud-detection-system.git
cd fraud-detection-system

# Build and start all services (PostgreSQL, FastAPI Backend, React Frontend)
docker compose up --build
```

- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`
- PostgreSQL: `localhost:5432`

---

### Option 2: Manual Local Setup

#### Prerequisites
- Python 3.11+
- Node.js 18+ and npm
- PostgreSQL running locally or a Neon connection string

#### 1. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

# Configure database URL (defaults to local PostgreSQL)
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/fraud_detection_db"

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. Frontend Setup
```bash
cd frontend
npm install

# Optional: set backend API URL (defaults to Render production backend)
export VITE_API_URL="http://localhost:8000"

npm run dev
```

---

## Running the Automated Test Suite

The repository contains unit tests validating the leak-free split, preprocessor isolation, SMOTE training boundary, metric reconciliation, inference output bounds, and API contracts.

### ML & Backend Tests (15 Tests)
```bash
# From repository root using the backend virtual environment:
./backend/venv/bin/python -m unittest ml/tests/test_pipeline.py
```
Expected output:
```text
Ran 15 tests in 0.409s

OK
```

### Frontend Production Build Test
```bash
cd frontend
npm run build
```
Expected output:
```text
vite v5.4.21 building for production...
✓ built in ~1s
```

---

## Deployment Configuration

- **Backend (Render Web Service):**
  - Build environment: Docker (`backend/Dockerfile`) running `python:3.11-slim`
  - Start command: `uvicorn main:app --host 0.0.0.0 --port 8000`
  - Environment variables: `DATABASE_URL` configured with Neon PostgreSQL connection string.
- **Frontend (Render Static Site):**
  - Build command: `npm run build`
  - Publish directory: `dist`
  - Environment variable: `VITE_API_URL=https://fraud-backend-km6z.onrender.com`

---

## Known Limitations & Security Considerations

1. **Portfolio Demo Mode:** The public endpoints do not require authentication (JWT/OAuth2) to allow seamless portfolio evaluation by recruiters. In an enterprise financial deployment, endpoints must be secured with token authentication, API rate limiting, and mTLS.
2. **Anonymized Feature Semantics:** Because features `V1`–`V28` were generated through PCA on sensitive customer data, individual features represent statistical components rather than human-readable transaction properties (e.g., merchant ID, IP location).
3. **Batch/Stream Architecture:** The system currently scores transactions via synchronous HTTP requests. For enterprise high-throughput banking systems, ingestion would be decoupled via streaming brokers (e.g., Apache Kafka or RabbitMQ) with event-driven scoring workers.

---

## Future Improvements

- **Authentication & Authorization:** Integration of JWT authentication and Role-Based Access Control (RBAC).
- **Streaming Pipeline:** Real-time transaction stream ingestion with Apache Kafka or Redis Streams.
- **Batch Processing:** Support for uploading CSV files for bulk offline transaction scoring.
- **Automated CI/CD:** GitHub Actions workflows for continuous integration testing and automated Render deployment.
- **Kubernetes Deployment:** Helm charts and Kubernetes manifests for container orchestration and auto-scaling.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

**Nikhil VS**
- GitHub: [@Nikhil-VS1811](https://github.com/Nikhil-VS1811)
- LinkedIn: [nikhil-vs](https://www.linkedin.com/in/nikhil-vs/)
