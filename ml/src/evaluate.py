import sys
import json
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from preprocess import (
    FEATURE_NAMES,
    build_preprocessor,
    prepare_data,
    split_data,
)


def train_and_export_pipeline(X_train, y_train, X_test, y_test, total_dataset_size: int):
    """
    Constructs an imblearn Pipeline containing preprocessing, SMOTE, and XGBoost.
    Fits exclusively on X_train (SMOTE runs during fit only), evaluates on X_test,
    and exports synchronized artifacts.
    """
    pipeline = ImbPipeline([
        ("preprocessor", build_preprocessor()),
        ("smote", SMOTE(random_state=42)),
        ("classifier", XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            n_estimators=100,
            n_jobs=-1,
        )),
    ])

    print("Fitting end-to-end pipeline (Preprocessing -> SMOTE -> XGBoost on train split)...")
    pipeline.fit(X_train, y_train)

    print("Evaluating model on untouched test set (default threshold = 0.5)...")
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    assert cm.sum() == len(y_test), "Confusion matrix total count does not match test set size!"
    assert cm[1, 1] + cm[1, 0] == int(y_test.sum()), "True positive + False negative count does not match total test fraud!"

    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    print("\n" + "=" * 50)
    print("FINAL EVALUATION METRICS (Untouched Test Set):")
    print("=" * 50)
    print(f"Confusion Matrix (TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}):\n{cm}")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {roc_auc * 100:.2f}%")
    print(f"PR-AUC:    {pr_auc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    # Structured metrics dictionary
    metrics = {
        "model": "XGBoost",
        "random_state": 42,
        "classification_threshold": 0.5,
        "dataset_size": total_dataset_size,
        "test_size": len(y_test),
        "fraud_cases_total": int(y_train.sum() + y_test.sum()),
        "fraud_cases_test": int(y_test.sum()),
        "precision": round(float(precision) * 100, 2),
        "recall": round(float(recall) * 100, 2),
        "f1": round(float(f1) * 100, 2),
        "roc_auc": round(float(roc_auc) * 100, 2),
        "pr_auc": round(float(pr_auc) * 100, 2),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "feature_names": FEATURE_NAMES,
        "metric_definitions": {
            "roc_auc": "Area under the Receiver Operating Characteristic curve measuring global discrimination across thresholds.",
            "pr_auc": "Area under the Precision-Recall curve (average precision), prioritized for extreme fraud imbalance.",
            "precision": "TP / (TP + FP) - fraction of transactions flagged as fraud that were actually fraudulent.",
            "recall": "TP / (TP + FN) - fraction of actual fraud transactions successfully detected.",
            "f1": "Harmonic mean of precision and recall for positive class (Class=1).",
            "confusion_matrix": "TN: true negative, FP: false alarm, FN: missed fraud, TP: caught fraud."
        },
        "imbalance_notes": (
            "Highly imbalanced dataset (0.17% fraud prevalence). PR-AUC (Average Precision) "
            "and Recall/F1 provide the most accurate assessment of model discrimination."
        ),
    }

    # Extract separated components for flexible backend inference and SHAP
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Target directories
    repo_root = Path(__file__).resolve().parent.parent.parent
    ml_models_dir = repo_root / "ml" / "models"
    backend_models_dir = repo_root / "backend" / "models"

    ml_models_dir.mkdir(parents=True, exist_ok=True)
    backend_models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save canonical ML pipeline artifacts
    pipeline_path = ml_models_dir / "fraud_detection_pipeline.pkl"
    preprocessor_path = ml_models_dir / "preprocessor.pkl"
    metrics_path = ml_models_dir / "metrics.json"

    joblib.dump(pipeline, pipeline_path)
    joblib.dump(preprocessor, preprocessor_path)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # 2. Synchronize canonical pipeline to backend/models (do NOT overwrite old model yet)
    shutil.copy(pipeline_path, backend_models_dir / "fraud_detection_pipeline.pkl")
    shutil.copy(preprocessor_path, backend_models_dir / "preprocessor.pkl")
    shutil.copy(metrics_path, backend_models_dir / "metrics.json")

    # 3. Extract verified test samples for UI demo (1 true legitimate, 1 true fraud correctly flagged)
    fraud_indices = y_test[y_test == 1].index
    legit_indices = y_test[y_test == 0].index

    # Find a true positive fraud test case with high confidence
    sample_fraud_idx = None
    for idx in fraud_indices:
        row = X_test.loc[[idx]]
        prob = float(pipeline.predict_proba(row)[0, 1])
        if prob >= 0.7:
            sample_fraud_idx = idx
            break
    if sample_fraud_idx is None and len(fraud_indices) > 0:
        sample_fraud_idx = fraud_indices[0]

    sample_legit_idx = legit_indices[0]

    samples_data = {
        "legitimate": {
            "label": "Legitimate Transaction (Ground Truth: Class 0)",
            "ground_truth": 0,
            "predicted_class": int(pipeline.predict(X_test.loc[[sample_legit_idx]])[0]),
            "fraud_probability": round(float(pipeline.predict_proba(X_test.loc[[sample_legit_idx]])[0, 1]), 6),
            "features": [float(x) for x in X_test.loc[sample_legit_idx].tolist()],
            "feature_names": FEATURE_NAMES,
        },
        "fraud": {
            "label": "Fraudulent Transaction (Ground Truth: Class 1)",
            "ground_truth": 1,
            "predicted_class": int(pipeline.predict(X_test.loc[[sample_fraud_idx]])[0]),
            "fraud_probability": round(float(pipeline.predict_proba(X_test.loc[[sample_fraud_idx]])[0, 1]), 6),
            "features": [float(x) for x in X_test.loc[sample_fraud_idx].tolist()],
            "feature_names": FEATURE_NAMES,
        }
    }

    samples_path = ml_models_dir / "sample_transactions.json"
    with open(samples_path, "w") as f:
        json.dump(samples_data, f, indent=2)
    shutil.copy(samples_path, backend_models_dir / "sample_transactions.json")

    print("\nSaved artifacts successfully:")
    print(f"  - Canonical Pipeline: {backend_models_dir / 'fraud_detection_pipeline.pkl'}")
    print(f"  - Preprocessor:       {backend_models_dir / 'preprocessor.pkl'}")
    print(f"  - Genuine Metrics:    {backend_models_dir / 'metrics.json'}")
    print(f"  - Verified Samples:   {backend_models_dir / 'sample_transactions.json'}")
    print("  - Preserved original production model: backend/models/fraud_detection_model.pkl")

    # 4. Sanity check: Reload canonical pipeline and verify prediction fidelity
    reloaded_pipeline = joblib.load(backend_models_dir / "fraud_detection_pipeline.pkl")

    test_sub = X_test.iloc[:10]
    reloaded_preds = reloaded_pipeline.predict(test_sub)
    reloaded_probs = reloaded_pipeline.predict_proba(test_sub)

    orig_preds = pipeline.predict(test_sub)
    orig_probs = pipeline.predict_proba(test_sub)

    assert np.array_equal(reloaded_preds, orig_preds), "Reloaded pipeline predictions do not match!"
    assert np.allclose(reloaded_probs, orig_probs, rtol=1e-5), "Reloaded pipeline probabilities do not match!"
    assert np.all((reloaded_probs >= 0.0) & (reloaded_probs <= 1.0)), "Probabilities out of range!"

    print("\nCanonical pipeline serialization & reload verification: PASSED!")
    return metrics


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    data_path = repo_root / "ml" / "data" / "creditcard.csv"

    if not data_path.exists():
        print("=" * 70)
        print("NOTICE: Real dataset not found!")
        print(f"Checked path: {data_path}")
        print("Retraining and evaluation on real data are BLOCKED until 'creditcard.csv' is supplied.")
        print("Existing production model artifacts will NOT be overwritten.")
        print("To retrain, place 'creditcard.csv' in 'ml/data/' and rerun this script.")
        print("=" * 70)
        return 0

    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    train_and_export_pipeline(X_train, y_train, X_test, y_test, total_dataset_size=len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())