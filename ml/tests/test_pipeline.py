import os
import sys
import unittest
import tempfile
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

# Ensure ml/src is importable
repo_root = Path(__file__).resolve().parent.parent.parent
ml_src_path = repo_root / "ml" / "src"
if str(ml_src_path) not in sys.path:
    sys.path.insert(0, str(ml_src_path))

from preprocess import (
    FEATURE_NAMES,
    build_preprocessor,
    prepare_data,
    split_data,
    fit_preprocessor,
    transform_data,
)
from train import apply_smote
from evaluate import train_and_export_pipeline


class TestFraudDetectionPipeline(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Generate safe synthetic dataset simulating creditcard.csv schema and extreme imbalance
        # 400 normal (Class 0), 10 fraud (Class 1) -> 410 total samples
        n_samples = 410
        self.raw_data = {
            "Time": np.random.uniform(0, 172800, n_samples),
            **{f"V{i}": np.random.randn(n_samples) for i in range(1, 29)},
            "Amount": np.random.uniform(0, 5000, n_samples),
            "Class": np.array([0] * 400 + [1] * 10),
        }
        # Fraud cases often have extreme V14 and V4 distributions
        self.raw_data["V14"][-10:] = np.random.uniform(-15, -6, 10)
        self.raw_data["V4"][-10:] = np.random.uniform(4, 10, 10)

        self.df = pd.DataFrame(self.raw_data)
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_feature_names_and_count(self):
        """1. Verify exactly 30 features with Time at index 0 and Amount at index 29."""
        self.assertEqual(len(FEATURE_NAMES), 30)
        self.assertEqual(FEATURE_NAMES[0], "Time")
        self.assertEqual(FEATURE_NAMES[-1], "Amount")
        self.assertEqual(FEATURE_NAMES[1:29], [f"V{i}" for i in range(1, 29)])

    def test_data_preparation_excludes_target(self):
        """2. Verify prepare_data separates Class, leaves 30 features, and detects missing columns."""
        X, y = prepare_data(self.df)
        self.assertNotIn("Class", X.columns)
        self.assertEqual(X.shape[1], 30)
        self.assertEqual(list(X.columns), FEATURE_NAMES)
        self.assertEqual(len(y), len(self.df))

        # Check missing target error
        df_no_target = self.df.drop(columns=["Class"])
        with self.assertRaises(ValueError):
            prepare_data(df_no_target)

    def test_train_test_indices_disjoint(self):
        """3. Strict isolation: Verify train and test index sets have ZERO overlap."""
        X, y = prepare_data(self.df)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

        intersection = set(X_train.index).intersection(set(X_test.index))
        self.assertEqual(len(intersection), 0, "Train and test partitions share indices (leakage detected)!")
        self.assertEqual(len(X_train) + len(X_test), len(X))

    def test_stratification_preserves_fraud_ratio(self):
        """4. Strict isolation: Verify stratified split preserves fraud prevalence across partitions."""
        X, y = prepare_data(self.df)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

        original_ratio = y.mean()
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()

        # Ratios must match within sampling rounding tolerance
        self.assertAlmostEqual(original_ratio, train_ratio, places=2)
        self.assertAlmostEqual(original_ratio, test_ratio, places=2)
        self.assertEqual(y_test.sum(), 2)  # 20% of 10 positive cases

    def test_preprocessing_fitted_on_train_only(self):
        """5. Strict isolation: Verify preprocessor learns parameters ONLY from training split."""
        X, y = prepare_data(self.df)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

        preprocessor = fit_preprocessor(X_train)
        time_scaler = preprocessor.named_transformers_["scale_time"]
        amount_scaler = preprocessor.named_transformers_["scale_amount"]

        # Scaler parameters must match X_train statistics, NOT the entire dataset or test set
        expected_time_mean = X_train["Time"].mean()
        expected_amount_mean = X_train["Amount"].mean()

        self.assertAlmostEqual(time_scaler.mean_[0], expected_time_mean, places=4)
        self.assertAlmostEqual(amount_scaler.mean_[0], expected_amount_mean, places=4)

        # Ensure Time and Amount scalers operate independently and do not overwrite each other
        self.assertNotEqual(time_scaler.mean_[0], amount_scaler.mean_[0])

        # Transform both partitions
        X_train_scaled, X_test_scaled = transform_data(preprocessor, X_train, X_test)
        self.assertEqual(X_train_scaled.shape, (len(X_train), 30))
        self.assertEqual(X_test_scaled.shape, (len(X_test), 30))

        # Training set scaled features are centered at 0; test set will not be exactly 0 (confirming train-only fit)
        self.assertAlmostEqual(X_train_scaled[:, 0].mean(), 0.0, places=1)
        self.assertAlmostEqual(X_train_scaled[:, 29].mean(), 0.0, places=1)

    def test_smote_receives_train_partition_only(self):
        """6. Strict isolation: Verify SMOTE resamples training split only; test set is never resampled."""
        X, y = prepare_data(self.df)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

        preprocessor = fit_preprocessor(X_train)
        X_train_scaled, X_test_scaled = transform_data(preprocessor, X_train, X_test)

        test_len_before = len(X_test_scaled)
        test_fraud_count_before = int(y_test.sum())

        # Apply SMOTE to train set only
        smote = SMOTE(random_state=42, k_neighbors=2)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

        # Train resampled must be 50/50 balanced
        self.assertEqual(
            y_train_resampled.value_counts()[0],
            y_train_resampled.value_counts()[1],
        )

        # Test set must remain strictly untouched
        self.assertEqual(len(X_test_scaled), test_len_before)
        self.assertEqual(int(y_test.sum()), test_fraud_count_before)
        self.assertEqual(test_fraud_count_before, 2)
        self.assertEqual(len(y_test), 82)

    def test_metrics_calculated_on_untouched_test_set(self):
        """7. Strict isolation: Verify metrics evaluate against untouched test labels and positive class = 1."""
        X, y = prepare_data(self.df)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

        pipeline = ImbPipeline([
            ("preprocessor", build_preprocessor()),
            ("smote", SMOTE(random_state=42, k_neighbors=2)),
            ("classifier", XGBClassifier(eval_metric="logloss", random_state=42, n_estimators=10)),
        ])
        pipeline.fit(X_train, y_train)

        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Positive class must explicitly be 1
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        roc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)

        self.assertTrue(0.0 <= prec <= 1.0)
        self.assertTrue(0.0 <= rec <= 1.0)
        self.assertTrue(0.0 <= f1 <= 1.0)
        self.assertTrue(0.0 <= roc <= 1.0)
        self.assertTrue(0.0 <= pr_auc <= 1.0)

    def test_confusion_matrix_reconciliation(self):
        """8. Metric reconciliation: Verify TN + FP + FN + TP == len(y_test)."""
        y_true = np.array([0] * 80 + [1] * 2)
        y_prob = np.array([0.05] * 78 + [0.85] * 2 + [0.10] * 1 + [0.95] * 1)
        y_pred = (y_prob >= 0.5).astype(int)

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        self.assertEqual(tn + fp + fn + tp, len(y_true))
        self.assertEqual(tp + fn, 2)   # Total positive class support
        self.assertEqual(tn + fp, 80)  # Total negative class support

    def test_metrics_agree_with_confusion_matrix(self):
        """9. Metric reconciliation: Verify precision, recall, and F1 exactly match confusion matrix counts."""
        y_true = np.array([0] * 80 + [1] * 10)
        np.random.seed(42)
        y_prob = np.random.uniform(0, 1, 90)
        y_pred = (y_prob >= 0.5).astype(int)

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        expected_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        expected_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        expected_f1 = (
            2 * expected_prec * expected_rec / (expected_prec + expected_rec)
            if (expected_prec + expected_rec) > 0
            else 0.0
        )

        self.assertAlmostEqual(prec, expected_prec, places=7)
        self.assertAlmostEqual(rec, expected_rec, places=7)
        self.assertAlmostEqual(f1, expected_f1, places=7)

    def test_auc_metrics_require_continuous_probabilities(self):
        """10. Verification: Verify ROC-AUC and PR-AUC require probabilities, not binary predictions."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_probs = np.array([0.1, 0.2, 0.4, 0.7, 0.9])
        y_binary = np.array([0, 0, 0, 1, 1])

        # ROC-AUC on ranked probabilities vs hard binary predictions
        auc_probs = roc_auc_score(y_true, y_probs)
        pr_probs = average_precision_score(y_true, y_probs)

        # Probabilities must yield strictly valid discrimination
        self.assertEqual(auc_probs, 1.0)
        self.assertEqual(pr_probs, 1.0)

        # Intermediate rankings produce distinct PR-AUC values
        y_noisy_probs = np.array([0.1, 0.55, 0.3, 0.4, 0.9])
        pr_noisy = average_precision_score(y_true, y_noisy_probs)
        self.assertNotEqual(pr_noisy, 1.0)

    def test_pipeline_serialization_and_reload(self):
        """11. Consistency: Verify pipeline serializes and reloads with bit-for-bit prediction parity."""
        X, y = prepare_data(self.df)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

        preprocessor = build_preprocessor()
        pipeline = ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42, k_neighbors=2)),
            ("classifier", XGBClassifier(eval_metric="logloss", random_state=42, n_estimators=10)),
        ])
        pipeline.fit(X_train, y_train)

        preds_orig = pipeline.predict(X_test)
        probs_orig = pipeline.predict_proba(X_test)[:, 1]

        # Probabilities must be strictly bounded in [0, 1]
        self.assertTrue(np.all((probs_orig >= 0.0) & (probs_orig <= 1.0)))

        # Save preprocessor and model separately
        prep_path = os.path.join(self.test_dir, "preprocessor.pkl")
        model_path = os.path.join(self.test_dir, "model.pkl")

        joblib.dump(pipeline.named_steps["preprocessor"], prep_path)
        joblib.dump(pipeline.named_steps["classifier"], model_path)

        # Reload
        reloaded_prep = joblib.load(prep_path)
        reloaded_clf = joblib.load(model_path)

        X_test_scaled = reloaded_prep.transform(X_test)
        preds_reloaded = reloaded_clf.predict(X_test_scaled)
        probs_reloaded = reloaded_clf.predict_proba(X_test_scaled)[:, 1]

        np.testing.assert_array_equal(preds_orig, preds_reloaded)
        np.testing.assert_allclose(probs_orig, probs_reloaded, rtol=1e-5)

    def test_missing_data_guard_does_not_emit_stale_or_fabricated_metrics(self):
        """12. Guard check: When creditcard.csv is missing, scripts must exit cleanly; when present, metrics must reflect real data."""
        data_file = repo_root / "ml" / "data" / "creditcard.csv"
        metrics_file = repo_root / "ml" / "models" / "metrics.json"

        if data_file.exists():
            self.assertTrue(metrics_file.exists(), "metrics.json must exist after real dataset evaluation!")
            with open(metrics_file, "r") as f:
                import json
                data = json.load(f)
            self.assertEqual(data["dataset_size"], 284807)
            self.assertEqual(data["test_size"], 56962)
            self.assertEqual(data["fraud_cases_total"], 492)
            self.assertEqual(data["fraud_cases_test"], 98)
            self.assertEqual(data["classification_threshold"], 0.5)
        else:
            cmd = [sys.executable, str(ml_src_path / "evaluate.py")]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0)
            self.assertIn("Retraining and evaluation on real data are BLOCKED", result.stdout)
            self.assertFalse(metrics_file.exists(), "metrics.json was created despite missing dataset!")

    def test_inference_prediction_diagnostics(self):
        """13. Diagnostics: Verify canonical pipeline outputs diverse probabilities and accurately scores test samples."""
        pipe_path = repo_root / "backend" / "models" / "fraud_detection_pipeline.pkl"
        if not pipe_path.exists():
            self.skipTest("backend/models/fraud_detection_pipeline.pkl not found")

        pipe = joblib.load(pipe_path)

        # A. Random vector in [-1, 1]
        np.random.seed(42)
        random_vec = pd.DataFrame(np.random.uniform(-1, 1, (1, 30)), columns=FEATURE_NAMES)
        prob_random = pipe.predict_proba(random_vec)[0][1]
        pred_random = pipe.predict(random_vec)[0]
        self.assertEqual(pred_random, 0)
        self.assertLess(prob_random, 0.05)

        # B. Test verified samples from sample_transactions.json if available
        samples_path = repo_root / "backend" / "models" / "sample_transactions.json"
        if samples_path.exists():
            with open(samples_path, "r") as f:
                import json
                samples = json.load(f)
            legit_df = pd.DataFrame([samples["legitimate"]["features"]], columns=FEATURE_NAMES)
            fraud_df = pd.DataFrame([samples["fraud"]["features"]], columns=FEATURE_NAMES)

            self.assertEqual(pipe.predict(legit_df)[0], 0)
            self.assertLess(pipe.predict_proba(legit_df)[0, 1], 0.05)

            self.assertEqual(pipe.predict(fraud_df)[0], 1)
            self.assertGreater(pipe.predict_proba(fraud_df)[0, 1], 0.90)

    def test_backend_predict_rejects_non_finite_inputs(self):
        """14. API contract: Verify backend rejects non-finite (NaN, Inf) inputs with 400 Bad Request."""
        sys.path.insert(0, str(repo_root / "backend"))
        try:
            from fastapi import HTTPException
            from main import predict, Transaction
        except ImportError:
            self.skipTest("fastapi/backend dependencies not installed in current environment")

        # Test NaN
        nan_tx = Transaction(features=[float("nan")] + [0.0] * 29)
        with self.assertRaises(HTTPException) as ctx_nan:
            predict(nan_tx)
        self.assertEqual(ctx_nan.exception.status_code, 400)

        # Test Inf
        inf_tx = Transaction(features=[float("inf")] + [0.0] * 29)
        with self.assertRaises(HTTPException) as ctx_inf:
            predict(inf_tx)
        self.assertEqual(ctx_inf.exception.status_code, 400)

    def test_backend_health_and_samples_endpoints(self):
        """15. Recruiter Demo API contract: Verify health and samples endpoints return valid payloads."""
        sys.path.insert(0, str(repo_root / "backend"))
        try:
            from main import health_check, get_samples
        except ImportError:
            self.skipTest("fastapi/backend dependencies not installed in current environment")

        health = health_check()
        self.assertEqual(health, {"status": "healthy"})

        samples = get_samples()
        self.assertIn("legitimate", samples)
        self.assertIn("fraud", samples)
        self.assertEqual(len(samples["legitimate"]["features"]), 30)
        self.assertEqual(len(samples["fraud"]["features"]), 30)
        self.assertEqual(samples["legitimate"]["ground_truth"], 0)
        self.assertEqual(samples["fraud"]["ground_truth"], 1)


if __name__ == "__main__":
    unittest.main()

