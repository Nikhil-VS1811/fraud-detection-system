import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from preprocess import (
    prepare_data,
    split_data,
    fit_preprocessor,
    transform_data,
)


def apply_smote(X_train_scaled, y_train, random_state: int = 42):
    """
    Applies SMOTE exclusively to the training set.
    Never resamples test or validation observations.
    """
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    return X_train_resampled, y_train_resampled


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    data_path = repo_root / "ml" / "data" / "creditcard.csv"

    if not data_path.exists():
        print("=" * 60)
        print("NOTICE: Real dataset not found!")
        print(f"Path checked: {data_path}")
        print("Retraining and evaluation remain blocked until 'creditcard.csv' is supplied.")
        print("To run training, place the Kaggle dataset in 'ml/data/creditcard.csv'.")
        print("=" * 60)
        return 0

    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # 1. Prepare raw features and target
    X, y = prepare_data(df)

    # 2. Split raw unscaled data first (stratified)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    print(f"Dataset split: Train={len(X_train)} samples, Test={len(X_test)} samples")
    print(f"Test set fraud prevalence: {y_test.sum()}/{len(y_test)} ({y_test.mean():.4%})")

    # 3. Fit preprocessing exclusively on X_train
    preprocessor = fit_preprocessor(X_train)

    # 4. Transform training and test sets
    X_train_scaled, X_test_scaled = transform_data(preprocessor, X_train, X_test)

    # 5. Apply SMOTE only to training set
    print(f"\nTraining set class balance before SMOTE:\n{y_train.value_counts()}")
    X_train_smote, y_train_smote = apply_smote(X_train_scaled, y_train, random_state=42)
    print(f"\nTraining set class balance after SMOTE:\n{y_train_smote.value_counts()}")

    # 6. Benchmark models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }

    for name, model in models.items():
        print(f"\n{'=' * 20} {name} {'=' * 20}")
        model.fit(X_train_smote, y_train_smote)

        # Evaluate on untouched, original-distribution test set
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        cm = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)

        print("\nConfusion Matrix (TN, FP / FN, TP):")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, digits=4, zero_division=0))
        print(f"ROC-AUC Score: {roc_auc:.4f} ({roc_auc * 100:.2f}%)")
        print(f"PR-AUC (Average Precision): {pr_auc:.4f} ({pr_auc * 100:.2f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())