import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
]


def build_preprocessor() -> ColumnTransformer:
    """
    Builds a ColumnTransformer that scales 'Time' (index 0) and 'Amount' (index 29)
    using independent StandardScaler instances, while passing through V1-V28 (indices 1-28).
    Preserves exact feature order: ['Time', 'V1'...'V28', 'Amount'].
    """
    return ColumnTransformer(
        transformers=[
            ("scale_time", StandardScaler(), [0]),
            ("passthrough_pca", "passthrough", list(range(1, 29))),
            ("scale_amount", StandardScaler(), [29]),
        ],
        verbose_feature_names_out=False,
    )


def prepare_data(df: pd.DataFrame):
    """
    Separates features X and target y, validating feature count and ordering.
    """
    if "Class" not in df.columns:
        raise ValueError("Target column 'Class' not found in dataset.")

    y = df["Class"].copy()
    X = df.drop(columns=["Class"]).copy()

    if X.shape[1] != 30:
        raise ValueError(f"Expected 30 features, but got {X.shape[1]}")

    if list(X.columns) != FEATURE_NAMES:
        X = X[FEATURE_NAMES]

    return X, y


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Splits RAW, unscaled data first into train and test sets using stratified sampling.
    Guarantees zero data leakage prior to preprocessing.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Sanity checks
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)
    assert len(train_indices.intersection(test_indices)) == 0, (
        "Train and test index sets overlap!"
    )
    assert len(X_train) + len(X_test) == len(X), "Sample count mismatch after split!"

    return X_train, X_test, y_train, y_test


def fit_preprocessor(X_train):
    """
    Fits preprocessor EXCLUSIVELY on training features.
    """
    preprocessor = build_preprocessor()
    X_tr_arr = X_train.to_numpy() if hasattr(X_train, "to_numpy") else X_train
    preprocessor.fit(X_tr_arr)
    return preprocessor


def transform_data(preprocessor, X_train, X_test):
    """
    Transforms X_train and X_test using the preprocessor fitted on training data.
    """
    X_tr_arr = X_train.to_numpy() if hasattr(X_train, "to_numpy") else X_train
    X_te_arr = X_test.to_numpy() if hasattr(X_test, "to_numpy") else X_test
    return preprocessor.transform(X_tr_arr), preprocessor.transform(X_te_arr)


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    data_path = repo_root / "ml" / "data" / "creditcard.csv"

    if not data_path.exists():
        print(f"Dataset not found at expected path: {data_path}")
        print("Please place 'creditcard.csv' into 'ml/data/' to run preprocessing on real data.")
        return 0

    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    print(f"Dataset shape: {df.shape}")
    print("\nClass distribution:")
    print(df["Class"].value_counts(normalize=True))

    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"\nRaw X_train shape: {X_train.shape}, Raw X_test shape: {X_test.shape}")
    print(f"Train fraud count: {y_train.sum()} ({y_train.mean():.4%})")
    print(f"Test fraud count: {y_test.sum()} ({y_test.mean():.4%})")

    # Fit preprocessor exclusively on X_train
    preprocessor = fit_preprocessor(X_train)
    X_train_scaled, X_test_scaled = transform_data(preprocessor, X_train, X_test)

    print(f"\nPreprocessed X_train shape: {X_train_scaled.shape}")
    print(f"Preprocessed X_test shape: {X_test_scaled.shape}")
    print("Preprocessing completed with zero data leakage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())