from __future__ import annotations

import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from pathlib import Path

# TODO: Don't forget to change the seed and file paths/names if you want to go back and re-create the original model

# Used the path function here because for whatever reason the relative path wasn't expanding properly
ROOT = Path(__file__).resolve().parents[1]

# loads extracted acoustic features (from CSV) & returns X (feature matrix), Y (emotion labels)
def load_dataset(features_csv: Path):
    df = pd.read_csv(features_csv)
    X = df.drop(columns=["label", "filepath"]).to_numpy(dtype=np.float32)
    y = df["label"].astype(str).to_numpy()

    return X, y

# builds a training pipeline using Support Vector Machines (SVM) for MFCC (emotion recognition)
# includes feature normalization + classifier
def build_model() -> Pipeline:
    return Pipeline([
        # std features to zero mean / unit variance
        ("scaler", StandardScaler()),

        # RBF-kernel SVM with class balancing and probability output
        ("clf", SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            class_weight="balanced"
        )),
    ])

# parse CLI args (for training)
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train emotion classifier from extracted features")
    p.add_argument(
        "--features",
        type=Path,
        default=ROOT / "results" / "combined_training_features.csv",
        help="Path to extracted feature CSV"
    )
    p.add_argument(
        "--model-out",
        type=Path,
        # Changed the name of the model output for the second model
        default=ROOT / "results" / "mixed_model.joblib",
        help="Output path for trained model"
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for testing"
    )
    # Using 17 as seed for reproducibility here to be consistent with the combined training dataset
    p.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for reproducibility"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.features.exists():
        raise FileNotFoundError(f"Missing features file: {args.features}")

    # loads features and labels
    X, y = load_dataset(args.features)

    # splits emotion class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y
    )

    # builds & trains model pipeline
    model = build_model()
    model.fit(X_train, y_train)

    # evals test set
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)
    print(f"Saved model in {args.model_out}")


if __name__ == "__main__":
    main()
