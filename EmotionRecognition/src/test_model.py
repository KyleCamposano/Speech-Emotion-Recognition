import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

# Set paths to various
ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "results" / "cremad_features.csv"

MODEL_PATH = ROOT / "results" / "mixed_model.joblib"


def main():
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Missing features CSV: {FEATURES_CSV}")

    #  Load the CREMA-D features
    print("Loading CREMA-D features...")
    df = pd.read_csv(FEATURES_CSV)

    # Separate features (X) and target labels (y)
    # Drops the text columns so only the 30 numerical features are leftover
    X_test = df.drop(columns=["label", "filepath"]).values
    y_test = df["label"]

    # 4. Load your pre-trained model
    print(f"Loading model from {MODEL_PATH}...")

    model = joblib.load(MODEL_PATH)

    # ------------------------------------------------------

    # Make predictions
    print("Running inference...")
    predictions = model.predict(X_test)

    # Grade the results
    acc = accuracy_score(y_test, predictions)
    print(f"\n--- Cross-Corpus Results (CREMA-D) ---")
    print(f"Overall Accuracy: {acc * 100:.2f}%\n")
    print("Detailed Classification Report:")
    print(classification_report(y_test, predictions, zero_division=0))


if __name__ == "__main__":
    main()