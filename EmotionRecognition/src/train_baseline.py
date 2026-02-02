from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# project files path
ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "results" / "features.csv"

def main():
    # loads features
    df = pd.read_csv(FEATURES_CSV)

    # splits into X (features) and y (labels)
    y = df["label"]
    X = df.drop(columns=["label", "filepath"])

    # trains/tests split (stratified keeps class ratios similar)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # pipeline: scale features into SVM classifier
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale"))
    ])

    # trains
    model.fit(X_train, y_train)

    # predicts
    y_pred = model.predict(X_test)

    # evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f}\n")

    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred), "\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
