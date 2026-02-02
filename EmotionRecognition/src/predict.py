from __future__ import annotations
import argparse
import joblib
import numpy as np
from pathlib import Path
from extract_features_csv import extract_features

# audio processing parameters
SR = 16000
N_MFCC = 13

ROOT = Path(__file__).resolve().parents[1]

# loads model from joblib
def load_artifacts(model_path: Path, scaler_path: Path | None, le_path: Path | None):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path else None
    label_encoder = joblib.load(le_path) if le_path else None
    return model, scaler, label_encoder

# predicts emotion for each .wav & returns label & score
def predict_emotion(
    wav_file: Path,
    model,
    scaler=None,
    label_encoder=None,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    x = extract_features(wav_file).reshape(1, -1)

    if scaler is not None:
        x = scaler.transform(x)

    # use probabilities (if model supports it), else use (decision_function / hard label)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        order = np.argsort(proba)[::-1][:top_k]
        labels = model.classes_
        results = [(labels[i], float(proba[i])) for i in order]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        scores = scores[0] if scores.ndim > 1 else scores
        order = np.argsort(scores)[::-1][:top_k]
        labels = model.classes_ if hasattr(model, "classes_") else np.arange(len(scores))
        results = [(labels[i], float(scores[i])) for i in order]
    else:
        pred = model.predict(x)[0]
        results = [(pred, 1.0)]

    # decodes labels (feature category)
    if label_encoder is not None:
        results = [(label_encoder.inverse_transform([lab])[0], score) for lab, score in results]

    return results

# parses CLI args (for prediction)
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict emotion from a WAV file.")
    p.add_argument("wav", type=Path, help="Path to a .wav file")
    p.add_argument("--model", type=Path, default=Path("results/model.joblib"), help="Trained model file")
    p.add_argument("--scaler", type=Path, default=None, help="Optional scaler file (joblib)")
    p.add_argument("--label-encoder", type=Path, default=None, help="Optional label encoder file (joblib)")
    p.add_argument("--top-k", type=int, default=3, help="How many top classes to show")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    wav_path = (ROOT / args.wav).resolve()
    if not wav_path.exists():
        raise FileNotFoundError(f".wav not found: {wav_path}")
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    model, scaler, le = load_artifacts(args.model, args.scaler, args.label_encoder)
    results = predict_emotion(args.wav, model, scaler=scaler, label_encoder=le, top_k=args.top_k)

    print(f".wav: {args.wav}")
    for rank, (lab, score) in enumerate(results, start=1):
        print(f"{rank}. {lab}  ({score:.4f})")

if __name__ == "__main__":
    main()
