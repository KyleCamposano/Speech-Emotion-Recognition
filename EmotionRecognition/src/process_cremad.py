import csv
import numpy as np
import librosa
from pathlib import Path

# project files path
ROOT = Path(__file__).resolve().parents[1]

# TODO: Adjust this path to where your CREMA-D AudioWAV folder is located
CREMA_DIR = ROOT / "data" / "AudioWAV"
OUT_CSV = ROOT / "results" / "cremad_features.csv"

# audio processing parameters (Exactly matching your model)
SR = 16000
N_MFCC = 13

# maps CREMA-D emotion codes to your target labels
EMO_MAP = {
    "NEU": "neutral",
    "HAP": "happy",
    "SAD": "sad",
    "ANG": "angry"
}


def extract_features(wav_path: str) -> np.ndarray:
    # Identical acoustic feature vector extraction
    y, sr = librosa.load(wav_path, sr=SR, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    feats = np.concatenate([
        mfcc_mean,
        mfcc_std,
        [np.mean(rms), np.std(rms), np.mean(zcr), np.std(zcr)]
    ])

    return feats


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # builds CSV header for feature columns
    header = [f"mfcc{i}_mean" for i in range(1, N_MFCC + 1)] + \
             [f"mfcc{i}_std" for i in range(1, N_MFCC + 1)] + \
             ["rms_mean", "rms_std", "zcr_mean", "zcr_std", "label", "filepath"]

    rows_written = 0

    with open(OUT_CSV, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)

        # traverse thru CREMA-D audio paths
        for wav in CREMA_DIR.rglob("*.wav"):
            # Example filename: 1001_DFA_ANG_XX.wav
            parts = wav.stem.split("_")

            if len(parts) >= 3:
                crema_code = parts[2]
                emotion = EMO_MAP.get(crema_code, "unknown")

                # Only extract if it's one of your 4 target classes
                if emotion in EMO_MAP.values():
                    try:
                        feats = extract_features(str(wav))
                        writer.writerow(list(feats) + [emotion, str(wav.relative_to(ROOT))])
                        rows_written += 1

                        # progress update
                        if rows_written % 50 == 0:
                            print(f"Processed {rows_written} files...")

                    except Exception as e:
                        print(f"Skipping {wav} due to error: {e}")

    print(f"Done. Wrote {rows_written} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()