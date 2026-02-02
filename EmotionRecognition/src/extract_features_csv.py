import csv
import numpy as np
import librosa
from pathlib import Path

# project files path
ROOT = Path(__file__).resolve().parents[1]
LABELS_CSV = ROOT / "results" / "labels.csv"
OUT_CSV = ROOT / "results" / "features.csv"

# audio processing parameters
SR = 16000 # target sample rate
N_MFCC = 13 # number of MFCC (Mel-Frequency Cepstral Coefficients)

# loads audio file & extracts acoustic feature vectors
def extract_features(wav_path: str) -> np.ndarray:
    # loads audio
    y, sr = librosa.load(wav_path, sr=SR, mono=True)

    # computes MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # computes energy (RMS) and zero-crossing rate
    rms = librosa.feature.rms(y=y)[0] # Root Mean Square (how loud speech signal is over time)
    zcr = librosa.feature.zero_crossing_rate(y)[0] # Zero-Crossing Rate

    # concatenates all stats into 1 feature vector
    feats = np.concatenate([
        mfcc_mean,
        mfcc_std,
        [np.mean(rms), np.std(rms), np.mean(zcr), np.std(zcr)]
    ])

    return feats

# traverse thru audio paths, extracts features, & saves results to CSV
def main():
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Missing labels file: {LABELS_CSV}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # builds CSV header for feature columns
    header = []
    for i in range(1, N_MFCC + 1):
        header.append(f"mfcc{i}_mean")
    for i in range(1, N_MFCC + 1):
        header.append(f"mfcc{i}_std")
    header += ["rms_mean", "rms_std", "zcr_mean", "zcr_std", "label", "filepath"]

    rows_written = 0

    # reads labels & writes extracted features
    with open(LABELS_CSV, "r", newline="") as f_in, open(OUT_CSV, "w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)
        writer.writerow(header)

        for row in reader:
            wav_path = ROOT / row["filepath"]
            label = row["emotion"]

            try:
                feats = extract_features(wav_path)
                writer.writerow(list(feats) + [label, str(wav_path.relative_to(ROOT))])
                rows_written += 1

                # progress update [DEBUGGING]
                if rows_written % 50 == 0:
                    print(f"Processed {rows_written} files...")

            # skips corrupted/ unreadable files
            except Exception as e:
                print(f"Skipping {wav_path} due to error: {e}")

    print(f"Done. Wrote {rows_written} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()
