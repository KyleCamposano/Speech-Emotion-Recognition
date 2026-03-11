from pathlib import Path
import csv

# maps CREMA-D emotion codes (from filenames) to emotion labels
EMO_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# project files path
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "cremad_dataset"
OUT_CSV = ROOT / "results" / "labels.csv"

# limits the project scope to four target emotions
EMO_MAP = {
    "ANG": "angry",
    "HAP": "happy",
    "SAD": "sad",
    "NEU": "neutral"
}

# extracts emotion label from CREMA-D
def parse_emotion(fname: str) -> str:
    parts = fname.split("_")
    if len(parts) < 3:
        return "unknown"

    return EMO_MAP.get(parts[2], "unknown")


# traverse thru CREMA-D dataset, extracts emotion labels (filters), & saves results to CSV
def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    # iterates over all .wav files in the dataset
    for wav in DATA_DIR.rglob("*.wav"):
        emotion = parse_emotion(wav.name)
        if emotion != "unknown":
            rows.append([str(wav.relative_to(ROOT)), emotion])

    # writes file paths and labels to CSV
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "emotion"])
        writer.writerows(rows)

    # displays class distribution [DEBUGGING]
    counts = {}
    for _, e in rows:
        counts[e] = counts.get(e, 0) + 1
    print("Class counts:", counts)

    print(f"\nSaved {len(rows)} rows to {OUT_CSV.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
