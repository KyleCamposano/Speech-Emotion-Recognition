from pathlib import Path
import csv

# maps RAVDESS emotion codes (from filenames) to emotion labels
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
RAVDESS_DIR = ROOT / "data" / "ravdess_dataset" / "Audio_Speech_Actors_01-24"
OUT_CSV = ROOT / "results" / "labels.csv"

# limits the project scope to four target emotions
KEEP = {"neutral", "happy", "sad", "angry"}

# extracts emotion label from RAVDESS
def parse_emotion(fname: str) -> str:
    parts = Path(fname).stem.split("-")
    if len(parts) < 3:
        return "unknown"
    return EMO_MAP.get(parts[2], "unknown")

# traverse thru RAVDESS dataset, extracts emotion labels (filters), & saves results to CSV
def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    # iterates over all .wav files in the dataset
    for wav in RAVDESS_DIR.rglob("*.wav"):
        emotion = parse_emotion(wav.name)
        if emotion in KEEP:
            rows.append([str(wav.relative_to(ROOT)), emotion])

    # writes file paths and labels to CSV
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "emotion"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {OUT_CSV}")

    # displays class distribution [DEBUGGING]
    counts = {}
    for _, e in rows:
        counts[e] = counts.get(e, 0) + 1
    print("Class counts:", counts)

if __name__ == "__main__":
    main()
