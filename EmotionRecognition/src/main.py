import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run_script(script_name, args=None):
    cmd = ["python3", str(ROOT / "src" / script_name)]
    if args:
        cmd.extend(args)

    print(f"\nRunning: {script_name}\n")
    subprocess.run(cmd, check=True)

    print(f"\n------------------------------------------------------")

def main():
    run_script("make_labels_csv.py")

    run_script("extract_features_csv.py")

    run_script("train.py")

    if len(sys.argv) == 2:
        wav_file = sys.argv[1]
        run_script("predict.py", [wav_file])

if __name__ == "__main__":
    main()
