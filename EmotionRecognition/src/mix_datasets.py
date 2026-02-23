import pandas as pd
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
RAVDESS_CSV = ROOT / "results" / "features.csv"
CREMAD_CSV = ROOT / "results" / "cremad_features.csv"
OUT_CSV = ROOT / "results" / "combined_training_features.csv"


def main():
    # Load the datasets
    print("Loading datasets...")
    df_ravdess = pd.read_csv(RAVDESS_CSV)
    df_cremad = pd.read_csv(CREMAD_CSV)

    print(f"Original RAVDESS shape: {df_ravdess.shape}")
    print(f"Original CREMA-D shape: {df_cremad.shape}")

    # Stratified Sampling of CREMA-D
    # Group by the 'label' column and randomly sample exactly 250 from each group
    samples_per_class = 350
    print(f"\nSampling {samples_per_class} rows per emotion from CREMA-D...")

    # Using a different random seed for reproducibility of this second model with a larger, mixed dataset
    df_cremad_sampled = df_cremad.groupby("label").sample(n=samples_per_class, random_state=17)

    print("Sampled CREMA-D distribution:")
    print(df_cremad_sampled["label"].value_counts())

    # Merge the datasets
    print("\nMerging datasets...")
    combined_df = pd.concat([df_ravdess, df_cremad_sampled], ignore_index=True)

    # Shuffle the combined dataset
    # frac=1 means return 100% of the data, but in a random order - again using 17 as seed for reproducibility
    combined_df = combined_df.sample(frac=1, random_state=17).reset_index(drop=True)

    # Save to new CSV
    combined_df.to_csv(OUT_CSV, index=False)
    print(f"\nSuccess! Saved {len(combined_df)} mixed rows to {OUT_CSV}")
    print("Final combined class distribution:")
    print(combined_df["label"].value_counts())


if __name__ == "__main__":
    main()