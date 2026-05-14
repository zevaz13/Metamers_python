import os
import pandas as pd

def load_behavioral_features():
    """
    Loads behavioral_features.parquet from metamers/behavioral/
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(module_dir, "..", "..", ".."))

    beh_dir = os.path.join(project_root, "behavioral")
    path = os.path.join(beh_dir, "behavioral_features.parquet")

    return pd.read_parquet(path)


def load_FM100_scores():
    """
    Loads FM100_scores.parquet from metamers/standardScores/
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(module_dir, "..", "..", ".."))

    fm_dir = os.path.join(project_root, "standardScores")
    path = os.path.join(fm_dir, "FM100_scores.csv")

    return pd.read_csv(path)


def combine_behavioral_FM100():
    """
    Creates a unified table linking behavioral features with FM100 scores.
    Saves to metamers/behFM100/.
    """
    beh = load_behavioral_features()
    fm = load_FM100_scores()

    # Merge on SubID (many FM100 sessions → one behavioral feature vector)
    combined = fm.merge(beh, on="SubID", how="left")

    # Build output directory
    module_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(module_dir, "..", "..", ".."))
    out_dir = os.path.join(project_root, "behFM100")
    os.makedirs(out_dir, exist_ok=True)

    # Save
    csv_path = os.path.join(out_dir, "behFM100_combined.csv")
    parquet_path = os.path.join(out_dir, "behFM100_combined.parquet")

    combined.to_csv(csv_path, index=False)
    combined.to_parquet(parquet_path, index=False)

    print("Saved combined behavioral + FM100 table to:")
    print(" -", csv_path)
    print(" -", parquet_path)

    return combined
