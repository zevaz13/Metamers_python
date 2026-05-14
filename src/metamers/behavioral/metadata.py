# src/metamers/behavioral/metadata.py
import pandas as pd

def load_metadata(csv_path):
    df = pd.read_csv(csv_path)

    # Compute PartType
    df["PartType"] = (
        df["HC"] * 1 +
        df["CVD"] * 2 +
        df["PD"] * 3 +
        df["other"] * 4
    )

    # Only keep rows that need processing
    df_to_process = df[df["added"] == 0].copy()

    return df, df_to_process
