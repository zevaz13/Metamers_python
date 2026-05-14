import pandas as pd

def parse_behavior_file(filepath):
    """
    Reads a behavioral TXT file and extracts the 3rd row (index 2)
    returning only red and green values.
    """
    # Read space-separated file
    df = pd.read_csv(filepath, sep=r"\s+", engine="python")

    # Extract row 3 (index 2)
    row = df.iloc[2]

    return {
        "Red": float(row["red"]),
        "Green": float(row["green"])
    }
