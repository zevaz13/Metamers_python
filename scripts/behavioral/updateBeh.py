from metamers.behavioral.metadata import load_metadata
from metamers.behavioral.loader import get_valid_files
from metamers.behavioral.parser import parse_behavior_file
from metamers.behavioral.table import build_behavior_table
import os

print("updateBeh is running!")

BASE = "/mnt/c/Users/zevaz/GUI_beh_metamers/BehData"
CSV = f"{BASE}/participant_beh_recordPY.csv"

# Load metadata
df_full, df_to_process = load_metadata(CSV)

# Build behavioral table
df_behavior = build_behavior_table(
    metadata_df=df_to_process,
    base_path=BASE,
    loader=get_valid_files,
    parser=parse_behavior_file
)

# Save table in top-level behavioral folder
OUTPUT_DIR = "behavioral"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df_behavior.to_csv(f"{OUTPUT_DIR}/behavioral_table.csv", index=False)
df_behavior.to_parquet(f"{OUTPUT_DIR}/behavioral_table.parquet", index=False)

# Mark processed rows
df_full.loc[df_to_process.index, "added"] = 1
df_full.to_csv(CSV, index=False)
