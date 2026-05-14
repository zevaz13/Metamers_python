# src/metamers/behavioral/loader.py
import glob
import os

def get_valid_files(base_path, folder_name):
    """
    Returns a sorted list of valid behavioral TXT files for a given session.
    Valid files contain '%VarTimPots%' in the filename.
    """
    session_path = os.path.join(base_path, folder_name)

    # Find all .txt files in the session folder
    all_txt = glob.glob(os.path.join(session_path, "*.txt"))

    # Keep only valid files
    valid = [
        f for f in all_txt
        if "VarTimPots" in os.path.basename(f)
    ]

    # Sort for deterministic run numbering
    valid.sort()

    return valid
