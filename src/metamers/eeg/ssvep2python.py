import numpy as np
import os
import pandas as pd
from scipy.io import loadmat


# ---------------------------------------------------------------------
# Directories (your structure preserved)
# ---------------------------------------------------------------------
mat_root = "/mnt/c/Users/zevaz/OneDrive/Escritorio/Metamers/eegExp/results/shareDarren"
npz_root = "/home/zevaz/projects/metamers/eeg/ssveps"

# ---------------------------------------------------------------------
# Convert a single .mat file → .npz
# ---------------------------------------------------------------------
def convert_mat_to_npz(mat_path, out_path):
    """Convert a MATLAB SSVEP file into a Python-friendly NPZ archive."""
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    # Try struct-style first (e.g., MET039b)
    struct_keys = [k for k in mat.keys() if k.startswith("MET")]

    if len(struct_keys) == 1:
        # MATLAB struct case
        m = mat[struct_keys[0]]

    else:
        # Variable-style case: fields saved individually
        expected = [
            "SubID", "session", "group", "subgroup",
            "baselines", "runMap", "greenArray", "redArray",
            "baseDIM", "mapDIM"
        ]

        # Check all required fields exist
        missing = [k for k in expected if k not in mat]
        if missing:
            raise ValueError(
                f"Missing fields in {mat_path}: {missing}\n"
                f"Available keys: {list(mat.keys())}"
            )

        # Build a simple object with attributes
        class Obj: pass
        m = Obj()
        for k in expected:
            setattr(m, k, mat[k])

    # Save to NPZ
    np.savez(
        out_path,
        participant_id=m.SubID,
        session=int(m.session),
        group=m.group,
        subgroup=m.subgroup,
        baselines=m.baselines,
        run_map=m.runMap,
        green_array=m.greenArray,
        red_array=m.redArray,
        base_dim=m.baseDIM,
        map_dim=m.mapDIM,
    )
# ---------------------------------------------------------------------
# Load a .npz file into a Python dict
# ---------------------------------------------------------------------
def load_ssvep_npz(npz_path):
    """Load a converted NPZ file into a Python dict."""
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}

# --------------------------------------------------------------------
# Load a .npz file into a Python dict
# ---------------------------------------------------------------------

def update_metadata_csv(npz_root, participant_id, session, group, subgroup, npz_path):
    csv_path = os.path.join(npz_root, "ssvep_metadata.csv")

    # Load existing metadata if present
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=["participant_id", "session", "group", "subgroup", "npz_path"])

    # Check if this entry already exists
    exists = (
        (df["participant_id"] == participant_id) &
        (df["session"] == session)
    ).any()

    if exists:
        return  # do nothing

    # Append new row
    new_row = {
        "participant_id": participant_id,
        "session": session,
        "group": group,
        "subgroup": subgroup,
        "npz_path": npz_path
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_path, index=False)

# ---------------------------------------------------------------------
# Incremental update: convert only new .mat files
# ---------------------------------------------------------------------
def update_ssvep_storage(mat_root=mat_root, npz_root=npz_root):
    """Convert all .mat files in mat_root to .npz, skipping existing ones."""
    os.makedirs(npz_root, exist_ok=True)

    mat_files = sorted(f for f in os.listdir(mat_root)
                       if f.startswith("MET") and f.endswith(".mat"))

    if not mat_files:
        print("No MET*.mat files found.")
        return

    for fname in mat_files:
        mat_path = os.path.join(mat_root, fname)
        npz_path = os.path.join(npz_root, fname.replace(".mat", ".npz"))

        if os.path.exists(npz_path):
            print(f"[skip] {fname} → already converted")
            continue

        print(f"[convert] {fname} → {npz_path}")
        convert_mat_to_npz(mat_path, npz_path)
        # Load the npz to extract metadata
        data = np.load(npz_path, allow_pickle=True)

        update_metadata_csv(
            npz_root=npz_root,
            participant_id=str(data["participant_id"]),
            session=int(data["session"]),
            group=str(data["group"]),
            subgroup=str(data["subgroup"]),
            npz_path=npz_path
        )

    print("Update complete.")
