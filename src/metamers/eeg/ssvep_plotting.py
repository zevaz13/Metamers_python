import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from metamers.eeg.ssvep2python import load_ssvep_npz

npz_root = "/home/zevaz/projects/metamers/eeg/ssveps"

def load_metadata(npz_root):
    csv_path = os.path.join(npz_root, "ssvep_metadata.csv")
    return pd.read_csv(csv_path)

def get_participant_mean_map(data, normalize=False, runs=None):
    """
    Returns a 10x10 map for a participant:
    - mean across runs
    - optionally baseline-normalized
    """
    if normalize:
        norm_maps = baseline_normalize_maps(data, runs)  # 10x10xR
        return np.mean(norm_maps, axis=2)                # 10x10
    else:
        run_map = data["run_map"]                        # 10x10xR
        return np.mean(run_map, axis=2)                  # 10x10

def compute_group_mean_map(df, label, session=None, normalize=False, runs=None):
    """
    Compute the mean map across all participants in a group or subgroup.
    label: group or subgroup name
    """
    subset = select_participants(df, label, session=session)

    if len(subset) == 0:
        raise ValueError(f"No participants found for '{label}' session={session}")

    maps = []

    for _, row in subset.iterrows():
        data = load_ssvep_npz(row["npz_path"])
        m = get_participant_mean_map(data, normalize=normalize, runs=runs)
        maps.append(m)

    group_mean = np.mean(np.stack(maps), axis=0)

    # Return example participant for axis scaling
    example = load_ssvep_npz(subset.iloc[0]["npz_path"])
    return group_mean, example

def select_participants(df, label, session=None):
    """
    Select participants by group OR subgroup.
    label: e.g., "CTR", "PD", "protan", "deutan"
    """
    label = label.lower()

    # Normalize columns for comparison
    df2 = df.copy()
    df2["group"] = df2["group"].str.lower()
    df2["subgroup"] = df2["subgroup"].str.lower()

    # Match group or subgroup
    subset = df2[(df2["group"] == label) | (df2["subgroup"] == label)]

    if session is not None:
        subset = subset[subset["session"] == session]

    return subset


def plot_group_mean_map(df, label, session=None, normalize=False, runs=None, cmap="viridis"):
    """
    Plot the group or subgroup mean map.
    label: e.g., "CTR", "protan", "deutan"
    """
    group_mean, example = compute_group_mean_map(
        df, label, session=session, normalize=normalize, runs=runs
    )

    red = example["red_array"]
    green = example["green_array"]

    plt.figure(figsize=(6, 5))
    plt.imshow(group_mean, origin="lower", cmap=cmap,
               extent=[red.min(), red.max(), green.min(), green.max()],
               aspect="auto")
    plt.colorbar(label="Mean SSVEP amplitude")

    title = f"{label.capitalize()} group"
    if session is not None:
        title += f" – Session {session}"
    if normalize:
        title += " (baseline normalized)"

    plt.title(title)
    plt.xlabel("Red intensity")
    plt.ylabel("Green intensity")
    plt.tight_layout()
    plt.show()

def plot_group_mean_map_with_points(
    df,
    label,
    session=None,
    normalize=False,
    runs=None,
    cmap="viridis",
    scatter_df=None,
    scatter_color="#000000",
    scatter_size=20,
    vmin=None,
    vmax=None,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None
):
    """
    Plot group mean SSVEP map and optionally overlay behavioral scatter points.
    Supports manual control of color scale (vmin/vmax) and axis limits.
    """
    group_mean, example = compute_group_mean_map(
        df, label, session=session, normalize=normalize, runs=runs
    )

    red = example["red_array"]
    green = example["green_array"]

    plt.figure(figsize=(6, 5))

    # Heatmap with optional color scale control
    plt.imshow(
        group_mean,
        origin="lower",
        cmap=cmap,
        extent=[red.min(), red.max(), green.min(), green.max()],
        aspect="auto",
        vmin=vmin,
        vmax=vmax
    )
    plt.colorbar(label="Mean SSVEP amplitude")

    # Overlay scatter points
    if scatter_df is not None:
        plt.scatter(
            scatter_df["Red"],
            scatter_df["Green"],
            c=scatter_color,
            s=scatter_size,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.3
        )

    # Axis limits (optional)
    if xmin is not None or xmax is not None:
        plt.xlim(left=xmin if xmin is not None else plt.xlim()[0],
                 right=xmax if xmax is not None else plt.xlim()[1])

    if ymin is not None or ymax is not None:
        plt.ylim(bottom=ymin if ymin is not None else plt.ylim()[0],
                 top=ymax if ymax is not None else plt.ylim()[1])

    # Title
    title = f"{label.capitalize()} group"
    if session is not None:
        title += f" – Session {session}"
    if normalize:
        title += " (baseline normalized)"

    plt.title(title)
    plt.xlabel("Red intensity")
    plt.ylabel("Green intensity")
    plt.tight_layout()
    plt.show()

def get_participant_mean_map(data, normalize=False, runs=None):
    """
    Returns a 10x10 map for a participant:
    - mean across runs
    - optionally baseline-normalized
    """
    if normalize:
        norm_maps = baseline_normalize_maps(data, runs)  # 10x10xR
        return np.mean(norm_maps, axis=2)                # 10x10
    else:
        run_map = data["run_map"]                        # 10x10xR
        return np.mean(run_map, axis=2)                  # 10x10

def plot_participant_map_with_points(
    data,
    df_beh=None,
    normalize=False,
    runs=None,
    cmap="viridis",
    scatter_color="#000000",
    scatter_size=20
):
    """
    Plot a participant's mean SSVEP map (raw or normalized) and overlay behavioral points.
    
    data: dict from load_ssvep_npz()
    df_beh: behavioral dataframe filtered to this participant only
            must contain columns ["Red", "Green"]
    """
    # Compute participant mean map
    mean_map = get_participant_mean_map(data, normalize=normalize, runs=runs)

    # Axes from SSVEP stimulus grid
    red = data["red_array"]
    green = data["green_array"]

    plt.figure(figsize=(6, 5))

    # Heatmap
    plt.imshow(
        mean_map,
        origin="lower",
        cmap=cmap,
        extent=[red.min(), red.max(), green.min(), green.max()],
        aspect="auto"
    )
    plt.colorbar(label="SSVEP amplitude")

    # Overlay behavioral scatter
    if df_beh is not None and len(df_beh) > 0:
        plt.scatter(
            df_beh["Red"],
            df_beh["Green"],
            c=scatter_color,
            s=scatter_size,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.3
        )

    # Title
    title = f"{data['participant_id']} – Mean Map"
    if normalize:
        title += " (baseline normalized)"
    plt.title(title)

    plt.xlabel("Red intensity")
    plt.ylabel("Green intensity")
    plt.tight_layout()
    plt.show()

def try_get_npz_path(df, pid, session):
    subset = df[(df["participant_id"] == pid) & (df["session"] == session)]
    if len(subset) == 0:
        return None
    return subset["npz_path"].iloc[0]


def plot_single_run_map(data, run_idx=0, cmap="viridis"):
    """
    Plot a single runMap heatmap for a participant.
    data: dict loaded from load_ssvep_npz()
    """
    run_map = data["run_map"][:, :, run_idx]
    red = data["red_array"]
    green = data["green_array"]

    plt.figure(figsize=(6, 5))
    plt.imshow(run_map, origin="lower", cmap=cmap,
               extent=[red.min(), red.max(), green.min(), green.max()],
               aspect="auto")
    plt.colorbar(label="SSVEP amplitude")
    plt.title(f"{data['participant_id']} – Run {run_idx+1}")
    plt.xlabel("Red intensity")
    plt.ylabel("Green intensity")
    plt.xlim(0, 3200)
    plt.ylim(0, 2000)
    plt.tight_layout()
    plt.show()

def plot_all_runs(data, cmap="viridis"):
    run_map = data["run_map"]
    num_runs = run_map.shape[2]

    cols = min(4, num_runs)
    rows = int(np.ceil(num_runs / cols))

    red = data["red_array"]
    green = data["green_array"]

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten()

    for i in range(num_runs):
        ax = axes[i]
        ax.imshow(run_map[:, :, i], origin="lower", cmap=cmap,
                  extent=[red.min(), red.max(), green.min(), green.max()],
                  aspect="auto")
        ax.set_title(f"Run {i+1}")
        ax.set_xlabel("Red")
        ax.set_ylabel("Green")

    # Hide unused axes
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"{data['participant_id']} – All Runs")
    plt.xlim(0, 3200)
    plt.ylim(0, 2000)
    plt.tight_layout()
    plt.show()

def plot_combined_raw_map(data, cmap="viridis"):
    run_map = data["run_map"]
    combined = np.mean(run_map, axis=2)

    red = data["red_array"]
    green = data["green_array"]

    plt.figure(figsize=(6, 5))
    plt.imshow(combined, origin="lower", cmap=cmap,
               extent=[red.min(), red.max(), green.min(), green.max()],
               aspect="auto")
    plt.colorbar(label="Mean SSVEP amplitude")
    plt.title(f"{data['participant_id']} – Combined Map")
    plt.xlabel("Red intensity")
    plt.ylabel("Green intensity")
    plt.tight_layout()
    plt.show()

def baseline_normalize_maps(data, runs=None):
    """
    Returns baseline-normalized runMap as percentage change.
    runs: list of run indices to include (0-based). If None, use all.
    """
    run_map = data["run_map"]
    baselines = data["baselines"]

    num_runs = run_map.shape[2]

    if runs is None:
        runs = list(range(num_runs))

    norm_maps = []

    for r in runs:
        baseline = np.mean(baselines[0:2, r])
        norm = (run_map[:, :, r] - baseline) / baseline
        norm_maps.append(norm)

    return np.stack(norm_maps, axis=2)

def plot_normalized_maps(data, runs=None, cmap="coolwarm"):
    norm_maps = baseline_normalize_maps(data, runs)
    num_runs = norm_maps.shape[2]

    cols = min(4, num_runs)
    rows = int(np.ceil(num_runs / cols))

    red = data["red_array"]
    green = data["green_array"]

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten()

    for i in range(num_runs):
        ax = axes[i]
        ax.imshow(norm_maps[:, :, i], origin="lower", cmap=cmap,
                  extent=[red.min(), red.max(), green.min(), green.max()],
                  aspect="auto")
        ax.set_title(f"Run {i+1} (norm)")
        ax.set_xlabel("Red")
        ax.set_ylabel("Green")

    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"{data['participant_id']} – Baseline Normalized Maps")
    plt.tight_layout()
    plt.show()

def plot_mean_normalized_map(data, runs=None, cmap="coolwarm"):
    """
    Plot the mean of baseline-normalized maps across selected runs.
    """
    norm_maps = baseline_normalize_maps(data, runs)   # shape: 10 x 10 x R
    mean_map = np.mean(norm_maps, axis=2)             # shape: 10 x 10

    red = data["red_array"]
    green = data["green_array"]

    plt.figure(figsize=(6, 5))
    plt.imshow(mean_map, origin="lower", cmap=cmap,
               extent=[red.min(), red.max(), green.min(), green.max()],
               aspect="auto")
    plt.colorbar(label="Normalized amplitude (pct change)")
    plt.title(f"{data['participant_id']} – Mean Normalized Map")
    plt.xlabel("Red intensity")
    plt.ylabel("Green intensity")
    plt.tight_layout()
    plt.show()
