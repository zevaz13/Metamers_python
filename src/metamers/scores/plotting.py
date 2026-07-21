import numpy as np
import matplotlib.pyplot as plt
from metamers.scores.FM100 import (
    load_participant_info,
    identify_participant,
    FM100_TESwhole
)

# ---------------------------------------------------------
# 1. moving average
# ---------------------------------------------------------

def movmean(x, w):
    """MATLAB-style moving mean with window w."""
    x = np.asarray(x, dtype=float)
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode='same')

# ---------------------------------------------------------
# 2. Extract linear errors
# ---------------------------------------------------------

def extract_error_profile(row, info_table, smooth=False, window=15):
    """
    Extracts:
      - SubID
      - Session
      - Group
      - Subgroup
      - raw error vector (85 values)
      - smoothed error vector (optional)

    Parameters
    ----------
    row : pandas.Series
        One row from repeatedSessionsPY.csv
    info_table : DataFrame
        Output of load_participant_info()
    smooth : bool
        Whether to apply moving average smoothing
    window : int
        Window size for smoothing

    Returns
    -------
    dict with:
        SubID, Session, Group, Subgroup,
        errors_raw (85,), errors_smooth (85 or None)
    """

    # Extract ID and caps
    raw_id = row.iloc[3]
    caps   = row.iloc[16:101].to_numpy()

    # Identify participant
    meta = identify_participant(raw_id, info_table)

    # Compute TES + error vector
    _, _, err_vals = FM100_TESwhole(caps)

    # Optional smoothing
    if smooth:
        err_smooth = movmean(err_vals, window)
    else:
        err_smooth = None

    return {
        "SubID": meta["ValidID"],
        "Session": meta["Session"],
        "Group": meta["Group"],
        "Subgroup": meta["Subgroup"],
        "errors_raw": err_vals,
        "errors_smooth": err_smooth
    }

def plot_participant_errors(FM100, info_table, subID, smooth=True, window=15):
    """
    Plot error curves for all sessions of a single participant.

    Parameters
    ----------
    FM100 : DataFrame
        Raw repeatedSessionsPY data
    info_table : DataFrame
        Participant info table
    subID : str
        Participant ID (e.g., "MET017")
    smooth : bool
        Whether to plot smoothed curve
    window : int
        Smoothing window size
    """

    # Find all rows belonging to this participant
    rows = []
    for _, row in FM100.iterrows():
        raw_id = row.iloc[3]
        meta = identify_participant(raw_id, info_table)
        if meta["ValidID"] == subID:
            rows.append((meta["Session"], row))

    if len(rows) == 0:
        print(f"No data found for participant {subID}")
        return

    # Sort by session number
    rows = sorted(rows, key=lambda x: x[0])

    plt.figure(figsize=(12, 5))

    for session, row in rows:
        profile = extract_error_profile(row, info_table, smooth=smooth, window=window)

        x = np.arange(1, 86)
        raw = profile["errors_raw"]
        sm  = profile["errors_smooth"]

        # Plot raw
        plt.plot(x, raw, alpha=0.4, label=f"Session {session} (raw)")

        # Plot smoothed
        if smooth:
            plt.plot(x, sm, linewidth=2, label=f"Session {session} (smoothed)")

    plt.title(f"FM100 Error Profile — {subID}")
    plt.xlabel("Cap Index (1–85)")
    plt.ylabel("Error Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_group_errors(
    FM100,
    info_table,
    group_name=None,
    subgroup_name=None,
    smooth=True,
    window=15,
    show_individuals=False
):
    """
    Plot group-level FM100 error curves with mean ± std shading.
    Supports filtering by group AND/OR subgroup.

    Parameters
    ----------
    group_name : str or None
        e.g., "CTR", "CVD", "PD", "HD"
    subgroup_name : str or None
        e.g., "protan", "deutan"
    """

    curves = []
    labels = []

    for _, row in FM100.iterrows():
        raw_id = row.iloc[3]
        meta = identify_participant(raw_id, info_table)

        # --- Filtering logic ---
        if group_name is not None and meta["Group"] != group_name:
            continue
        if subgroup_name is not None and meta["Subgroup"] != subgroup_name:
            continue

        profile = extract_error_profile(row, info_table, smooth=smooth, window=window)
        curve = profile["errors_smooth"] if smooth else profile["errors_raw"]

        curves.append(curve)
        labels.append(f"{meta['ValidID']}_S{meta['Session']}")

    if len(curves) == 0:
        print("No participants match the requested group/subgroup.")
        return

    curves = np.vstack(curves)
    x = np.arange(1, 86)

    mean_curve = curves.mean(axis=0)
    std_curve  = curves.std(axis=0)

    plt.figure(figsize=(12, 5))

    # Optional: plot individual curves
    if show_individuals:
        for c in curves:
            plt.plot(x, c, color="gray", alpha=0.2)

    # Mean curve
    plt.plot(x, mean_curve, color="blue", linewidth=2, label="Mean")

    # Std shading
    plt.fill_between(
        x,
        mean_curve - std_curve,
        mean_curve + std_curve,
        color="blue",
        alpha=0.2,
        label="±1 std"
    )

    # Title logic
    title = "FM100 Group Error Profile"
    if group_name:
        title += f" — {group_name}"
    if subgroup_name:
        title += f" ({subgroup_name})"

    plt.title(title)
    plt.xlabel("Cap Index (1–85)")
    plt.ylabel("Error Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_group_errors_lims(
    FM100,
    info_table,
    group_name=None,
    subgroup_name=None,
    smooth=True,
    window=15,
    show_individuals=False,
    ymin=None,
    ymax=None
):
    """
    Same as plot_group_errors(), but allows manual y-axis limits.
    """

    curves = []
    labels = []

    for _, row in FM100.iterrows():
        raw_id = row.iloc[3]
        meta = identify_participant(raw_id, info_table)

        # --- Filtering logic ---
        if group_name is not None and meta["Group"] != group_name:
            continue
        if subgroup_name is not None and meta["Subgroup"] != subgroup_name:
            continue

        profile = extract_error_profile(row, info_table, smooth=smooth, window=window)
        curve = profile["errors_smooth"] if smooth else profile["errors_raw"]

        curves.append(curve)
        labels.append(f"{meta['ValidID']}_S{meta['Session']}")

    if len(curves) == 0:
        print("No participants match the requested group/subgroup.")
        return

    curves = np.vstack(curves)
    x = np.arange(1, 86)

    mean_curve = curves.mean(axis=0)
    std_curve  = curves.std(axis=0)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Optional: plot individual curves
    if show_individuals:
        for c in curves:
            ax.plot(x, c, color="gray", alpha=0.2)

    # Mean curve
    ax.plot(x, mean_curve, color="blue", linewidth=2, label="Mean")

    # Std shading
    ax.fill_between(
        x,
        mean_curve - std_curve,
        mean_curve + std_curve,
        color="blue",
        alpha=0.2,
        label="±1 std"
    )

    # Title logic
    title = "FM100 Group Error Profile"
    if group_name:
        title += f" — {group_name}"
    if subgroup_name:
        title += f" ({subgroup_name})"

    ax.set_title(title)
    ax.set_xlabel("Cap Index (1–85)")
    ax.set_ylabel("Error Value")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --- Apply y-axis limits only if provided ---
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    plt.show()

def plot_population_comparison(
    FM100,
    info_table,
    smooth=True,
    window=15
):
    """
    Plot mean FM100 error curves for:
        - CTR
        - PD
        - CVD/protan
        - CVD/deutan
    All in the same plot, different colors, no individuals.

    Parameters
    ----------
    FM100 : DataFrame
    info_table : DataFrame
    smooth : bool
    window : int
    """

    # Define the populations to compare
    populations = [
        ("CTR", None, "blue", "CTR"),
        ("PD",  None, "red",  "PD"),
        ("CVD", "protan", "green", "CVD protan"),
        ("CVD", "deutan", "orange", "CVD deutan"),
    ]

    plt.figure(figsize=(12, 6))

    x = np.arange(1, 86)

    for group_name, subgroup_name, color, label in populations:

        curves = []

        for _, row in FM100.iterrows():
            raw_id = row.iloc[3]
            meta = identify_participant(raw_id, info_table)

            # --- Filtering logic ---
            if meta["Group"] != group_name:
                continue
            if subgroup_name is not None and meta["Subgroup"] != subgroup_name:
                continue

            profile = extract_error_profile(row, info_table, smooth=smooth, window=window)
            curve = profile["errors_smooth"] if smooth else profile["errors_raw"]
            curves.append(curve)

        if len(curves) == 0:
            continue

        curves = np.vstack(curves)
        mean_curve = curves.mean(axis=0)

        # Plot mean curve
        plt.plot(x, mean_curve, color=color, linewidth=2.5, label=label)

    plt.title("FM100 Error Profiles — Population Comparison")
    plt.xlabel("Cap Index (1–85)")
    plt.ylabel("Error Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_population_comparison_shading(
    FM100,
    info_table,
    smooth=True,
    window=15
):
    """
    Plot mean FM100 error curves for:
        - CTR
        - PD
        - CVD/protan
        - CVD/deutan
    All in the same plot, with std shading.

    Parameters
    ----------
    FM100 : DataFrame
    info_table : DataFrame
    smooth : bool
    window : int
    """

    # Define the populations to compare
    populations = [
        ("CTR", None, "blue",   "CTR"),
        ("PD",  None, "red",    "PD"),
        ("CVD", "protan", "green",  "CVD protan"),
        ("CVD", "deutan", "orange", "CVD deutan"),
    ]

    plt.figure(figsize=(12, 6))
    x = np.arange(1, 86)

    for group_name, subgroup_name, color, label in populations:

        curves = []

        # Collect curves for this population
        for _, row in FM100.iterrows():
            raw_id = row.iloc[3]
            meta = identify_participant(raw_id, info_table)

            if meta["Group"] != group_name:
                continue
            if subgroup_name is not None and meta["Subgroup"] != subgroup_name:
                continue

            profile = extract_error_profile(row, info_table, smooth=smooth, window=window)
            curve = profile["errors_smooth"] if smooth else profile["errors_raw"]
            curves.append(curve)

        if len(curves) == 0:
            continue

        curves = np.vstack(curves)
        mean_curve = curves.mean(axis=0)
        std_curve  = curves.std(axis=0)

        # Plot mean curve
        plt.plot(x, mean_curve, color=color, linewidth=2.5, label=label)

        # Plot std shading
        plt.fill_between(
            x,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=color,
            alpha=0.15
        )

    plt.title("FM100 Error Profiles — Population Comparison")
    plt.xlabel("Cap Index (1–85)")
    plt.ylabel("Error Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
