# src/metamers/behavioral/plotting.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
# ---------------------------------------------------------
# 1. Load behavioral table
# ---------------------------------------------------------
def load_behavior_table(filename="behavioral_table.csv"):
    # Get the absolute path to THIS file (plotting.py)
    module_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up two levels: src/metamers/behavioral → src/metamers → metamers/
    project_root = os.path.abspath(os.path.join(module_dir, "..", "..", ".."))

    # Build the path to the behavioral folder at the project root
    behavioral_dir = os.path.join(project_root, "behavioral")

    # Full path to the table
    full_path = os.path.join(behavioral_dir, filename)

    return pd.read_csv(full_path)
# ---------------------------------------------------------
# 2. Plot a single participant (all runs)
# ---------------------------------------------------------
def plot_participant(df, subid):
    df_sub = df[df["SubID"] == subid]

    plt.figure(figsize=(10, 5))
    sns.scatterplot(
        data=df_sub,
        x="Red",
        y="Green",
        hue="session",
        palette="viridis",
        s=80
    )
    plt.xlim(0, 3200)
    plt.ylim(0, 2000)
    plt.title(f"Participant {subid}: Red vs Green")
    plt.xlabel("Red")
    plt.ylabel("Green")
    plt.legend(title="Session")
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# 3. Plot all participants in a grid
# ---------------------------------------------------------
def plot_all_participants(df):
    subs = sorted(df["SubID"].unique())
    n = len(subs)

    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
    axes = axes.flatten()

    for ax, sub in zip(axes, subs):
        df_sub = df[df["SubID"] == sub]
        ax.scatter(df_sub["Red"], df_sub["Green"], s=20)
        ax.set_title(f"{sub}")
        ax.set_xlabel("Red")
        ax.set_ylabel("Green")
        ax.set_xlim(0, 3200)
        ax.set_ylim(0, 2000)
    # Hide unused axes
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# 4. Plot participants by group (PartType)
# ---------------------------------------------------------
def plot_group(df, parttype, color=None):
    df_group = df[df["PartType"] == parttype]

    # default color 
    if color is None:
        color = "#000000"

    plt.figure(figsize=(7, 7))
    sns.scatterplot(
        data=df_group,
        x="Red",
        y="Green",
        hue=None,
        color=color,
        palette="tab20",
        s=80
    )
    plt.xlim(0, 3200)
    plt.ylim(0, 2000)
    plt.title(f"Group {parttype}: Red vs Green")
    plt.xlabel("Red")
    plt.ylabel("Green")
    plt.tight_layout()
    plt.show()
# ---------------------------------------------------------
# 5. Plot multiple participants on the same axes
# ---------------------------------------------------------
def plot_participants(df, subids, color=None, title=None):
    """
    Plot multiple participants on the same axes.

    Parameters
    ----------
    df : DataFrame
        Behavioral table.
    subids : list
        List of participant IDs to plot.
    color : str or None
        Hex color string like "#534b7e". If None, defaults to black.
    """
    # Default color
    if color is None:
        color = "#000000"
    if title is None:
        title = f"group Red vs Green"
    # Filter only the requested participants
    df_sel = df[df["SubID"].isin(subids)]

    plt.figure(figsize=(10, 5))

    # Plot all selected participants with the same color
    sns.scatterplot(
        data=df_sel,
        x="Red",
        y="Green",
        hue=None,      # no hue mapping
        color=color,   # user-defined or black
        s=80
    )

    plt.xlim(0, 3200)
    plt.ylim(0, 2000)

    plt.title(title)
    plt.xlabel("Red")
    plt.ylabel("Green")
    plt.tight_layout()
    plt.show()
# ---------------------------------------------------------
# 6. Plot multiple participants on the same axes
# ---------------------------------------------------------

def plot_participants_huemapped(df, subids, title=None, colors=None):
    """
    Plot multiple participants on the same axes, each with its own color.

    Parameters
    ----------
    df : DataFrame
        Behavioral table.
    subids : list
        List of participant IDs to plot.
    title : str or None
        Title for the plot.
    colors : list of hex strings or None
        Custom colors for each participant. Must match len(subids).
        If None, a default seaborn palette is used.
    """
    # Default title
    if title is None:
        title = f"Participants {subids}: Red vs Green"

    # Filter only the requested participants
    df_sel = df[df["SubID"].isin(subids)]

    # Build a palette
    if colors is None:
        # Use seaborn default palette with as many colors as participants
        palette = sns.color_palette("tab10", n_colors=len(subids))
    else:
        # Validate custom colors
        if len(colors) != len(subids):
            raise ValueError("Length of colors must match length of subids.")
        palette = colors

    # Map SubID → color
    color_map = {sid: palette[i] for i, sid in enumerate(subids)}

    plt.figure(figsize=(10, 5))

    sns.scatterplot(
        data=df_sel,
        x="Red",
        y="Green",
        hue="SubID",
        palette=color_map,
        s=80
    )

    plt.xlim(0, 3200)
    plt.ylim(0, 2000)

    plt.title(title)
    plt.xlabel("Red")
    plt.ylabel("Green")
    plt.tight_layout()
    plt.show()
