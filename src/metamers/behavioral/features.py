import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
import os

def compute_entropy(points, bandwidth=0.2, grid_size=50):
    """
    Estimate 2D entropy using KDE on a grid.
    """
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(points)

    # Grid for density evaluation
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)

    xs = np.linspace(x_min, x_max, grid_size)
    ys = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.vstack([xx.ravel(), yy.ravel()]).T

    log_density = kde.score_samples(grid)
    density = np.exp(log_density)
    density /= density.sum()  # normalize

    entropy = -np.sum(density * np.log(density + 1e-12))
    return entropy


def compute_convex_hull_area(points):
    if len(points) < 3:
        return 0.0
    hull = ConvexHull(points)
    return hull.area


def extract_features_for_participant(df_part):
    """
    df_part: subset of df for a single participant.
    Must contain columns ['Red', 'Green'].
    """
    pts = df_part[['Red', 'Green']].values

    # --- PCA ---
    pca = PCA(n_components=2)
    pca.fit(pts)
    pc1_var, pc2_var = pca.explained_variance_
    angle_rad = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
    angle_deg = np.degrees(angle_rad)
    # --- Covariance ---
    cov = np.cov(pts, rowvar=False)
    cov_rr = cov[0, 0]
    cov_gg = cov[1, 1]
    cov_rg = cov[0, 1]

    # --- Ellipse area (1 SD ellipse) ---
    ellipse_area = np.pi * np.sqrt(pc1_var) * np.sqrt(pc2_var)

    # --- Convex hull area ---
    hull_area = compute_convex_hull_area(pts)

    # --- Distribution shape ---
    skew_r = skew(pts[:, 0])
    skew_g = skew(pts[:, 1])
    kurt_r = kurtosis(pts[:, 0])
    kurt_g = kurtosis(pts[:, 1])

    # --- Entropy ---
    entropy = compute_entropy(pts)

    return {
        "pc1_var": pc1_var,
        "pc2_var": pc2_var,
        "anisotropy": pc1_var / pc2_var if pc2_var > 0 else np.nan,
        "pca_angle_rad": angle_rad,
        "pca_angle_deg": angle_deg,
        "cov_rr": cov_rr,
        "cov_gg": cov_gg,
        "cov_rg": cov_rg,
        "ellipse_area": ellipse_area,
        "hull_area": hull_area,
        "skew_r": skew_r,
        "skew_g": skew_g,
        "kurt_r": kurt_r,
        "kurt_g": kurt_g,
        "entropy": entropy,
        "n_points": len(pts),
    }


def extract_behavioral_features(df):
    """
    df: output of load_behavior_table()
    Must contain columns ['SubID', 'Red', 'Green'].
    Returns a participant-level feature table.
    """
    participants = df['SubID'].unique()
    rows = []

    for pid in participants:
        df_part = df[df['SubID'] == pid]
        feats = extract_features_for_participant(df_part)
        feats['SubID'] = pid
        rows.append(feats)

    # Convert rows → DataFrame BEFORE saving
    feat_df = pd.DataFrame(rows)

    # Get the absolute path to THIS file (features.py)
    module_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up three levels: src/metamers/behavioral → src/metamers → src → project root
    project_root = os.path.abspath(os.path.join(module_dir, "..", "..", ".."))

    # Path to the behavioral folder at the project root
    behavioral_dir = os.path.join(project_root, "behavioral")
    os.makedirs(behavioral_dir, exist_ok=True)

    # Full paths
    csv_path = os.path.join(behavioral_dir, "behavioral_features.csv")
    parquet_path = os.path.join(behavioral_dir, "behavioral_features.parquet")

    # Save
    feat_df.to_csv(csv_path, index=False)
    feat_df.to_parquet(parquet_path, index=False)

    print("Saved behavioral features to:")
    print(" -", csv_path)
    print(" -", parquet_path)

    return feat_df
