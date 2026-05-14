import seaborn as sns
import matplotlib.pyplot as plt

def plot_combined_correlation(df):
    # Drop non-numeric columns
    num = df.select_dtypes(include=['float64', 'int64'])

    plt.figure(figsize=(14, 12))
    sns.heatmap(num.corr(), cmap="coolwarm", center=0)
    plt.title("Correlation Between FM100 Metrics and Behavioral Features")
    plt.show()

def plot_angle_relationship(df):
    """
    Plot behavioral PCA angle (degrees) vs FM100 Vingrys angle (degrees),
    coloring points by Group, but splitting CVD into Protan/Deutan.
    """

    # Create a plotting group column
    df = df.copy()
    df["GroupPlot"] = df["Group"]

    # Replace CVD with its Subgroup (Protan/Deutan)
    mask = df["Group"] == "CVD"
    df.loc[mask, "GroupPlot"] = df.loc[mask, "Subgroup"]

    plt.figure(figsize=(7, 6))

    sns.scatterplot(
        data=df,
        x="pca_angle_deg",
        y="VKS_Angle",
        hue="GroupPlot",
        palette="tab10",
        s=80,
        alpha=0.85
    )

    # Identity line
    plt.axline((0, 0), slope=1, color="gray", linestyle="--", linewidth=1)

    plt.xlabel("Behavioral PCA Angle (deg)")
    plt.ylabel("FM100 Vingrys Angle (deg)")
    plt.title("PCA Angle vs Vingrys Angle (CVD split into Protan/Deutan)")
    plt.legend(title="Group")
    plt.tight_layout()
    plt.show()



def plot_spread_vs_severity(df):
    plt.figure(figsize=(6,6))
    sns.scatterplot(data=df, x="ellipse_area", y="TES")
    plt.xlabel("Behavioral Ellipse Area")
    plt.ylabel("FM100 Total Error Score (TES)")
    plt.title("Behavioral Spread vs FM100 Severity")
    plt.show()

def plot_anisotropy_vs_RG(df):
    plt.figure(figsize=(6,6))
    sns.scatterplot(data=df, x="anisotropy", y="PES_RG")
    plt.xlabel("Behavioral Anisotropy (PC1/PC2)")
    plt.ylabel("FM100 PES_RG")
    plt.title("Behavioral Anisotropy vs Red-Green Error")
    plt.show()

def plot_anisotropy_vs_BY(df):
    plt.figure(figsize=(6,6))
    sns.scatterplot(data=df, x="anisotropy", y="PES_BY")
    plt.xlabel("Behavioral Anisotropy (PC1/PC2)")
    plt.ylabel("FM100 PES_BY")
    plt.title("Behavioral Anisotropy vs Blue-Yellow Error")
    plt.show()

def plot_entropy_vs_TES(df):
    plt.figure(figsize=(6,6))
    sns.scatterplot(data=df, x="entropy", y="TES")
    plt.xlabel("Behavioral Entropy")
    plt.ylabel("FM100 TES")
    plt.title("Behavioral Entropy vs FM100 Severity")
    plt.show()

def plot_group_distributions(df):
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df, x="Group", y="ellipse_area")
    plt.title("Behavioral Spread Across Groups")
    plt.show()
