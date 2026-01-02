# interaction_gdp_pairs.py
# Visualize interactions of log GDP with Life expectancy and Generosity
# coloured by K-Means clusters

import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = "results/cluster_assignments.csv"
OUT_LIFE = "results/gdp_lifeexpectancy_clusters.png"
OUT_GEN = "results/gdp_generosity_clusters.png"


def choose_column(df, candidates):
    """Return the first column name from candidates that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of {candidates} found in dataframe columns.")


def plot_pair(df, x_col, y_col, out_file, title):
    """Scatter of two features, coloured by cluster."""
    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        df[x_col],
        df[y_col],
        c=df["cluster"],
        cmap="viridis",
        alpha=0.8,
    )

    # Legend for clusters
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend1)

    # Zero lines (helpful if data are z-scores; if not, they still
    # show the point where each variable is equal to 0)
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.axhline(0, color="grey", linestyle="--", linewidth=1)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved plot to: {out_file}")


def main():
    print(">>> INTERACTION_GDP_PAIRS.PY IS RUNNING <<<")

    # 1) Load data with features and clusters
    df = pd.read_csv(DATA_FILE)
    print("Columns in cluster_assignments.csv:")
    print(list(df.columns))

    # Prefer standardized columns if they exist; otherwise use original
    x_col = choose_column(df, ["log_GDP_std", "log_GDP"])
    y_life = choose_column(df, ["Life_expectancy_std", "Life_expectancy"])
    y_gen = choose_column(df, ["Generosity_std", "Generosity"])

    # Check cluster column
    if "cluster" not in df.columns:
        print("Error: 'cluster' column not found.")
        return

    print(f"\nUsing columns:")
    print(f"x (GDP):        {x_col}")
    print(f"y (Life exp.):  {y_life}")
    print(f"y (Generosity): {y_gen}")

    # 2) Print correlations for information
    corr_life = df[[x_col, y_life]].corr().iloc[0, 1]
    corr_gen = df[[x_col, y_gen]].corr().iloc[0, 1]
    print(f"\nCorrelation {x_col} vs {y_life}: {corr_life:.3f}")
    print(f"Correlation {x_col} vs {y_gen}:  {corr_gen:.3f}")

    # 3) Plot GDP vs Life expectancy
    plot_pair(
        df,
        x_col,
        y_life,
        OUT_LIFE,
        f"{x_col} vs {y_life} (corr = {corr_life:.2f})",
    )

    # 4) Plot GDP vs Generosity
    plot_pair(
        df,
        x_col,
        y_gen,
        OUT_GEN,
        f"{x_col} vs {y_gen} (corr = {corr_gen:.2f})",
    )


if __name__ == "__main__":
    main()
