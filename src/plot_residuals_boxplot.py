# plot_residuals_boxplot.py
# Boxplot of residual happiness (happiness beyond GDP) by cluster

import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = "results/cluster_assignments_with_resid.csv"


def main():
    print(">>> PLOT_RESIDUALS_BOXLOT.PY IS RUNNING <<<")

    df = pd.read_csv(DATA_FILE)

    print("\nColumns in data:")
    print(list(df.columns))

    # Ensure cluster is int
    df["cluster"] = df["cluster"].astype(int)

    # Prepare data in order of clusters
    clusters = sorted(df["cluster"].unique())
    data = [df.loc[df["cluster"] == c, "happiness_resid"] for c in clusters]

    plt.figure(figsize=(8, 6))
    plt.boxplot(data, labels=[str(c) for c in clusters])

    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Cluster")
    plt.ylabel("Residual happiness (actual - predicted from GDP)")
    plt.title("Residual happiness by cluster")

    plt.tight_layout()
    plt.savefig("results/residuals_boxplot.png", dpi=300)
    plt.close()

    print("\nSaved boxplot to: results/residuals_boxplot.png")


if __name__ == "__main__":
    main()

