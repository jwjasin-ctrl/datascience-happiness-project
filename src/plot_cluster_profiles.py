# plot_cluster_profiles.py
# Bar chart of mean standardized factor values (z-scores) by cluster

import pandas as pd
import matplotlib.pyplot as plt

PROFILE_FILE = "results/cluster_profiles.csv"


def main():
    print(">>> PLOT_CLUSTER_PROFILES.PY IS RUNNING <<<")

    # 1) Load cluster profiles (means of standardized factors)
    df = pd.read_csv(PROFILE_FILE)

    print("\nCluster profiles (input):")
    print(df)

    # Ensure clusters are sorted
    df = df.sort_values("cluster")

    # 2) Select only the standardized factor columns
    factor_cols = [
        "log_GDP_std",
        "Social_Support_std",
        "Life_expectancy_std",
        "Freedom_std",
        "Generosity_std",
        "Corruption_std",
    ]

    # Put 'cluster' as index
    df_plot = df.set_index("cluster")[factor_cols]

    # 3) Make a grouped bar chart
    ax = df_plot.T.plot(kind="bar", figsize=(10, 6))

    ax.set_xlabel("Factor")
    ax.set_ylabel("Mean standardized value (z-score)")
    ax.set_title("Cluster profiles: mean z-scores of happiness drivers")
    ax.axhline(0, color="black", linewidth=0.8)

    # Make legend nicer
    ax.legend(title="Cluster", loc="best")

    plt.tight_layout()
    plt.savefig("results/cluster_profiles_bars.png", dpi=300)
    plt.close()

    print("\nSaved bar chart to: results/cluster_profiles_bars.png")


if __name__ == "__main__":
    main()
