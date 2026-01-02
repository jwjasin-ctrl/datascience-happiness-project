# analyze_clusters.py

import pandas as pd

CLUSTER_FILE = "results/cluster_assignments.csv"


def main():
    print(">>> ANALYZE_CLUSTERS.PY IS RUNNING <<<")

    # Load cluster assignments created by run_kmeans.py
    df = pd.read_csv(CLUSTER_FILE)

    print("\nColumns in cluster_assignments.csv:")
    print(list(df.columns))

    # 1) Number of countries in each cluster
    print("\nNumber of countries per cluster:")
    cluster_sizes = df["cluster"].value_counts().sort_index()
    print(cluster_sizes)

    # 2) Mean happiness (Ladder score) per cluster
    print("\nMean Ladder score (happiness) per cluster:")
    happiness_stats = df.groupby("cluster")["Ladder score"].agg(["mean", "std"])
    print(happiness_stats)

    # 3) Mean of the explanatory variables per cluster
    #    (take all columns except country name, ladder score and cluster)
    feature_cols = [
        col
        for col in df.columns
        if col not in ["Country name", "Ladder score", "cluster"]
    ]

    print("\nMean of explanatory variables per cluster:")
    feature_means = df.groupby("cluster")[feature_cols].mean()
    print(feature_means)

    # 4) Build a summary table and save it for the report
    summary = feature_means.copy()
    summary.insert(0, "n_countries", cluster_sizes)
    summary.insert(1, "mean_ladder", happiness_stats["mean"])

    summary.to_csv("results/cluster_summary.csv")
    print("\nSaved cluster summary to: results/cluster_summary.csv")


if __name__ == "__main__":
    main()