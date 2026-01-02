# compare_clusters.py
# Compare baseline clusters with clusters from model without GDP (in run_kmeans_no_gdp.py)

import pandas as pd

BASE_FILE = "results/cluster_assignments.csv"
NO_GDP_FILE = "results/cluster_assignments_no_gdp.csv"


def main():
    print(">>> COMPARE_CLUSTERS.PY IS RUNNING <<<")

    # 1) Load both sets of labels
    df_base = pd.read_csv(BASE_FILE)[["Country name", "cluster"]]
    df_no_gdp = pd.read_csv(NO_GDP_FILE)[["Country name", "cluster_no_gdp"]]

    # 2) Merge on country name
    df_merged = df_base.merge(df_no_gdp, on="Country name", how="inner")

    print(f"\nNumber of countries in comparison: {len(df_merged)}")

    # 3) Confusion matrix (crosstab)
    confusion = pd.crosstab(df_merged["cluster"], df_merged["cluster_no_gdp"])
    print("\nConfusion matrix (rows = baseline cluster, cols = no-GDP cluster):")
    print(confusion)

    # 4) Percentage of unchanged labels (direct match)
    same_label = (df_merged["cluster"] == df_merged["cluster_no_gdp"])
    pct_same = same_label.mean() * 100
    print(f"\nShare of countries with identical cluster label: {pct_same:.1f}%")

    # Save confusion matrix
    confusion.to_csv("results/cluster_confusion_no_gdp.csv")
    print("\nSaved confusion matrix to: results/cluster_confusion_no_gdp.csv")


if __name__ == "__main__":
    main()
