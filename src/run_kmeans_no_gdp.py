# run_kmeans_no_gdp.py
# K-Means clustering on all standardized factors EXCEPT log_GDP
# robustness check: clustering without log_GDP_std to could next compare with initial cluster model in compare_clusters.py

import pandas as pd
from sklearn.cluster import KMeans

STD_FILE = "results/happiness_standardized.csv"


def main():
    print(">>> RUN_KMEANS_NO_GDP.PY IS RUNNING <<<")

    # 1) Load standardized data (already cleaned)
    df = pd.read_csv(STD_FILE)

    # Columns used for K-Means (exclude log_GDP_std)
    factor_std_cols = [
        "Social_Support_std",
        "Life_expectancy_std",
        "Freedom_std",
        "Generosity_std",
        "Corruption_std",
    ]

    print("\nUsing standardized factor columns (no GDP):")
    print(factor_std_cols)
    print("Number of rows in standardized data:", len(df))
    
    X = df[factor_std_cols].values

    # 2) Run K-Means with K=3 (same K as baseline)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels_no_gdp = kmeans.fit_predict(X)

    df["cluster_no_gdp"] = labels_no_gdp

    # 3) Save new assignments
    out_cols = ["Country name", "cluster_no_gdp"]
    df[out_cols].to_csv("results/cluster_assignments_no_gdp.csv", index=False)

    print(
        "\nSaved cluster assignments without GDP to: "
        "results/cluster_assignments_no_gdp.csv"
    )
    
if __name__ == "__main__":
    main()
