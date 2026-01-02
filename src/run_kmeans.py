# run_kmeans.py
# This script:
# loads the standardized happiness data
# Drop rows with any missing standardized values
# runs K-means for K = 3, 4, 5, 6
# computes silhouette scores
# chooses the best K
# runs final K-means and saves cluster assignments + profiles

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DATA_FILE = "results/happiness_standardized.csv"


def main():
    print(">>> RUN_KMEANS.PY IS RUNNING <<<")

    # 1) Load standardized data (already cleaned) - read happiness_standardized.csv
    df = pd.read_csv(DATA_FILE)

    # Extract standardized feature columns into X
    feature_cols_std = [
        "log_GDP_std",
        "Social_Support_std",
        "Life_expectancy_std",
        "Freedom_std",
        "Generosity_std",
        "Corruption_std",
    ]

    print("Number of rows in standardized data:", len(df))
    
    X = df[feature_cols_std].values


    # 2) Try different K and compute silhouette scores, using:
    # random_state=42 and n_init=10
    
    k_values = [3, 4, 5, 6]
    scores = {}

    print("Silhouette scores for different K:")
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        scores[k] = score
        print(f"K = {k}: silhouette score = {score:.4f}")

    # 3) Choose the best K (highest silhouette score)
    best_k = max(scores, key=scores.get)
    print(f"\nBest K according to silhouette score: K = {best_k}")

    # 4) Fit final K-means with best K
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    
    # run .fit_predict(X)
    df["cluster"] = final_kmeans.fit_predict(X)

    # 5) Save cluster assignments (per country)
    cluster_cols_to_save = [
        "Country name",
        "Ladder score",
        "log_GDP",
        "Social_Support",
        "Life_expectancy",
        "Freedom",
        "Generosity",
        "Corruption",
        "cluster",
    ]
    df[cluster_cols_to_save].to_csv(
        "results/cluster_assignments.csv", index=False
    )
    print("\nSaved cluster assignments to: results/cluster_assignments.csv")

    # 6) Create cluster profile table: mean of standardized vars per cluster
    profile_cols = feature_cols_std + ["cluster"]
    profiles = (
        df[profile_cols]
        .groupby("cluster")
        .mean()
        .reset_index()
    )
    
    # save profiles in results
    profiles.to_csv("results/cluster_profiles.csv", index=False)
    print("Saved cluster profiles to: results/cluster_profiles.csv")

    print("\nCluster profiles (mean standardized values):")
    print(profiles)


if __name__ == "__main__":
    main()
