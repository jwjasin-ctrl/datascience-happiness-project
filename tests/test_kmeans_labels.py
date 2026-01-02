# test_kmeans_labels.py
# Check that there are exactly 3 distinct clusters in the baseline solution

import pandas as pd

def test_three_clusters_present():
    df = pd.read_csv("results/cluster_assignments.csv")

    unique_clusters = sorted(df["cluster"].unique())
    assert len(unique_clusters) == 3

if __name__ == "__main__":
    test_three_clusters_present()
    print("K-Means cluster label test passed.")
