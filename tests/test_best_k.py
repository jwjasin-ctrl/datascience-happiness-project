# test_best_k.py
# Re-run K-Means for K=3,4,5,6 and check that K=3 has the best silhouette score.
# used the same cleaning as in run_kmeans.py, otherwise an error apears: "ValueError: Input contains NaN, infinity or a value too large for dtype('float64')"

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def test_k3_has_best_silhouette():
    df = pd.read_csv("results/happiness_standardized.csv")

    cols = [
        "log_GDP_std",
        "Social_Support_std",
        "Life_expectancy_std",
        "Freedom_std",
        "Generosity_std",
        "Corruption_std",
    ]

    # clean the data (as run_kmeans.py)
    rows_before = len(df)
    df_clean = df.dropna(subset=cols)
    rows_after = len(df_clean)
    
    print(f"Rows before dropping NaNs: {rows_before}")
    print(f"Rows after dropping NaNs: {rows_after}")

    # use df_clean
    X = df_clean[cols].values

    silhouette_scores = {}
    for k in [3, 4, 5, 6]:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores[k] = score
        print(f"K={k}: silhouette={score:.4f}")

    best_k = max(silhouette_scores, key=silhouette_scores.get)
    print(f"Best K according to silhouette in this test: {best_k}")

    assert best_k == 3

if __name__ == "__main__":
    test_k3_has_best_silhouette()
    print("Silhouette K=3 test passed.")
