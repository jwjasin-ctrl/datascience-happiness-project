# tests/test_pca_var.py
# Check that the first two principal components explain at least 60% of the variance.
# used the same cleaning as in run_kmeans.py

import pandas as pd
from sklearn.decomposition import PCA

def test_pca_two_components_explain_enough_variance():
    df = pd.read_csv("results/happiness_standardized.csv")

    factor_std_cols = [
        "log_GDP_std",
        "Social_Support_std",
        "Life_expectancy_std",
        "Freedom_std",
        "Generosity_std",
        "Corruption_std",
    ]
    
    X = df[factor_std_cols].values

    pca = PCA(n_components=2)
    pca.fit(X)
    
    var_ratio = pca.explained_variance_ratio_
    total_explained = var_ratio.sum()

    print("Explained variance ratios (PC1, PC2):", var_ratio)
    print("Total variance explained by first two PCs:", total_explained)

    # Require at least 60% of total variance explained by PC1+PC2
    assert total_explained >= 0.60

if __name__ == "__main__":
    test_pca_two_components_explain_enough_variance()
    print("PCA variance test passed.")
