# pca_clusters.py
# PCA visualization of K-Means clusters

import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

STD_FILE = "results/happiness_standardized.csv"
CLUSTER_FILE = "results/cluster_assignments.csv"

def main():
    print(">>> PCA_CLUSTERS.PY IS RUNNING <<<")
    
    # 1) load standardized data
    df_std = pd.read_csv(STD_FILE)
    
    # 2) Load cluster labels (may not include _std columns)
    df_clusters = pd.read_csv(CLUSTER_FILE)[["Country name", "cluster"]]
    
    # 3) Merge to attach cluster label to standardized features
    df = df_std.merge(df_clusters, on="Country name", how="left")
    
    # Drop rows without a cluster (the 3 countries we dropped in K-Means)
    df = df.dropna(subset=["cluster"]).copy()
    df["cluster"] = df["cluster"].astype(int)
    
    factor_std_cols = [
        "log_GDP_std",
        "Social_Support_std",
        "Life_expectancy_std",
        "Freedom_std",
        "Generosity_std",
        "Corruption_std",
    ]
    
    print("\nUsing standardized factor columns for PCA:")
    print(factor_std_cols)
    
    X = df[factor_std_cols].values
    
    # 4) Run PCA with 2 components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    df["PC1"] = X_pca[:, 0]
    df["PC2"] = X_pca[:, 1]
    
    expl_var = pca.explained_variance_ratio_
    pc1_var = expl_var[0] * 100
    pc2_var = expl_var[1] * 100
    
    print(f"\nExplained variance by PC1: {pc1_var:.1f}%")
    print(f"Explained variance by PC2: {pc2_var:.1f}%")
    
    # 5) Scatter plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        df["PC1"],
        df["PC2"],
        c=df["cluster"],
        cmap="tab10",
        alpha=0.8,
        edgecolors="k",
        linewidths=0.5,
    )
    
    plt.xlabel(f"PC1 ({pc1_var:.1f}% variance)")
    plt.ylabel(f"PC2 ({pc2_var:.1f}% variance)")
    plt.title("PCA of happiness drivers with K-Means clusters")
    
    # Legend: one entry per cluster
    handles, labels = scatter.legend_elements(prop="colors", num=df["cluster"].nunique())
    plt.legend(handles, labels, title="Cluster", loc="best")
                                              
    plt.tight_layout()
    plt.savefig("results/pca_clusters.png", dpi=300)
    plt.close()
    
    print("\nSaved PCA scatter plot to: results/pca_clusters.png")
                                              
if __name__ == "__main__":
    main()