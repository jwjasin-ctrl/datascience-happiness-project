# gdp_residuals.py
# Regress happiness on log_GDP and analyze residuals by cluster

import pandas as pd
from sklearn.linear_model import LinearRegression

CLUSTER_FILE = "results/cluster_assignments.csv"

def main():
    print(">>> GDP_RESIDUALS.PY IS RUNNING <<<")
    
    # 1) Load data with cluster labels
    df = pd.read_csv(CLUSTER_FILE)
    
    print("\nColumns in cluster_assignments.csv:")
    print(list(df.columns))
    
    # 2) Simple regression: Ladder score ~log_GDP
    X = df[["log_GDP"]]     # predictor
    y = df["Ladder score"]  # outcome (happiness)
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    df["happiness_resid"] = residuals
    
    print("\nLinear regression: Ladder score ~ log_GDP")
    print(f"Intercept: {model.intercept_:.3f}")
    print(f"Slope (log_GDP): {model.coef_[0]:.3f}")
    print(f"R^2: {model.score(X, y):.3f}")
    
    print("\nSummary of residuals (happiness beyond GDP):")
    print(df["happiness_resid"].describe())
    
    # 3) Residual happiness by cluster
    print("\nMean residual happiness by cluster:")
    resid_by_cluster = df.groupby("cluster")["happiness_resid"].agg(["count", "mean", "std"])
    print(resid_by_cluster)
    
    # 4) Save outputs
    df.to_csv("results/cluster_assignments_with_resid.csv", index=False)
    resid_by_cluster.to_csv("results/cluster_residuals_summary.csv")
    
    print("\nSaved:")
    print(" - results/cluster_assignments_with_resid.csv")
    print(" - results/cluster_residuals_summary.csv")
    
if __name__ == "__main__":
    main()
