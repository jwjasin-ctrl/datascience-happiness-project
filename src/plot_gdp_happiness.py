# plot_gdp_happiness.py
# Scatter of happiness vs log GDP, coloured by cluster,
# with labels for countries that strongly over-/under-perform
# relative to their GDP (based on residuals).

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

DATA_FILE = "results/cluster_assignments_with_resid.csv"


def main():
    print(">>> PLOT_GDP_HAPPINESS.PY IS RUNNING <<<")

    # 1) Load data (already contains log_GDP, Ladder score, cluster, happiness_resid)
    df = pd.read_csv(DATA_FILE)

    print("\nColumns in data:")
    print(list(df.columns))

    # Make sure cluster is integer
    df["cluster"] = df["cluster"].astype(int)

    # 2) Fit a simple linear regression: happiness ~ log_GDP
    X = df[["log_GDP"]].values
    y = df["Ladder score"].values

    model = LinearRegression()
    model.fit(X, y)

    intercept = model.intercept_
    slope = model.coef_[0]
    r2 = model.score(X, y)

    print(f"\nLinear regression: Ladder score ~ log_GDP")
    print(f"Intercept: {intercept:.3f}")
    print(f"Slope: {slope:.3f}")
    print(f"R^2: {r2:.3f}")

    # 3) Identify top/bottom 5 by residual (already computed, but sort again)
    df_sorted = df.sort_values("happiness_resid")

    bottom5 = df_sorted.head(5)   # least happy given their GDP
    top5 = df_sorted.tail(5)      # happiest given their GDP

    print("\n5 countries MUCH LESS happy than their GDP predicts:")
    print(bottom5[["Country name", "log_GDP", "Ladder score", "happiness_resid", "cluster"]])

    print("\n5 countries MUCH MORE happy than their GDP predicts:")
    print(top5[["Country name", "log_GDP", "Ladder score", "happiness_resid", "cluster"]])

    # 4) Scatter plot coloured by cluster
    plt.figure(figsize=(8, 6))

    clusters = sorted(df["cluster"].unique())
    for c in clusters:
        subset = df[df["cluster"] == c]
        plt.scatter(
            subset["log_GDP"],
            subset["Ladder score"],
            label=f"Cluster {c}",
            alpha=0.7,
        )

    # Regression line
    x_vals = np.linspace(df["log_GDP"].min(), df["log_GDP"].max(), 100).reshape(-1, 1)
    y_pred = model.predict(x_vals)
    plt.plot(x_vals, y_pred, linewidth=2)

    plt.xlabel("log GDP per capita")
    plt.ylabel("Happiness (Ladder score)")
    plt.title("Happiness vs log GDP with clusters and extreme residuals")
    plt.axhline(0, color="grey", linewidth=0.5)

    # 5) Label top/bottom 5 on the plot
    def add_labels(subset, color):
        for _, row in subset.iterrows():
            plt.text(
                row["log_GDP"],
                row["Ladder score"],
                row["Country name"],
                fontsize=7,
                ha="left",
                va="bottom",
            )

    add_labels(bottom5, "black")
    add_labels(top5, "black")

    plt.legend()
    plt.tight_layout()
    plt.savefig("results/gdp_happiness_scatter_labeled.png", dpi=300)
    plt.close()

    print("\nSaved labeled scatter plot to: results/gdp_happiness_scatter_labeled.png")


if __name__ == "__main__":
    main()
