# regression_interactions.py
# Multiple linear regression with interactions and a high-corruption dummy,
# plus a bar plot of standardized coefficients.

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

DATA_FILE = "results/clean_happiness_data.csv"
OUT_COEFFS = "results/reginteractions_coeffs.csv"
OUT_PLOT = "results/reginteractions_coeffs.png"


def plot_coefficients(coefs):
    """
    Create a horizontal bar chart of standardized regression coefficients.
    """
    features = coefs["feature"]
    values = coefs["coef_standardized"]

    plt.figure(figsize=(8, 5))
    plt.barh(features, values)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Standardized coefficient")
    plt.title("Regression coefficients with interactions")
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=300)
    plt.close()
    print(f"Saved coefficient plot to: {OUT_PLOT}")


def main():
    print(">>> REGINTERACTIONS.PY IS RUNNING <<<")

    # 1) Load cleaned data
    df = pd.read_csv(DATA_FILE)

    print("\nColumns in data:")
    print(list(df.columns))

    # 2) Choose base features (raw scale) using original WHR names
    gdp_col = "Explained by: Log GDP per capita"
    social_col = "Explained by: Social support"
    life_col = "Explained by: Healthy life expectancy"
    freedom_col = "Explained by: Freedom to make life choices"
    generosity_col = "Explained by: Generosity"
    corruption_col = "Explained by: Perceptions of corruption"

    base_cols = [
        gdp_col,
        social_col,
        life_col,
        freedom_col,
        generosity_col,
        corruption_col,
    ]

    # Target: happiness
    target_col = "Ladder score"

    # 3) High-corruption dummy (bottom 30% of corruption score)
    corr_values = df[corruption_col]
    threshold = corr_values.quantile(0.30)
    df["HighCorruption"] = (corr_values <= threshold).astype(int)

    print(f"\nHighCorruption dummy: threshold (30% quantile) = {threshold:.3f}")
    print(df["HighCorruption"].value_counts())

    # 4) Interaction terms
    df["logGDP_x_SocialSupport"] = df[gdp_col] * df[social_col]
    df["logGDP_x_HighCorr"] = df[gdp_col] * df["HighCorruption"]

    feature_cols = base_cols + [
        "HighCorruption",
        "logGDP_x_SocialSupport",
        "logGDP_x_HighCorr",
    ]

    print("\nFeatures used in regression:")
    print(feature_cols)

    # 5) Drop rows with missing values
    # normally, missing rows were already removed before, but I drop again here to make sure
    df_model = df.dropna(subset=feature_cols + [target_col]).copy()
    print(f"\nRows before dropping NaNs: {len(df)}")
    print(f"Rows after dropping NaNs: {len(df_model)}")

    X_raw = df_model[feature_cols].values
    y = df_model[target_col].values

    # 6) Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # 7) Fit linear regression
    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)

    print(f"\nLinear regression: {target_col} ~ features with interactions")
    print(f"R^2 (in-sample): {r2:.3f}")
    print(f"RMSE (in-sample): {rmse:.3f}")

    # 8) Collect coefficients in a table
    coefs = pd.DataFrame({
        "feature": feature_cols,
        "coef_standardized": model.coef_,
    })
    coefs["abs_coef"] = coefs["coef_standardized"].abs()
    coefs = coefs.sort_values("abs_coef", ascending=False)

    print("\nRegression coefficients (on standardized features):")
    print(coefs[["feature", "coef_standardized"]])

    # 9) Save coefficients to CSV
    coefs.drop(columns=["abs_coef"]).to_csv(OUT_COEFFS, index=False)
    print(f"\nSaved regression coefficients to: {OUT_COEFFS}")

    # 10) Plot coefficients
    plot_coefficients(coefs)


if __name__ == "__main__":
    main()
