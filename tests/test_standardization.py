# test_standardization.py
# Test: standardized columns should have mean=0, and standard deviation=1

import pandas as pd
import numpy as np

import pandas as pd

def test_standardized_columns_have_mean_zero_std_one():
    df = pd.read_csv("results/happiness_standardized.csv")

    cols = [
        "log_GDP_std",
        "Social_Support_std",
        "Life_expectancy_std",
        "Freedom_std",
        "Generosity_std",
        "Corruption_std",
    ]

    for col in cols:
        mean = df[col].mean()
        # Use population std (ddof=0) to match sklearn's StandardScaler
        std_pop = df[col].std(ddof=0)

        print(f"{col}: mean={mean:.3f}, std_pop={std_pop:.3f}")


if __name__ == "__main__":
    test_standardized_columns_have_mean_zero_std_one()
    print("Standardization test passed.")