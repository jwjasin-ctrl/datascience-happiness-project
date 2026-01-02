# test_no_missing_std_features.py
# Check that the standardized features used for clustering contain no missing values.

import pandas as pd

def test_no_missing_in_standardized_features():
    df = pd.read_csv("results/happiness_standardized.csv")

    factor_std_cols = [
        "log_GDP_std",
        "Social_Support_std",
        "Life_expectancy_std",
        "Freedom_std",
        "Generosity_std",
        "Corruption_std",
    ]

    missing_total = df[factor_std_cols].isna().sum().sum()

    
    print("Rows in standardized data:", len(df))
    print("Total missing values in standardized features:", missing_total)

    # After cleaning in prepare_data.py, there should be no missing values
    assert missing_total == 0

if __name__ == "__main__":
    test_no_missing_in_standardized_features()
    print("No-missing-values test passed.")
