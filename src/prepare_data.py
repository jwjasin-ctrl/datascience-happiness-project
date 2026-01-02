# prepare_data.py
# Loads the full 2024 dataset
# Selects country name and ladder score and the 6 variables
# creates a clean DataFrame
# Saves it to results/clean_happiness_data.csv
# Prints a preview
# This script prepares a clean dataset for clustering.
# It selects only the variables needed for the analysis.
# Drops countries with missing values

import pandas as pd

# Path to the original 2024 dataset
DATA_FILE = "data/world-happiness-2024.csv"
OUT_FILE = "results/clean_happiness_data.csv"


def main():
    print(">>> PREPARE_DATA.PY IS RUNNING <<<")

    # Load the full dataset with correct separators
    df = pd.read_csv(DATA_FILE, sep=";", decimal=",")

    # Select the variables needed for the project
    selected_columns = [
        "Country name",
        "Ladder score",  # happiness score (NOT used in clustering)
        "Explained by: Log GDP per capita",
        "Explained by: Social support",
        "Explained by: Healthy life expectancy",
        "Explained by: Freedom to make life choices",
        "Explained by: Generosity",
        "Explained by: Perceptions of corruption",
    ]
    
    # Keep only these columns
    clean_df = df[selected_columns].copy()

    # Show the shape to verify it worked
    print("Clean dataset shape BEFORE dropping missing values:", clean_df.shape)
    
    # Drop any country with missing values in these columns
    rows_before = len(clean_df)
    clean_df = clean_df.dropna(subset=selected_columns).copy()
    rows_after = len(clean_df)
    
    print(f"Rows before dropping NaNs: {rows_before}")
    print(f"Rows after dropping NaNs: {rows_after}")
    print(f"Dropped {rows_before - rows_after} countries with missing values.")

    # Show the shape after dropping
    print("Final clean dataset shape:", clean_df.shape)

    # Preview the first few rows
    print("\nFirst 5 rows of cleaned dataset:")
    print(clean_df.head())

    # Save the cleaned dataset into the results folder
    clean_df.to_csv(OUT_FILE, index=False)
    print(f"\nSaved cleaned dataset to: {OUT_FILE}")


if __name__ == "__main__":
    main()
