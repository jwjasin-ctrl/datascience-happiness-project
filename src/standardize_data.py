# standardize_data.py 
# This script:
# 1) loads the cleaned dataset
# 2) renames the 6 explanatory variables to shorter names
# 3) standardizes them with StandardScaler (z-scores)
# 4) saves the standardized data for clustering

import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1) load cleaned data

CLEAN_FILE = "results/clean_happiness_data.csv"

def main():
    print(">>> STANDARDIZE_DATA.PY IS RUNNING <<<")
    
    df = pd.read_csv(CLEAN_FILE)
    
    print("Original columns:", list(df.columns))
    
# 2) Rename the 6 explanatory variables to shorter names
    rename_map = {
        "Explained by: Log GDP per capita": "log_GDP",
        "Explained by: Social support": "Social_Support",
        "Explained by: Healthy life expectancy": "Life_expectancy",
        "Explained by: Freedom to make life choices": "Freedom",
        "Explained by: Generosity": "Generosity",
        "Explained by: Perceptions of corruption": "Corruption",
    }
    
    df = df.rename(columns=rename_map)
    
    print("\nRenamed columns:", list(df.columns))
    
# 3) Standarize the 6 variables by using StandardScaler
    feature_cols = [
        "log_GDP",
        "Social_Support",
        "Life_expectancy",
        "Freedom",
        "Generosity",
        "Corruption",
    ]
    
    scaler = StandardScaler()
    X_std = scaler.fit_transform(df[feature_cols])
    
    #Put standardied values into a new DataFrame
    df_std = pd.DataFrame(X_std, columns=[col + "_std" for col in feature_cols])
    
# 4) Combine with country name and ladder score for later analysis
    result_df = pd.concat(
        [
            df[["Country name", "Ladder score"]],
            df[feature_cols],
            df_std,
        ],
        axis=1,
    )
    
    print("\nStandardized data shape:", result_df.shape)
    print("\nFirst 5 rows of standardized data:")
    print(result_df.head())
    
# Save for clustering
    result_df.to_csv("results/happiness_standardized.csv", index=False)
    print("\nSaved standardized data to: results/happiness_standardized.csv")
    
if __name__ == "__main__":
    main()