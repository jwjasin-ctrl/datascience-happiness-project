# explore_factors.py
# prints summary stats and saves factor_summary.csv
# prints correlation matrix of the six explanatory variables
# saves factor_correlations.csv
# builds the heatmap (factor_corr_heatmap.png)


import pandas as pd
import matplotlib.pyplot as plt

# Use the cleaned dataset (with the original long names)
DATA_FILE = "results/happiness_standardized.csv"

def main():
    print(">>> EXPLORE_FACTORS.PY IS RUNNING <<<")
    
    # 1) Load standardized data
    df = pd.read_csv(DATA_FILE)
    
    print("\nAvailable columns in the file:")
    print(list(df.columns))
    
    # Columns for the six standardized drivers
    factor_cols = [
        "log_GDP_std",
        "Social_Support_std",
        "Life_expectancy_std",
        "Freedom_std",
        "Generosity_std",
        "Corruption_std",
    ]
    print("\nFactor columns used after renaming:")
    print(factor_cols)
    
    # 2) Summary statistics (mean, std, min, max, etc)
    summary = df[factor_cols].describe().T # transpose for nicer view
    print("\nSummary statistics for the six factors:")
    print(summary)
    
    # Save for the report in "results" file
    summary.to_csv("results/factor_summary.csv")
    print("\nSaved factor summary to: results/factor_summary.csv")
    
    # 3) Correlation matrix
    corr = df[factor_cols].corr()
    print("\nCorrelation matrix of the six factors:")
    print(corr)
    corr.to_csv("results/factor_correlations.csv")
    print("Saved factor correlations to: results/factor_correlations.csv")
    
    # 4) Correlation heatmap figure
    # figsize=(8, 6) sets the size of the output image in inches
    fig, ax = plt.subplots(figsize=(8, 6))

    # Show the correlation matrix as a coloured image
    # vmin=-1 and vmax=1 fix the colour scale to the full correlation range [-1, 1]
    # "coolwarm" - colormap that highlights negative vs positive values
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")

    # Put tick marks on all rows/columns of the matrix
    ax.set_xticks(range(len(factor_cols)))
    ax.set_yticks(range(len(factor_cols)))
    
    # Label the ticks with the factor name
    # rotation=45 and ha="right" make the x-labels readable
    ax.set_xticklabels(factor_cols, rotation=45, ha="right")
    ax.set_yticklabels(factor_cols)

    # Add a colour bar on the side that shows the mapping from colour to correlation value
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation")

    # Add a descriptive title for the figure
    ax.set_title("Correlation heatmap of happiness drivers (standardized)")

    # Tighten the layout so labels and title are not cut off in the saved image
    plt.tight_layout()
    
    # Save the figure to the results folder (300 dpi = high resolution)
    plt.savefig("results/factor_corr_heatmap.png", dpi=300)
    plt.close()
    
    print("Saved factor correlation heatmap to: results/factor_corr_heatmap.png")
    
    
if __name__ == "__main__":
    main()
