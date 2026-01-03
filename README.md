# Explaining and Clustering Global Happiness Project

Author: Julia Jasinska
Course: Data Science & Advanced Programming

## Run without conda (recommended fallback)

### macOS / Linux
```bash
git clone https://github.com/jwjasin-ctrl/datascience-happiness-project.git
cd datascience-happiness-project

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

python run_all.py
```

### Windows 
```bash
git clone https://github.com/jwjasin-ctrl/datascience-happiness-project.git
cd datascience-happiness-project

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

python run_all.py
```

#### 1. Project Overview

This project uses the World Happiness Report 2024 dataset to identify groups of countries
with similar drivers of happiness. I focus on six explanatory factors:

- Log GDP per capita  
- Social support  
- Healthy life expectancy  
- Freedom to make life choices  
- Generosity  
- Perceptions of corruption  

Using these drivers, I:

1. Clean and standardize the data,
2. Apply K-Means clustering to group countries,
3. Use PCA for 2D visualization of the clusters,
4. Regress happiness on GDP to study happiness beyond income (residuals),
5. Check the robustness of clusters when GDP is excluded,
6. Summarize and visualize the cluster profiles.

All code is written in Python as part of the Data Science & Advanced Programming course.


## 2. Repository structure
'''text
datascience-happiness-project/
├── README.md                         # Setup and usage instructions
├── environment.yml                   # Conda dependencies 
├── run_all.py                        # small runner that executes the full pipeline
├── data/
│   └── world-happiness-2024.csv      # raw input data
├── results/
│   ├── clean_happiness_data.csv           # cleaned subset used for the project
│   ├── happiness_standardized.csv         # standardized drivers (z-scores)
│   ├── cluster_assignments.csv            # country → cluster (K-Means with GDP)
│   ├── cluster_profiles.csv               # mean standardized drivers per cluster
│   ├── cluster_summary.csv                # cluster sizes and mean happiness
│   ├── factor_summary.csv                 # summary stats for six drivers
│   ├── factor_correlations.csv            # correlation matrix of drivers
│   ├── cluster_assignments_with_resid.csv # clusters + residual happiness
│   ├── cluster_residuals_summary.csv      # residual happiness by cluster
│   ├── cluster_assignments_no_gdp.csv     # clustering without GDP
│   ├── cluster_confusion_no_gdp.csv       # comparison full vs no-GDP clusters
│   ├── pca_clusters.png                   # PCA plot with clusters (Figure 2 in the report)
│   ├── cluster_profiles_bars.png          # bar chart of mean z-scores per cluster (Figure 6 in the report)
│   ├── residuals_boxplot.png              # boxplot of residual happiness by cluster (Figure 8 in the report)
│   ├── gdp_happiness_scatter_labeled.png  # happiness vs log GDP with clusters and labels (Figure 3 in the report)
│   ├── reginteractions_coeff.csv          # table of standardized regression coefficients (model with interactions)
│   ├── reginteractions_coeff.png          # horizontal bar chart of standardized regression coefficients (Figure 7 in the report) 
│   ├── factor_corr_heatmap.png            # Correlation heatmap of the six standardized happiness drivers (Figure 1 in the report)
│   ├── gdp_generosity_clusters.png        # scatter: log GDP per capita vs generosity (Figure 5 in the report)
│   └── gdp_lifeexpectancy_clusters.png    # scatter: log GDP per capita vs healthy life expectancy (Figure 4 in the report)
├── src/
│   ├── load_data.py                  # uploading the data
│   ├── explore_data.py               # load the file, check that reading works, list all variables
│   ├── prepare_data.py               # select variables, rename, save clean CSV
│   ├── standardize_data.py           # compute z-scores for the six drivers
│   ├── explore_factors.py            # summary stats + correlation matrix
│   ├── run_kmeans.py                 # K-Means for K=3..6, choose best K by silhouette
│   ├── analyze_clusters.py           # cluster sizes, mean happiness, mean drivers
│   ├── pca_clusters.py               # PCA + 2D scatter plot of clusters
│   ├── gdp_residuals.py              # regress happiness on log GDP, compute residuals
│   ├── plot_cluster_profiles.py      # bar chart of cluster mean z-scores
│   ├── plot_residuals_boxplot.py     # boxplot of residual happiness by cluster
│   ├── plot_gdp_happiness.py         # scatter: happiness vs log GDP with regression line
│   ├── run_kmeans_no_gdp.py          # robustness: clustering without GDP and confusion matrix
│   ├── compare_clusters.py           # Compare baseline clusters with clusters from model without GDP 
│   ├── reginteractions.py            # Regression with interactions & corruption dummy
│   └── interaction_gdp_pairs.py      # plots GDP–life expectancy and GDP-generosity scatter plots by cluster
└── tests/
    ├── test_standardization.py          # check means≈0 and std≈1 (with tolerance)
    ├── test_no_missing_std_features.py  # ensure no NaNs in features used for K-Means
    ├── test_best_k.py                   # confirm K=3 has best silhouette among {3,4,5,6}
    ├── test_kmeans_labels.py            # Check that there are exactly 3 distinct clusters in the baseline solution
    └── test_pca_var.py                  # check first two PCs explain ≥ 60% variance
'''