# run_all.py
# this script is to run the full analysis pipeline for the World Happiness clustering project.


import subprocess
from pathlib import Path

# project root = folder where this file lives
PROJECT_ROOT = Path(__file__).resolve().parent


def run_step(description, script_rel_path):
    """Run one Python script and print a header."""
    print("\n" + "=" * 100)
    print(f"STEP: {description}")
    print(f"Running: python {script_rel_path}")
    print("=" * 100)

    # build the full path so we can run from project root
    script_full_path = PROJECT_ROOT / script_rel_path
    subprocess.run(["python", str(script_full_path)], check=True)


def main():
  
    steps = [
        ("Explore dataset",                                 "src/explore_data.py"),
        ("Prepare cleaned dataset",                         "src/prepare_data.py"),
        ("Standardize happiness drivers",                   "src/standardize_data.py"),
        ("Explore factors (summary & corr)",                "src/explore_factors.py"),
        ("Run K-Means clustering",                          "src/run_kmeans.py"),
        ("Summarize clusters",                              "src/analyze_clusters.py"),
        ("PCA + cluster visualization",                     "src/pca_clusters.py"),
        ("Regression & residual happiness",                 "src/gdp_residuals.py"),
        ("Plot cluster profiles (bar chart)",               "src/plot_cluster_profiles.py"),
        ("Plot residual happiness boxplot",                 "src/plot_residuals_boxplot.py"),
        ("Plot GDP vs happiness scatter",                   "src/plot_gdp_happiness.py"),
        ("Robustness: clustering without GDP",              "src/run_kmeans_no_gdp.py"),
        ("Regression with interactions & corruption dummy", "src/reginteractions.py"),
        ("Interactions of log GDP with Life expectancy and Generosity", "src/interaction_gdp_pairs.py"),
    ]

    for desc, script in steps:
        try:
            run_step(desc, script)
        except FileNotFoundError:
            print(
                f"WARNING: Script not found: {script}. "
            )
        except subprocess.CalledProcessError:
            print(f"\nERROR while running {script}. Stopping pipeline.")
            break


if __name__ == "__main__":
    main()