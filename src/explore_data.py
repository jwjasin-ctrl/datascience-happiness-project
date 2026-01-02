# explore_data.py
# my 1st EDA step: load the file, check that reading works, list all variables

import pandas as pd

# path to my World Happiness Report 2024 file
DATA_FILE = "data/world-happiness-2024.csv"

def main():
    print(">>> EXPLORE_DATA.PY IS RUNNING <<<")

    # Load the dataset
    df = pd.read_csv(DATA_FILE, sep=";", decimal=",")

    # show how many rows (countries) and columns (variables) I have
    print("Data shape (rows, columns):", df.shape)

    # print the names of all columns
    # to check the exact spelling of each variable
    print("\n=== columns in the dataset ===")
    for col in df.columns:
        print(col)
        
        #Examples of important columns:
        # Country name
        # Ladder score (happiness score 0-10)
        # Log GDP per capita (GDP contribution)
        # social support
        # Healthy life expectancy
        # Freedom to make life choices
        # Generocity
        # Perceptions of corruption
        

if __name__ == "__main__":
    main()
