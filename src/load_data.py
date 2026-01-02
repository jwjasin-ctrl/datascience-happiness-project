# load_data.py
# uploading the data and checking if it works

import pandas as pd

DATA_FILE = "data/world-happiness-2024.csv"

def main () :
    """Load the csv file and show basic info."""
    # read the Excel file into a pandas DataFrame
    df = pd.read_csv(DATA_FILE, sep=";", decimal=",")
    
    # print number of rows and columns
    print("Data shape (rows, columns):", df.shape)
    
    # show the first 5 rows
    print("\nFirst 5 rows")
    print(df.head())
    
if __name__ == "__main__":
    main()
