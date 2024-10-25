
import pandas as pd
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path)
    # Perform basic preprocessing (e.g., handling missing values, duplicates)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df
