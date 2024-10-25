import unittest
import pandas as pd
from data_loader import load_data

class TestDataLoader(unittest.TestCase):

    def test_load_data(self):
        # Replace "news.csv" with the path to a sample CSV file for testing
        df = load_data("news.csv")
        
        # Check if DataFrame is not empty
        self.assertFalse(df.empty, "Loaded data should not be empty")

        # Check if there are any missing values
        self.assertFalse(df.isnull().values.any(), "Data should not contain missing values")

        # Check for duplicates
        self.assertEqual(len(df), len(df.drop_duplicates()), "Data should not contain duplicates")

if __name__ == '__main__':
    unittest.main()
