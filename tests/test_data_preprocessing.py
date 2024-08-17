
import sys
sys.path.append('/content/asthma-diagnosis/src')

import unittest
import pandas as pd
import numpy as np
from data_preprocessing import drop_columns, standardize_columns, check_missing_values, preprocess_data

class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        self.df = pd.DataFrame({
            'patient_id': [1, 2, 3],
            'age': [25, 35, 45],
            'bmi': [22.4, 27.5, 30.2],
            'gender': [0, 1, 1],
            'physical_activity': [3.5, 2.0, 0.5],
            'diet_quality': [7, 6, 5],
            'sleep_quality': [8, 7, 6],
            'lung_function_fev1': [2.1, 3.3, 2.9],
            'lung_function_fvc': [3.8, 4.1, 3.9]
        })

    def test_drop_columns(self):
        df_dropped = drop_columns(self.df, ['patient_id'])
        self.assertNotIn('patient_id', df_dropped.columns)
        self.assertEqual(len(df_dropped.columns), 8)  # Ensure only one column was dropped

    def test_standardize_columns(self):
        df_standardized = standardize_columns(self.df, ['age', 'bmi'])
        print("Standard deviation of 'age':", df_standardized['age'].std())
        print("Standard deviation of 'bmi':", df_standardized['bmi'].std())
        # Adjusted the test to check against 1.2247 due to small sample size
        self.assertTrue(np.allclose(df_standardized['age'].std(), 1.2247, atol=1e-4))
        self.assertTrue(np.allclose(df_standardized['bmi'].std(), 1.2247, atol=1e-4))
        self.assertTrue(np.allclose(df_standardized['age'].mean(), 0, atol=1e-7))
        self.assertTrue(np.allclose(df_standardized['bmi'].mean(), 0, atol=1e-7))

    def test_check_missing_values(self):
        df_with_nan = self.df.copy()
        df_with_nan.iloc[0, 1] = np.nan  # Introduce a NaN value
        missing = check_missing_values(df_with_nan)
        self.assertEqual(missing['age'], 1)
        self.assertEqual(missing['bmi'], 0)

    def test_preprocess_data(self):
        processed_df = preprocess_data(self.df)
        self.assertNotIn('patient_id', processed_df.columns)
        self.assertTrue(np.allclose(processed_df['age'].mean(), 0, atol=1e-7))
        self.assertTrue(np.allclose(processed_df['bmi'].mean(), 0, atol=1e-7))
        self.assertEqual(processed_df.isnull().sum().sum(), 0)  # Ensure no missing values

if __name__ == '__main__':
    unittest.main()
