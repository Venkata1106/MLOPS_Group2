import unittest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add project root to Python path
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, current_dir)

from scripts.data_validation import DataValidator

class TestDataValidation(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.test_data_dir = os.path.join(current_dir, "tests", "test_data")
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Create valid test data
        dates = pd.date_range(start='2024-01-01', periods=30)
        self.valid_data = pd.DataFrame({
            'Date': dates,
            'Open': [100 + i + np.random.randn() for i in range(30)],
            'High': [102 + i + np.random.randn() for i in range(30)],
            'Low': [98 + i + np.random.randn() for i in range(30)],
            'Close': [101 + i + np.random.randn() for i in range(30)],
            'Volume': [1000000 + np.random.randint(-100000, 100000) for _ in range(30)]
        })
        
        # Ensure price consistency
        for i in range(len(self.valid_data)):
            row = self.valid_data.iloc[i]
            self.valid_data.at[i, 'High'] = max(row['Open'], row['Close'], row['High'])
            self.valid_data.at[i, 'Low'] = min(row['Open'], row['Close'], row['Low'])
        
        self.valid_file = os.path.join(self.test_data_dir, "valid_stock.csv")
        self.valid_data.to_csv(self.valid_file, index=False)
        
        # Create invalid test data
        self.invalid_data = self.valid_data.copy()
        self.invalid_data.loc[5:7, 'Close'] = np.nan  # Add some NaN values
        self.invalid_data.loc[10, 'High'] = self.invalid_data.loc[10, 'Low'] - 1  # Add price inconsistency
        
        self.invalid_file = os.path.join(self.test_data_dir, "invalid_stock.csv")
        self.invalid_data.to_csv(self.invalid_file, index=False)
        
        # Initialize validator
        self.validator = DataValidator()

    def tearDown(self):
        """Clean up test files"""
        import shutil
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)

    def test_schema_validation(self):
        """Test schema validation"""
        self.validator.load_data(self.valid_file)
        results = self.validator.validate_schema()
        self.assertTrue(results['status'])
        self.assertEqual(len(results['missing_columns']), 0)

    def test_data_types(self):
        """Test data type validation"""
        self.validator.load_data(self.valid_file)
        results = self.validator.validate_data_types()
        self.assertTrue(all(r['status'] for r in results.values()))

    def test_value_ranges(self):
        """Test value range validation"""
        self.validator.load_data(self.valid_file)
        results = self.validator.validate_value_ranges()
        self.assertTrue(all(r['status'] for r in results.values()))

    def test_date_continuity(self):
        """Test date continuity validation"""
        self.validator.load_data(self.valid_file)
        results = self.validator.validate_date_continuity()
        self.assertTrue(results['status'])
        self.assertEqual(results['gaps_found'], 0)

    def test_price_consistency(self):
        """Test price consistency validation"""
        self.validator.load_data(self.valid_file)
        results = self.validator.validate_price_consistency()
        self.assertTrue(all(results.values()))

    def test_invalid_data(self):
        """Test validation with invalid data"""
        self.validator.load_data(self.invalid_file)
        results = self.validator.run_all_validations()
        self.assertFalse(results['overall_status'])

if __name__ == '__main__':
    unittest.main(verbosity=2) 