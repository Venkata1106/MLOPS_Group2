import unittest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add project root to Python path
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, current_dir)

from scripts.bias_detection import BiasAnalyzer

class TestBiasDetection(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.test_data_dir = os.path.join(current_dir, "tests", "test_data")
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Create sample test data
        dates = pd.date_range(start='2024-01-01', periods=30)
        self.test_data = pd.DataFrame({
            'Date': dates,
            'Close': [100 + i + np.random.randn() for i in range(30)],
            'Volume': [1000000 + np.random.randint(-100000, 100000) for _ in range(30)],
            'Open': [99 + i + np.random.randn() for i in range(30)],
            'High': [101 + i + np.random.randn() for i in range(30)],
            'Low': [98 + i + np.random.randn() for i in range(30)]
        })
        
        # Add some NaN values to test bias detection
        self.test_data.loc[5:7, 'Close'] = np.nan
        self.test_data.loc[15:16, 'Volume'] = np.nan
        
        self.test_file = os.path.join(self.test_data_dir, "test_stock.csv")
        self.test_data.to_csv(self.test_file, index=False)
        
        # Initialize analyzer
        self.analyzer = BiasAnalyzer()
        print(f"Created test file at: {self.test_file}")
        print(f"Test data shape: {self.test_data.shape}")

    def tearDown(self):
        """Clean up test files"""
        import shutil
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)
        print("Cleaned up test files")

    def test_load_data(self):
        """Test data loading"""
        self.analyzer.load_data(self.test_file)
        self.assertIsNotNone(self.analyzer.data)
        self.assertEqual(len(self.analyzer.data), len(self.test_data))
        print(f"Loaded data shape: {self.analyzer.data.shape}")

    def test_sampling_bias(self):
        """Test sampling bias detection"""
        self.analyzer.load_data(self.test_file)
        results = self.analyzer.check_sampling_bias()
        print(f"Sampling bias results: {results}")
        self.assertIn('missing_values', results)
        self.assertIn('data_coverage', results)
        self.assertIn('date_range', results)
        self.assertEqual(results['data_coverage'], 30)  # Should match our test data size

    def test_survivorship_bias(self):
        """Test survivorship bias detection"""
        self.analyzer.load_data(self.test_file)
        results = self.analyzer.check_survivorship_bias()
        print(f"Survivorship bias results: {results}")
        self.assertIn('total_records', results)
        self.assertIn('missing_data_points', results)
        self.assertIn('data_completeness', results)
        self.assertEqual(results['total_records'], 30)  # Should match our test data size

    def test_time_period_bias(self):
        """Test time period bias detection"""
        self.analyzer.load_data(self.test_file)
        results = self.analyzer.check_time_period_bias()
        print(f"Time period bias results: {results}")
        self.assertIn('time_span', results)
        self.assertIn('trading_days', results)
        self.assertIn('gaps', results)
        self.assertEqual(results['trading_days'], 30)  # Should match our test data size

if __name__ == '__main__':
    unittest.main(verbosity=2)