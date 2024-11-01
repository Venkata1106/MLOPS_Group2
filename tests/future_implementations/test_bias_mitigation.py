import unittest
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from scripts.bias_mitigation import BiasMitigator

class TestBiasMitigation(unittest.TestCase):
    """Test cases for bias mitigation"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data"""
        cls.test_dir = os.path.join('tests', 'test_data')
        cls.output_dir = os.path.join(cls.test_dir, 'mitigated')
        os.makedirs(cls.test_dir, exist_ok=True)
        os.makedirs(cls.output_dir, exist_ok=True)
        
        # Create test data
        cls.create_test_data()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files"""
        try:
            for file in os.listdir(cls.output_dir):
                os.remove(os.path.join(cls.output_dir, file))
            for file in os.listdir(cls.test_dir):
                os.remove(os.path.join(cls.test_dir, file))
            os.rmdir(cls.output_dir)
            os.rmdir(cls.test_dir)
        except Exception as e:
            print(f"Error in cleanup: {str(e)}")
    
    @classmethod
    def create_test_data(cls):
        """Create synthetic stock data with known biases"""
        # Generate dates with gaps
        dates = pd.date_range(start='2024-01-01', periods=100, freq='B')
        dates = dates.drop(dates[10:15])  # Create gap
        
        # Generate biased price data
        np.random.seed(42)
        close_prices = 100 + np.random.randn(len(dates)).cumsum()
        
        # Add some outliers
        close_prices[30] *= 1.5  # Price spike
        close_prices[60] *= 0.5  # Price drop
        
        # Generate volume data with bias
        volume = np.random.randint(1000, 2000, len(dates))
        volume[30:40] *= 3  # Volume bias period
        
        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Open': close_prices * 0.99,
            'High': close_prices * 1.02,
            'Low': close_prices * 0.98,
            'Close': close_prices,
            'Volume': volume
        })
        
        # Save test data
        cls.test_file = os.path.join(cls.test_dir, 'test_stock.csv')
        df.to_csv(cls.test_file, index=False)
    
    def test_initialization(self):
        """Test BiasMitigator initialization"""
        mitigator = BiasMitigator()
        self.assertIsNotNone(mitigator)
    
    def test_handle_missing_dates(self):
        """Test missing dates handling"""
        mitigator = BiasMitigator()
        df = pd.read_csv(self.test_file)
        result = mitigator.handle_missing_dates(df)
        
        # Check if gaps are filled
        self.assertTrue(len(result) > len(df))
        self.assertFalse(result['Close'].isnull().any())
    
    def test_normalize_features(self):
        """Test feature normalization"""
        mitigator = BiasMitigator()
        df = pd.read_csv(self.test_file)
        result = mitigator.normalize_features(df)
        
        # Check normalized columns
        self.assertIn('Volume_Normalized', result.columns)
        self.assertIn('Price_Normalized', result.columns)
    
    def test_add_technical_indicators(self):
        """Test technical indicators calculation"""
        mitigator = BiasMitigator()
        df = pd.read_csv(self.test_file)
        result = mitigator.add_technical_indicators(df)
        
        # Check technical indicators
        for window in [5, 10, 20, 50]:
            self.assertIn(f'MA_{window}', result.columns)
        self.assertIn('RSI', result.columns)
        self.assertIn('MACD', result.columns)
    
    def test_handle_outliers(self):
        """Test outlier handling"""
        mitigator = BiasMitigator()
        df = pd.read_csv(self.test_file)
        columns = ['Close', 'Volume']
        result = mitigator.handle_outliers(df, columns)
        
        # Check if outliers are mitigated
        for col in columns:
            original_std = df[col].std()
            mitigated_std = result[col].std()
            self.assertLess(mitigated_std, original_std * 1.1)
    
    def test_full_mitigation_process(self):
        """Test complete bias mitigation process"""
        mitigator = BiasMitigator()
        results = mitigator.mitigate_bias(self.test_file, self.output_dir)
        
        # Check results
        self.assertEqual(results['status'], 'success')
        self.assertIn('statistics', results)
        self.assertTrue(os.path.exists(results['output_path']))
        
        # Load and check mitigated data
        mitigated_df = pd.read_csv(results['output_path'])
        self.assertGreater(len(mitigated_df.columns), 6)  # More features added
        self.assertGreater(len(mitigated_df), 80)  # Dates filled

if __name__ == '__main__':
    unittest.main()
