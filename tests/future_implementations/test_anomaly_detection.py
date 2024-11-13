import unittest
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from scripts.anomaly_detection import AnomalyDetector, run_anomaly_detection

class TestAnomalyDetection(unittest.TestCase):
    """Test cases for anomaly detection"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data"""
        cls.test_dir = os.path.join('tests', 'test_data')
        cls.output_dir = os.path.join(cls.test_dir, 'anomalies')
        os.makedirs(cls.test_dir, exist_ok=True)
        os.makedirs(cls.output_dir, exist_ok=True)
        
        # Create test data
        cls.create_test_data()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files"""
        try:
            # Remove test files
            for file in os.listdir(cls.output_dir):
                file_path = os.path.join(cls.output_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            for file in os.listdir(cls.test_dir):
                file_path = os.path.join(cls.test_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                
            # Remove directories
            if os.path.exists(cls.output_dir):
                os.rmdir(cls.output_dir)
            if os.path.exists(cls.test_dir):
                os.rmdir(cls.test_dir)
        except Exception as e:
            print(f"Error in cleanup: {str(e)}")
    
    @classmethod
    def create_test_data(cls):
        """Create synthetic stock data with known anomalies"""
        # Generate dates
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Generate normal price data
        np.random.seed(42)
        close_prices = 100 + np.random.randn(100).cumsum()
        
        # Add some known anomalies
        close_prices[30] += 10  # Price spike
        close_prices[60] -= 8   # Price drop
        
        # Generate other price data
        high_prices = close_prices + np.random.rand(100)
        low_prices = close_prices - np.random.rand(100)
        open_prices = close_prices - np.random.rand(100) * 0.5
        
        # Generate volume data with anomalies
        volume = np.random.randint(1000, 2000, 100)
        volume[30] *= 3  # Volume spike
        volume[60] *= 4  # Volume spike
        
        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volume
        })
        
        # Save test data
        cls.test_file = os.path.join(cls.test_dir, 'test_stock.csv')
        df.to_csv(cls.test_file, index=False)
    
    def test_initialization(self):
        """Test AnomalyDetector initialization"""
        detector = AnomalyDetector(contamination=0.1)
        self.assertEqual(detector.contamination, 0.1)
        self.assertIsNotNone(detector.model)
    
    def test_feature_calculation(self):
        """Test feature calculation"""
        detector = AnomalyDetector()
        df = pd.read_csv(self.test_file)
        features = detector.calculate_features(df)
        
        # Check if all expected features are present
        expected_features = ['Returns', 'Price_Volatility', 'Volume_Change', 
                           'Volume_MA_Ratio', 'Price_Range']
        for feature in expected_features:
            self.assertIn(feature, features.columns)
        
        # Check if features are properly calculated
        self.assertFalse(features['Returns'].isnull().any())
        self.assertFalse(features['Volume_Change'].isnull().any())
    
    def test_anomaly_detection(self):
        """Test anomaly detection process"""
        detector = AnomalyDetector()
        results = detector.detect_anomalies(self.test_file, self.output_dir)
        
        # Check results structure
        self.assertEqual(results['status'], 'success')
        self.assertIn('statistics', results)
        self.assertIn('output_path', results)
        
        # Check statistics
        stats = results['statistics']
        self.assertEqual(stats['total_samples'], 100)
        self.assertGreater(stats['anomaly_count'], 0)
        self.assertLess(stats['anomaly_percentage'], 20)  # Should be less than 20%
        
        # Check output file
        self.assertTrue(os.path.exists(results['output_path']))
        output_df = pd.read_csv(results['output_path'])
        self.assertIn('Is_Anomaly', output_df.columns)
    
    def test_convenience_function(self):
        """Test the convenience function"""
        results = run_anomaly_detection(self.test_file, self.output_dir)
        self.assertEqual(results['status'], 'success')
        self.assertIn('statistics', results)
    
    def test_error_handling(self):
        """Test error handling"""
        detector = AnomalyDetector()
        
        # Test with non-existent file
        results = detector.detect_anomalies("nonexistent_file.csv")
        self.assertEqual(results['status'], 'error')
        self.assertIn('error_message', results)
        
        # Test with invalid data
        invalid_file = os.path.join(self.test_dir, 'invalid.csv')
        pd.DataFrame({'A': [1, 2, 3]}).to_csv(invalid_file)
        results = detector.detect_anomalies(invalid_file)
        self.assertEqual(results['status'], 'error')

if __name__ == '__main__':
    unittest.main()
