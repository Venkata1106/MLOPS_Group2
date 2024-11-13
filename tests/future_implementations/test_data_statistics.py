import unittest
import pandas as pd
import numpy as np
import os
from datetime import datetime
from scripts.data_statistics import StockAnalyzer

class TestStockAnalyzer(unittest.TestCase):
    """Test cases for stock data analysis"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data"""
        cls.test_dir = os.path.join('tests', 'test_data')
        cls.output_dir = os.path.join(cls.test_dir, 'analyzed')
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
        """Create synthetic stock data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='B')
        
        np.random.seed(42)
        close_prices = 100 + np.random.randn(100).cumsum()
        
        df = pd.DataFrame({
            'Date': dates,
            'Open': close_prices * 0.99,
            'High': close_prices * 1.02,
            'Low': close_prices * 0.98,
            'Close': close_prices,
            'Volume': np.random.randint(1000, 2000, 100)
        })
        
        cls.test_file = os.path.join(cls.test_dir, 'test_stock.csv')
        df.to_csv(cls.test_file, index=False)
    
    def test_initialization(self):
        """Test StockAnalyzer initialization"""
        analyzer = StockAnalyzer()
        self.assertIsNotNone(analyzer)
    
    def test_returns_metrics(self):
        """Test returns metrics calculation"""
        analyzer = StockAnalyzer()
        df = pd.read_csv(self.test_file)
        metrics = analyzer.calculate_returns_metrics(df)
        
        self.assertIn('daily_returns', metrics)
        self.assertIn('annualized_return', metrics)
        self.assertIn('sharpe_ratio', metrics)
    
    def test_price_metrics(self):
        """Test price metrics calculation"""
        analyzer = StockAnalyzer()
        df = pd.read_csv(self.test_file)
        metrics = analyzer.calculate_price_metrics(df)
        
        self.assertIn('price_stats', metrics)
        self.assertIn('price_momentum', metrics)
        self.assertIn('price_trends', metrics)
    
    def test_volume_metrics(self):
        """Test volume metrics calculation"""
        analyzer = StockAnalyzer()
        df = pd.read_csv(self.test_file)
        metrics = analyzer.calculate_volume_metrics(df)
        
        self.assertIn('volume_stats', metrics)
        self.assertIn('volume_trends', metrics)
    
    def test_technical_indicators(self):
        """Test technical indicators calculation"""
        analyzer = StockAnalyzer()
        df = pd.read_csv(self.test_file)
        result = analyzer.calculate_technical_indicators(df)
        
        self.assertIn('MA_20', result.columns)
        self.assertIn('BB_middle', result.columns)
        self.assertIn('RSI', result.columns)
    
    def test_correlation_metrics(self):
        """Test correlation metrics calculation"""
        analyzer = StockAnalyzer()
        df = pd.read_csv(self.test_file)
        metrics = analyzer.calculate_correlation_metrics(df)
        
        self.assertIn('price_volume_corr', metrics)
        self.assertIn('returns_volume_corr', metrics)
    
    def test_full_analysis(self):
        """Test complete analysis process"""
        analyzer = StockAnalyzer()
        results = analyzer.analyze_stock_data(self.test_file, self.output_dir)
        
        self.assertEqual(results['status'], 'success')
        self.assertIn('statistics', results)
        self.assertTrue(os.path.exists(results['statistics'].get('output_path', '')))

if __name__ == '__main__':
    unittest.main()
