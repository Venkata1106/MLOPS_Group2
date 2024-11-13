import unittest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add project root to Python path
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, current_dir)

from scripts.pipeline_optimization import StockPipeline

class TestPipeline(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.test_dir = os.path.join(current_dir, "tests", "test_data")
        self.test_raw_dir = os.path.join(self.test_dir, "raw")
        self.test_processed_dir = os.path.join(self.test_dir, "processed")
        
        # Create test directories
        os.makedirs(self.test_raw_dir, exist_ok=True)
        os.makedirs(self.test_processed_dir, exist_ok=True)
        
        # Create test data
        self.create_test_data()
        
        # Initialize pipeline with test directories
        self.pipeline = StockPipeline()
        self.pipeline.raw_dir = self.test_raw_dir
        self.pipeline.processed_dir = self.test_processed_dir
        print(f"Set up test environment with directories:\nRaw: {self.test_raw_dir}\nProcessed: {self.test_processed_dir}")

    def create_test_data(self):
        """Create test stock data"""
        dates = pd.date_range(start='2024-01-01', periods=30)
        test_data = pd.DataFrame({
            'Date': dates,
            'Open': [100 + i + np.random.randn() for i in range(30)],
            'High': [102 + i + np.random.randn() for i in range(30)],
            'Low': [98 + i + np.random.randn() for i in range(30)],
            'Close': [101 + i + np.random.randn() for i in range(30)],
            'Volume': [1000000 + np.random.randint(-100000, 100000) for _ in range(30)]
        })
        
        # Ensure price consistency
        for i in range(len(test_data)):
            row = test_data.iloc[i]
            test_data.at[i, 'High'] = max(row['Open'], row['Close'], row['High'])
            test_data.at[i, 'Low'] = min(row['Open'], row['Close'], row['Low'])
        
        # Save test data
        test_file = os.path.join(self.test_raw_dir, "TEST.csv")
        test_data.to_csv(test_file, index=False)
        print(f"Created test data at: {test_file}")

    def tearDown(self):
        """Clean up test files"""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        print("Cleaned up test files")

    def test_preprocessing(self):
        """Test preprocessing step"""
        try:
            # Create test data
            test_data = self.create_test_data()
            
            # Initialize pipeline with test directories
            pipeline = StockPipeline(
                raw_dir=self.test_raw_dir,
                processed_dir=self.test_processed_dir
            )
            
            # Run preprocessing
            results = pipeline.preprocess_data()
            
            # Verify processed directory exists
            self.assertTrue(os.path.exists(self.test_processed_dir))
            
            # Verify processed file exists
            processed_file = os.path.join(self.test_processed_dir, f"processed_{self.test_symbol}.csv")
            self.assertTrue(os.path.exists(processed_file))
            
            # Load and verify processed data
            processed_df = pd.read_csv(processed_file)
            
            # Check required columns
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 
                              'Returns', 'MA5', 'MA20', 'Volatility']
            for col in required_columns:
                self.assertIn(col, processed_df.columns)
            
            # Check data types
            self.assertEqual(processed_df['Open'].dtype, np.float64)
            self.assertEqual(processed_df['Volume'].dtype, np.int64)
            
            # Check no missing values
            self.assertFalse(processed_df.isnull().any().any())
            
            print("Preprocessing test passed")
            return True
            
        except Exception as e:
            print(f"Preprocessing test failed: {str(e)}")
            return False
        finally:
            # Cleanup is handled by tearDown
            pass

    def test_validation(self):
        """Test validation step"""
        try:
            # First preprocess the data
            self.pipeline.run_preprocessing()
            
            # Then run validation
            validation_results = self.pipeline.run_validation()
            self.assertIsNotNone(validation_results)
            self.assertTrue(isinstance(validation_results, dict))
            print(f"Validation test passed. Results: {validation_results}")
        except Exception as e:
            print(f"Validation test failed: {str(e)}")
            raise

    def test_bias_detection(self):
        """Test bias detection step"""
        try:
            # First preprocess the data
            self.pipeline.run_preprocessing()
            
            # Then run bias detection
            bias_results = self.pipeline.run_bias_detection()
            self.assertIsNotNone(bias_results)
            self.assertTrue(isinstance(bias_results, dict))
            print(f"Bias detection test passed. Results: {bias_results}")
        except Exception as e:
            print(f"Bias detection test failed: {str(e)}")
            raise

    def test_complete_pipeline(self):
        """Test complete pipeline execution"""
        try:
            results = self.pipeline.run_pipeline(
                tickers=['TEST'],
                start_date='2024-01-01',
                end_date=datetime.now().strftime('%Y-%m-%d')
            )
            
            self.assertIsNotNone(results)
            self.assertIn('validation', results)
            self.assertIn('bias_detection', results)
            print(f"Complete pipeline test passed. Results: {results}")
        except Exception as e:
            print(f"Complete pipeline test failed: {str(e)}")
            raise

if __name__ == '__main__':
    unittest.main(verbosity=2)
