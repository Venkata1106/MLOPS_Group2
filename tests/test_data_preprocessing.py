import unittest
import pandas as pd
import os
import sys
import numpy as np

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from scripts.data_preprocessing import process_and_save_all_data

class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.test_input_dir = os.path.join(project_root, "tests", "test_data", "raw")
        self.test_output_dir = os.path.join(project_root, "tests", "test_data", "processed")
        
        # Create directories
        os.makedirs(self.test_input_dir, exist_ok=True)
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        # Create test data
        dates = pd.date_range(start='2024-01-01', periods=30)
        self.test_data = pd.DataFrame({
            'Date': dates,
            'Open': [100 + i + np.random.randn() for i in range(30)],
            'High': [101 + i + np.random.randn() for i in range(30)],
            'Low': [99 + i + np.random.randn() for i in range(30)],
            'Close': [100 + i + np.random.randn() for i in range(30)],
            'Volume': [1000000 + np.random.randint(-100000, 100000) for _ in range(30)]
        })
        
        # Save test data
        test_file = os.path.join(self.test_input_dir, "TEST.csv")
        print(f"Saving test data to: {test_file}")
        print(f"Test data columns: {self.test_data.columns.tolist()}")
        self.test_data.to_csv(test_file, index=False)

    def tearDown(self):
        """Clean up test files"""
        import shutil
        if os.path.exists(os.path.join(project_root, "tests", "test_data")):
            shutil.rmtree(os.path.join(project_root, "tests", "test_data"))

    def test_process_and_save_all_data(self):
        """Test the data preprocessing function"""
        try:
            # Run preprocessing
            output_folder = process_and_save_all_data(self.test_input_dir, self.test_output_dir)
            
            # Check if output file exists
            processed_file = os.path.join(output_folder, "processed_TEST.csv")
            self.assertTrue(os.path.exists(processed_file), "Processed file does not exist")
            
            # Load and verify processed data
            processed_df = pd.read_csv(processed_file)
            print(f"\nProcessed data file: {processed_file}")
            print(f"Processed data columns: {processed_df.columns.tolist()}")
            
            # Check if new columns were added
            required_columns = ['Returns', 'MA5', 'MA20']
            for col in required_columns:
                self.assertIn(col, processed_df.columns, f"Missing column: {col}")
                print(f"Verified column exists: {col}")
            
            # Verify data
            self.assertEqual(len(processed_df), len(self.test_data))
            self.assertFalse(processed_df['Returns'].isnull().all())
            
            # Print sample of processed data
            print("\nSample of processed data:")
            print(processed_df.head())
            
        except Exception as e:
            print(f"\nError in test: {str(e)}")
            raise

if __name__ == '__main__':
    unittest.main(verbosity=2)