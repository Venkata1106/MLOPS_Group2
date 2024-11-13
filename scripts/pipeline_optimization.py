import os
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any, Optional
import sys

# Add project root to Python path
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

# Use absolute imports matching your implementations
from scripts.data_acquisition import fetch_stock_data
from scripts.data_preprocessing import process_and_save_all_data
from scripts.data_validation import DataValidator
from scripts.bias_detection import BiasAnalyzer
from scripts.anomaly_detection import AnomalyDetector

class StockPipeline:
    """Pipeline for stock data processing and optimization"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize pipeline with configuration"""
        self.config = config or {}
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Set default paths
        self.data_dir = os.path.join(self.project_root, 'data')
        self.raw_dir = os.path.join(self.data_dir, 'raw')
        self.processed_dir = os.path.join(self.data_dir, 'processed')
        
        # Create directories if they don't exist
        for directory in [self.data_dir, self.raw_dir, self.processed_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized StockPipeline with directories:\n"
                        f"Raw: {self.raw_dir}\n"
                        f"Processed: {self.processed_dir}")

    def run_acquisition(self, tickers: List[str], 
                       start_date: Optional[str] = None) -> None:
        """Run data acquisition step"""
        self.logger.info(f"Starting data acquisition for tickers: {tickers}")
        
        try:
            fetch_stock_data(tickers, start_date, self.raw_dir)
        except Exception as e:
            self.logger.error(f"Data acquisition failed: {str(e)}")
            raise

    def run_preprocessing(self) -> None:
        """Run data preprocessing step"""
        self.logger.info("Starting data preprocessing...")
        
        try:
            process_and_save_all_data(self.raw_dir, self.processed_dir)
            self.logger.info(f"Data preprocessing completed. Results saved to {self.processed_dir}")
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {str(e)}")
            raise

    def run_validation(self) -> Dict:
        """Run data validation step"""
        self.logger.info("Starting data validation...")
        
        try:
            validation_results = {}
            for file in os.listdir(self.processed_dir):
                if file.startswith('processed_'):
                    file_path = os.path.join(self.processed_dir, file)
                    validator = DataValidator(self.processed_dir)
                    validator.load_data()
                    validation_results[file] = validator.run_all_validations()
            
            self.logger.info("Data validation completed")
            return validation_results
        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            raise

    def run_bias_detection(self) -> Dict:
        """Run bias detection step"""
        self.logger.info("Starting bias detection...")
        
        try:
            bias_results = {}
            for file in os.listdir(self.processed_dir):
                if file.startswith('processed_'):
                    # Initialize the BiasAnalyzer with the input folder
                    analyzer = BiasAnalyzer(self.processed_dir)  # Pass the processed_dir to the constructor
                    analyzer.load_data()  # Call load_data without arguments
                    bias_results[file] = analyzer.generate_report()
            
            self.logger.info("Bias detection completed")
            return bias_results
        except Exception as e:
            self.logger.error(f"Bias detection failed: {str(e)}")
            raise

    def run_anomaly_detection(self) -> Dict:
        """Run anomaly detection step"""
        self.logger.info("Starting anomaly detection...")
        
        try:
            anomaly_dir = os.path.join(self.data_dir, 'anomalies')
            os.makedirs(anomaly_dir, exist_ok=True)
            
            anomaly_results = {}
            # Initialize the AnomalyDetector
            detector = AnomalyDetector()  # Create an instance of AnomalyDetector
            
            # Load data from the processed directory
            detector.load_data(self.processed_dir)  # Load all processed data
            
            # Detect anomalies
            results = detector.detect_anomalies()  # Detect anomalies from the loaded data
            
            # Save results
            detector.save_results(results, anomaly_dir)  # Save the results to the anomaly directory
            
            anomaly_results['anomalies'] = results
            
            self.logger.info("Anomaly detection completed")
            return anomaly_results
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {str(e)}")
            raise

    def run_pipeline(self, tickers: List[str], 
                    start_date: Optional[str] = None, 
                    end_date: Optional[str] = None) -> Dict:
        """Run complete pipeline"""
        self.logger.info(f"Starting pipeline execution for tickers: {tickers}")
        
        try:
            # Step 1: Data Acquisition
            self.run_acquisition(tickers, start_date)
            
            # Step 2: Data Preprocessing
            self.run_preprocessing()
            
            # Step 3: Data Validation
            validation_results = self.run_validation()
            
            # Step 4: Bias Detection
            bias_results = self.run_bias_detection()
            
            # Step 5: Anomaly Detection
            anomaly_results = self.run_anomaly_detection()
            
            pipeline_results = {
                'validation': validation_results,
                'bias_detection': bias_results,
                'anomaly_detection': anomaly_results
            }
            
            self.logger.info("Pipeline execution completed successfully")
            return pipeline_results
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    pipeline = StockPipeline()
    results = pipeline.run_pipeline(
        tickers=['AAPL', 'MSFT'],
        start_date='2024-01-01',
        end_date=datetime.now().strftime('%Y-%m-%d')
    )
    print("Pipeline Results:", results)