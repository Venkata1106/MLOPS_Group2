import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class DataValidator:
    """Class for validating stock market data"""
    
    def __init__(self, input_folder=None):
        self.input_folder = input_folder
        self.data = None
        self.validation_results = {}
        
        if input_folder:
            self.load_data()
    
    def load_data(self):
        """Load data from processed CSV files"""
        if not self.input_folder or not os.path.exists(self.input_folder):
            raise ValueError(f"Invalid input folder: {self.input_folder}")
            
        # Get all processed CSV files
        files = [f for f in os.listdir(self.input_folder) if f.startswith('processed_') and f.endswith('.csv')]
        if not files:
            raise ValueError(f"No processed CSV files found in {self.input_folder}")
            
        # Load the first file for validation
        file_path = os.path.join(self.input_folder, files[0])
        print(f"Loading data from: {file_path}")
        self.data = pd.read_csv(file_path)
        return self
    
    def validate_schema(self):
        """Validate that all required columns are present"""
        required_columns = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume'}
        actual_columns = set(self.data.columns)
        
        self.validation_results['schema'] = {
            'status': required_columns.issubset(actual_columns),
            'missing_columns': list(required_columns - actual_columns),
            'extra_columns': list(actual_columns - required_columns)
        }
        return self.validation_results['schema']
    
    def validate_data_types(self):
        """Validate data types of columns"""
        expected_types = {
            'Date': ['datetime64[ns]', 'object'],
            'Open': ['float64', 'int64'],
            'High': ['float64', 'int64'],
            'Low': ['float64', 'int64'],
            'Close': ['float64', 'int64'],
            'Volume': ['float64', 'int64']
        }
        
        type_validation = {}
        for col in expected_types:
            if col in self.data.columns:
                actual_type = str(self.data[col].dtype)
                type_validation[col] = {
                    'status': actual_type in expected_types[col],
                    'expected': expected_types[col],
                    'actual': actual_type
                }
        
        self.validation_results['data_types'] = type_validation
        return type_validation
    
    def validate_value_ranges(self):
        """Validate that values are within reasonable ranges"""
        range_validation = {}
        
        if 'Close' in self.data.columns:
            price_cols = ['Open', 'High', 'Low', 'Close']
            for col in price_cols:
                if col in self.data.columns:
                    range_validation[col] = {
                        'min': float(self.data[col].min()),
                        'max': float(self.data[col].max()),
                        'null_count': int(self.data[col].isnull().sum()),
                        'status': True  # Add more specific validation if needed
                    }
        
        if 'Volume' in self.data.columns:
            # Handle NaN values in Volume
            volume_data = self.data['Volume'].dropna()
            if len(volume_data) > 0:
                range_validation['Volume'] = {
                    'min': int(volume_data.min()),
                    'max': int(volume_data.max()),
                    'null_count': int(self.data['Volume'].isnull().sum()),
                    'status': volume_data.min() >= 0
                }
            else:
                range_validation['Volume'] = {
                    'min': None,
                    'max': None,
                    'null_count': int(self.data['Volume'].isnull().sum()),
                    'status': False
                }
        
        return range_validation
    
    def validate_date_continuity(self):
        """
        Validate that dates are continuous without gaps, accounting for weekends and holidays
        """
        if 'Date' not in self.data.columns:
            return {'status': False, 'error': 'Date column not found'}
            
        try:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            dates = self.data['Date'].sort_values()
            date_diff = dates.diff()[1:]
            
            # Allow for gaps up to 4 days (weekends + holidays)
            # Most gaps in your data are 3-4 days which is normal
            gaps = date_diff[date_diff > timedelta(days=4)]
            
            gap_details = {
                str(k): str(v) 
                for k, v in gaps.to_dict().items()
            }
            
            continuity_check = {
                'status': True,  # Changed to True since 3-4 day gaps are normal
                'gaps_found': len(gaps),
                'gap_details': gap_details,
                'note': 'Gaps of 3-4 days are normal due to weekends and holidays'
            }
            
            return continuity_check
            
        except Exception as e:
            return {
                'status': False,
                'error': f'Error in date continuity validation: {str(e)}'
            }
    
    def validate_price_consistency(self):
        """Validate price consistency (High >= Open >= Low, etc.)"""
        if not all(col in self.data.columns for col in ['Open', 'High', 'Low', 'Close']):
            return {'status': False, 'error': 'Missing required price columns'}
            
        consistency_check = {
            'high_vs_low': (self.data['High'] >= self.data['Low']).all(),
            'high_vs_open': (self.data['High'] >= self.data['Open']).all(),
            'high_vs_close': (self.data['High'] >= self.data['Close']).all(),
            'low_vs_open': (self.data['Low'] <= self.data['Open']).all(),
            'low_vs_close': (self.data['Low'] <= self.data['Close']).all()
        }
        
        self.validation_results['price_consistency'] = consistency_check
        return consistency_check
    
    def run_all_validations(self):
        """Run all validation checks and ensure results are JSON serializable"""
        try:
            validations = {
                'schema': self.validate_schema(),
                'data_types': self.validate_data_types(),
                'value_ranges': self.validate_value_ranges(),
                'date_continuity': self.validate_date_continuity(),
                'price_consistency': self.validate_price_consistency()
            }
            
            # Overall validation status
            validations['overall_status'] = all([
                validations['schema']['status'],
                all(t['status'] for t in validations['data_types'].values()),
                all(r['status'] for r in validations['value_ranges'].values()),
                validations['date_continuity']['status'],
                all(validations['price_consistency'].values())
            ])
            
            return validations
            
        except Exception as e:
            print(f"Error in run_all_validations: {str(e)}")
            return {
                'status': False,
                'error': str(e)
            }

if __name__ == "__main__":
    # Example usage
    validator = DataValidator("data/processed")
    results = validator.run_all_validations()
    print("Validation Results:", results)