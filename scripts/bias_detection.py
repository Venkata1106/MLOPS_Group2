import pandas as pd
import numpy as np
import os
from datetime import datetime
import json
class BiasAnalyzer:
    """Class for detecting bias in stock market data"""
    
    def __init__(self, input_folder=None):
        self.input_folder = input_folder
        self.data = None
    
    def load_data(self):
        """Load data from validated CSV files"""
        if not self.input_folder or not os.path.exists(self.input_folder):
            raise ValueError(f"Invalid input folder: {self.input_folder}")
            
        # Get all validated CSV files
        files = [f for f in os.listdir(self.input_folder) if f.endswith('.csv')]
        if not files:
            raise ValueError(f"No CSV files found in {self.input_folder}")
            
        # Load and combine all files
        dfs = []
        for file in files:
            file_path = os.path.join(self.input_folder, file)
            df = pd.read_csv(file_path)
            df['Symbol'] = file.replace('.csv', '').replace('validated_', '')
            dfs.append(df)
            
        self.data = pd.concat(dfs, ignore_index=True)
        return self
    
    def generate_report(self):
        """Generate bias analysis report"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        # Perform bias analysis
        report = {
            'timestamp': datetime.now().isoformat(),
            'analysis': {
                'price_distribution': self._analyze_price_distribution(),
                'volume_distribution': self._analyze_volume_distribution(),
                'temporal_patterns': self._analyze_temporal_patterns()
            }
        }
        
        return report
    
    def save_results(self, bias_report, output_folder):
        """Save bias analysis results"""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        output_file = os.path.join(output_folder, 'bias_analysis.json')
        with open(output_file, 'w') as f:
            json.dump(bias_report, f, indent=4)
    
    def _analyze_price_distribution(self):
        """Analyze price distribution for bias"""
        return {
            'mean_price': float(self.data['Close'].mean()),
            'median_price': float(self.data['Close'].median()),
            'price_skew': float(self.data['Close'].skew())
        }
    
    def _analyze_volume_distribution(self):
        """Analyze trading volume distribution"""
        return {
            'mean_volume': float(self.data['Volume'].mean()),
            'median_volume': float(self.data['Volume'].median()),
            'volume_skew': float(self.data['Volume'].skew())
        }
    
    def _analyze_temporal_patterns(self):
        """Analyze temporal patterns in the data"""
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data['DayOfWeek'] = self.data['Date'].dt.dayofweek
        
        return {
            'day_of_week_stats': self.data.groupby('DayOfWeek')['Close'].mean().to_dict(),
            'trading_frequency': len(self.data) / len(self.data['Date'].unique())
        }