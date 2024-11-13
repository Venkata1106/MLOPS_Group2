from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta

class BiasMitigator:
    """Class for mitigating bias in stock market data"""
    
    def __init__(self):
        self.data = None
        
    def load_data(self, input_folder):
        """Load data from validated CSV files"""
        if not os.path.exists(input_folder):
            raise ValueError(f"Input folder does not exist: {input_folder}")
            
        files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
        if not files:
            raise ValueError(f"No CSV files found in {input_folder}")
            
        dfs = []
        for file in files:
            file_path = os.path.join(input_folder, file)
            df = pd.read_csv(file_path)
            df['Symbol'] = file.replace('.csv', '').replace('validated_', '')
            dfs.append(df)
            
        self.data = pd.concat(dfs, ignore_index=True)
        return self
        
    def mitigate_bias(self, data=None):
        """Apply bias mitigation techniques"""
        if data is not None:
            self.data = data
            
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        # Apply mitigation techniques
        mitigated_data = self._apply_mitigation()
        return mitigated_data
        
    def save_results(self, mitigated_data, output_folder):
        """Save mitigated data to CSV files"""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        for symbol in mitigated_data['Symbol'].unique():
            symbol_data = mitigated_data[mitigated_data['Symbol'] == symbol]
            output_file = os.path.join(output_folder, f'mitigated_{symbol}.csv')
            symbol_data.to_csv(output_file, index=False)
        
    def _apply_mitigation(self):
        """Apply bias mitigation techniques to the data"""
        mitigated_data = self.data.copy()
        
        # 1. Handle outliers using winsorization
        for col in ['Open', 'High', 'Low', 'Close']:
            q_low = mitigated_data[col].quantile(0.01)
            q_high = mitigated_data[col].quantile(0.99)
            mitigated_data[col] = mitigated_data[col].clip(q_low, q_high)
            
        # 2. Normalize volume data
        mitigated_data['Volume'] = (mitigated_data['Volume'] - mitigated_data['Volume'].mean()) / mitigated_data['Volume'].std()
        
        # 3. Add technical indicators for better balance
        mitigated_data['Returns'] = mitigated_data.groupby('Symbol')['Close'].pct_change()
        mitigated_data['Volatility'] = mitigated_data.groupby('Symbol')['Returns'].rolling(window=20).std().reset_index(0, drop=True)
        
        # 4. Handle missing values
        mitigated_data = mitigated_data.fillna(method='ffill').fillna(method='bfill')
        
        return mitigated_data

def run_bias_mitigation(input_folder: str, output_folder: str) -> Dict[str, Any]:
    """
    Convenience function to run bias mitigation
    """
    mitigator = BiasMitigator()
    
    # Load the data
    mitigator.load_data(input_folder)
    
    # Apply bias mitigation
    mitigated_data = mitigator.mitigate_bias()
    
    # Save the results
    mitigator.save_results(mitigated_data, output_folder)
    
    return mitigated_data  # Return the mitigated data or any other results you want

if __name__ == "__main__":
    # Example usage
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_folder = os.path.join(project_root, 'data', 'processed')  # Use absolute path
    output_folder = os.path.join(project_root, 'data', 'mitigated')  # Use absolute path
    
    results = run_bias_mitigation(input_folder, output_folder)
    print("Bias Mitigation Results:", results)