import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.ensemble import IsolationForest
import os
import logging
from datetime import datetime
import json

class AnomalyDetector:
    """Class for detecting anomalies in stock market data"""
    
    def __init__(self, contamination: float = 0.1):
        """
        Initialize AnomalyDetector
        
        Parameters:
        -----------
        contamination : float
            Expected proportion of outliers in the data
        """
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.data = None
    
    def load_data(self, input_folder):
        """Load data from mitigated CSV files"""
        if not os.path.exists(input_folder):
            raise ValueError(f"Input folder does not exist: {input_folder}")
            
        files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
        if not files:
            raise ValueError(f"No CSV files found in {input_folder}")
            
        dfs = []
        for file in files:
            file_path = os.path.join(input_folder, file)
            df = pd.read_csv(file_path)
            df['Symbol'] = file.replace('.csv', '').replace('mitigated_', '')
            dfs.append(df)
            
        self.data = pd.concat(dfs, ignore_index=True)
        return self
        
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features for anomaly detection
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with stock data
            
        Returns:
        --------
        pd.DataFrame : DataFrame with calculated features
        """
        features = pd.DataFrame()
        
        # Price-based features
        features['Returns'] = df['Close'].pct_change()
        features['Price_Volatility'] = features['Returns'].rolling(window=20).std()
        
        # Volume-based features
        features['Volume_Change'] = df['Volume'].pct_change()
        features['Volume_MA_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        
        # Price range features
        features['Price_Range'] = (df['High'] - df['Low']) / df['Close']
        
        # Update fillna to use newer methods
        features = features.ffill().fillna(0)
        
        return features
    
    def detect_anomalies(self):
        """Detect anomalies in the data"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        anomalies = self._detect_anomalies()
        return anomalies
        
    def save_results(self, anomalies, output_folder):
        """Save anomaly detection results"""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        output_file = os.path.join(output_folder, 'anomalies.json')
        with open(output_file, 'w') as f:
            json.dump(anomalies, f, indent=4)
    
    def _detect_anomalies(self):
        """Detect anomalies in the stock data"""
        anomalies = {
            'price_anomalies': self._detect_price_anomalies(),
            'volume_anomalies': self._detect_volume_anomalies(),
            'volatility_anomalies': self._detect_volatility_anomalies()
        }
        return anomalies
        
    def _detect_price_anomalies(self):
        """Detect anomalies in price movements"""
        price_anomalies = {}
        for symbol in self.data['Symbol'].unique():
            symbol_data = self.data[self.data['Symbol'] == symbol]
            
            # Calculate z-scores for price changes
            returns = symbol_data['Close'].pct_change()
            z_scores = (returns - returns.mean()) / returns.std()
            
            # Identify anomalous days (z-score > 3 or < -3)
            anomalous_days = symbol_data[abs(z_scores) > 3]
            
            price_anomalies[symbol] = {
                'dates': anomalous_days['Date'].tolist(),
                'prices': anomalous_days['Close'].tolist(),
                'z_scores': z_scores[abs(z_scores) > 3].tolist()
            }
        return price_anomalies
        
    def _detect_volume_anomalies(self):
        """Detect anomalies in trading volume"""
        volume_anomalies = {}
        for symbol in self.data['Symbol'].unique():
            symbol_data = self.data[self.data['Symbol'] == symbol]
            
            # Calculate z-scores for volume
            volume_zscore = (symbol_data['Volume'] - symbol_data['Volume'].mean()) / symbol_data['Volume'].std()
            
            # Identify anomalous volume days
            anomalous_days = symbol_data[abs(volume_zscore) > 3]
            
            volume_anomalies[symbol] = {
                'dates': anomalous_days['Date'].tolist(),
                'volumes': anomalous_days['Volume'].tolist(),
                'z_scores': volume_zscore[abs(volume_zscore) > 3].tolist()
            }
        return volume_anomalies
        
    def _detect_volatility_anomalies(self):
        """Detect anomalies in volatility"""
        volatility_anomalies = {}
        window = 20  # 20-day rolling window
        
        for symbol in self.data['Symbol'].unique():
            symbol_data = self.data[self.data['Symbol'] == symbol]
            
            # Calculate rolling volatility
            returns = symbol_data['Close'].pct_change()
            rolling_vol = returns.rolling(window=window).std()
            
            # Calculate z-scores for volatility
            vol_zscore = (rolling_vol - rolling_vol.mean()) / rolling_vol.std()
            
            # Identify anomalous volatility periods
            anomalous_days = symbol_data[abs(vol_zscore) > 3]
            
            volatility_anomalies[symbol] = {
                'dates': anomalous_days['Date'].tolist(),
                'volatility': rolling_vol[abs(vol_zscore) > 3].tolist(),
                'z_scores': vol_zscore[abs(vol_zscore) > 3].tolist()
            }
        return volatility_anomalies

def run_anomaly_detection(input_folder: str, output_folder: str) -> Dict[str, Any]:
    """
    Convenience function to run anomaly detection
    """
    detector = AnomalyDetector()
    detector.load_data(input_folder)
    return detector.detect_anomalies()

if __name__ == "__main__":
    # Example usage
    input_folder = "data/mitigated"
    output_folder = "data/anomalies"
    
    results = run_anomaly_detection(input_folder, output_folder)
    print("Anomaly Detection Results:", results)