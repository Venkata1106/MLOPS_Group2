import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
import os
from scipy import stats
from datetime import datetime
import json

class StockAnalyzer:
    """Class for analyzing stock market data statistics"""
    
    def __init__(self):
        """Initialize StockAnalyzer"""
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
        
    def calculate_returns_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate return-based metrics"""
        returns = df['Close'].pct_change()
        log_returns = np.log(df['Close']/df['Close'].shift(1))
        
        return {
            'daily_returns': returns.describe().to_dict(),
            'log_returns': log_returns.describe().to_dict(),
            'annualized_return': returns.mean() * 252,
            'annualized_volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }
    
    def calculate_price_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate price-based metrics"""
        return {
            'price_stats': df['Close'].describe().to_dict(),
            'price_momentum': {
                'daily': df['Close'].pct_change(1).mean(),
                'weekly': df['Close'].pct_change(5).mean(),
                'monthly': df['Close'].pct_change(21).mean()
            },
            'price_trends': {
                'up_days': (df['Close'].pct_change() > 0).sum(),
                'down_days': (df['Close'].pct_change() < 0).sum(),
                'flat_days': (df['Close'].pct_change() == 0).sum()
            }
        }
    
    def calculate_volume_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume-based metrics"""
        return {
            'volume_stats': df['Volume'].describe().to_dict(),
            'volume_trends': {
                'avg_up_volume': df.loc[df['Close'].pct_change() > 0, 'Volume'].mean(),
                'avg_down_volume': df.loc[df['Close'].pct_change() < 0, 'Volume'].mean(),
                'volume_momentum': df['Volume'].pct_change(5).mean()
            }
        }
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # Moving averages
        for window in [20, 50, 200]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df
    
    def calculate_correlation_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate correlation metrics"""
        return {
            'price_volume_corr': df['Close'].corr(df['Volume']),
            'returns_volume_corr': df['Close'].pct_change().corr(df['Volume'].pct_change())
        }
    
    def analyze_stock_data(self):
        """Perform statistical analysis"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        analysis = self._perform_analysis()
        return analysis
        
    def save_results(self, analysis, output_folder):
        """Save analysis results"""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        output_file = os.path.join(output_folder, 'stock_analysis.json')
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=4)
    
    def _perform_analysis(self):
        """Perform statistical analysis on the data"""
        analysis = {
            'summary_statistics': self._calculate_summary_stats(),
            'correlation_analysis': self._calculate_correlations(),
            'trend_analysis': self._analyze_trends(),
            'volatility_analysis': self._analyze_volatility()
        }
        return analysis
        
    def _calculate_summary_stats(self):
        """Calculate basic summary statistics"""
        summary = {}
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            summary[col] = {
                'mean': float(self.data[col].mean()),
                'median': float(self.data[col].median()),
                'std': float(self.data[col].std()),
                'min': float(self.data[col].min()),
                'max': float(self.data[col].max())
            }
        return summary
        
    def _calculate_correlations(self):
        """Calculate correlations between different metrics"""
        corr_matrix = self.data[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
        return corr_matrix.to_dict()
        
    def _analyze_trends(self):
        """Analyze price trends"""
        trends = {}
        for symbol in self.data['Symbol'].unique():
            symbol_data = self.data[self.data['Symbol'] == symbol]
            trends[symbol] = {
                'price_trend': float(symbol_data['Close'].iloc[-1] - symbol_data['Close'].iloc[0]),
                'avg_daily_return': float(symbol_data['Returns'].mean()),
                'trading_days': len(symbol_data)
            }
        return trends
        
    def _analyze_volatility(self):
        """Analyze price volatility"""
        volatility = {}
        for symbol in self.data['Symbol'].unique():
            symbol_data = self.data[self.data['Symbol'] == symbol]
            volatility[symbol] = {
                'daily_volatility': float(symbol_data['Returns'].std()),
                'annualized_volatility': float(symbol_data['Returns'].std() * (252 ** 0.5)),
                'max_drawdown': float(self._calculate_max_drawdown(symbol_data['Close']))
            }
        return volatility
        
    def _calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown from peak"""
        peak = prices.expanding(min_periods=1).max()
        drawdown = (prices - peak) / peak
        return drawdown.min()

def analyze_stock_data(input_folder: str, output_folder: str) -> Dict[str, Any]:
    """
    Convenience function to run statistical analysis
    """
    analyzer = StockAnalyzer()
    analyzer.load_data(input_folder)
    return analyzer.analyze_stock_data()

if __name__ == "__main__":
    # Example usage
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_folder = os.path.join(project_root, 'data', 'mitigated')  # Use absolute path
    output_folder = os.path.join(project_root, 'data', 'statistics')  # Use absolute path
    
    results = analyze_stock_data(input_folder, output_folder)
    print("Data Statistics Results:", results)