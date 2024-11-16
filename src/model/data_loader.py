import pandas as pd
import os
from typing import Dict, List, Tuple
import logging

class StockDataLoader:
    def __init__(self, base_path: str = 'data/mitigated'):
        self.base_path = base_path
        
    def load_stock_data(self, symbol: str) -> pd.DataFrame:
        """Load mitigated data for a specific stock"""
        file_path = os.path.join(self.base_path, f'mitigated_{symbol}.csv')
        try:
            data = pd.read_csv(file_path)
            logging.info(f"Loaded data for {symbol}: {len(data)} records")
            return data
        except Exception as e:
            logging.error(f"Error loading data for {symbol}: {str(e)}")
            raise
            
    def load_all_stocks(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Load data for all specified stocks"""
        return {
            symbol: self.load_stock_data(symbol)
            for symbol in symbols
        } 