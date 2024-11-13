import pytest
import pandas as pd
from scripts.data_acquisition import fetch_stock_data
import os

def test_fetch_stock_data():
    """Test stock data acquisition"""
    tickers = ["AAPL"]
    start_date = "2024-01-01"
    end_date = "2024-01-31"
    output_folder = "test_data/raw"
    
    # Test data fetching
    result_path = fetch_stock_data(tickers, start_date, end_date, output_folder)
    
    # Assertions
    assert os.path.exists(result_path)
    assert os.path.exists(os.path.join(result_path, "AAPL.csv"))
    
    # Test data content
    df = pd.read_csv(os.path.join(result_path, "AAPL.csv"))
    assert not df.empty
    assert all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
