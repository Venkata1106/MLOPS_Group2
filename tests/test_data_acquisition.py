import os
import pytest
from src.data.data_ingestion import fetch_stock_data, validate_dates  # Adjust import paths as needed
import pandas as pd

# Test date validation
def test_validate_dates():
    # Valid dates
    assert validate_dates("2020-01-01", "2021-01-01") == True
    
    # Invalid date order
    assert validate_dates("2021-01-01", "2020-01-01") == False
    
    # Incorrect format
    assert validate_dates("2020-13-01") == False  # Invalid month

# Test empty tickers list
def test_fetch_stock_data_empty_tickers():
    with pytest.raises(ValueError, match="Tickers list cannot be empty"):
        fetch_stock_data([], "2020-01-01", "test_output")

# Test successful download with valid tickers (mocked)
from unittest.mock import patch

@patch("src.data.data_ingestion.yf.download")
def test_fetch_stock_data_success(mock_download, tmp_path):
    mock_download.return_value = pd.DataFrame({
        "Open": [100, 101],
        "Close": [110, 111]
    })
    
    output_folder = tmp_path / "output"
    fetch_stock_data(["AAPL"], "2020-01-01", str(output_folder))
    
    assert len(os.listdir(output_folder)) == 1
