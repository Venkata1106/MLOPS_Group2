import yfinance as yf
import os
import pandas as pd
from datetime import datetime
import logging
from typing import List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fetch_stock_data(
    tickers: List[str],
    start_date: str,
    output_folder: str,
    end_date: Optional[str] = None
) -> str:
    """
    Fetch stock data and save to CSV with validation and error handling
    
    Args:
        tickers (List[str]): List of stock ticker symbols
        start_date (str): Start date in YYYY-MM-DD format
        output_folder (str): Path to save the data
        end_date (Optional[str]): End date in YYYY-MM-DD format
    
    Returns:
        str: Path to the output folder
    """
    try:
        # Input validation
        if not tickers:
            raise ValueError("Tickers list cannot be empty")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Set end date if not provided
        end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
        failed_downloads = []
        successful_downloads = []
        
        for ticker in tickers:
            try:
                # Download data
                logging.info(f"Downloading data for {ticker}")
                data = yf.download(ticker, start=start_date, end=end_date)
                
                # Validate downloaded data
                if data.empty:
                    raise ValueError(f"No data retrieved for {ticker}")
                
                # Save to CSV
                output_file = os.path.join(output_folder, f"{ticker}.csv")
                data.to_csv(output_file)
                
                # Basic data quality check
                if os.path.getsize(output_file) > 0:
                    successful_downloads.append(ticker)
                    logging.info(f"Successfully downloaded data for {ticker} to {output_file}")
                else:
                    failed_downloads.append(ticker)
                    logging.error(f"Downloaded file for {ticker} is empty")
                
            except Exception as e:
                failed_downloads.append(ticker)
                logging.error(f"Failed to download {ticker}: {str(e)}")
                continue
        
        # Summary logging
        logging.info(f"Download complete. Success: {len(successful_downloads)}, Failed: {len(failed_downloads)}")
        if failed_downloads:
            logging.warning(f"Failed downloads: {', '.join(failed_downloads)}")
        
        return output_folder
        
    except Exception as e:
        logging.error(f"Critical error in fetch_stock_data: {str(e)}")
        raise

def validate_dates(start_date: str, end_date: Optional[str] = None) -> bool:
    """
    Validate date formats and ranges
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (Optional[str]): End date in YYYY-MM-DD format
    
    Returns:
        bool: True if dates are valid
    """
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        if end_date:
            end = datetime.strptime(end_date, '%Y-%m-%d')
            if end < start:
                raise ValueError("End date cannot be earlier than start date")
        return True
    except ValueError as e:
        logging.error(f"Date validation error: {str(e)}")
        return False

if __name__ == "__main__":
    # Example usage
    tickers = ["AAPL", "GOOGL", "MSFT"]
    start_date = "2020-01-01"
    output_folder = os.path.join("data", "raw")
    
    # Validate dates before proceeding
    if validate_dates(start_date):
        fetch_stock_data(tickers, start_date, output_folder)
        