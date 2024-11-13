import yfinance as yf
import os
import pandas as pd
from datetime import datetime  # Import datetime module

def fetch_stock_data(tickers, start_date, output_folder):
    """
    Fetch stock data and save to CSV
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        output_files = []
        end_date = datetime.now().strftime('%Y-%m-%d')  # Get today's date in YYYY-MM-DD format
        
        for ticker in tickers:
            # Download data
            data = yf.download(ticker, start=start_date, end=end_date)
            
            # Save to CSV
            output_file = os.path.join(output_folder, f"{ticker}.csv")
            data.to_csv(output_file)
            output_files.append(output_file)
            print(f"Downloaded data for {ticker} to {output_file}")
        
        return output_folder  # Return the folder path
        
    except Exception as e:
        print(f"Error in fetch_stock_data: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    tickers = ["AAPL", "GOOGL", "MSFT"]  # Add tickers you want
    output_folder = os.path.join("data", "raw")  # Set the output folder path
    fetch_stock_data(tickers, "2020-01-01", output_folder)  # Call with start date only