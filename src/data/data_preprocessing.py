import pandas as pd
import os
import numpy as np
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_and_save_all_data(input_folder, output_folder):
    """
    Process all stock data files in the input folder with additional technical indicators
    """
    try:
        input_folder = os.path.abspath(input_folder)
        output_folder = os.path.abspath(output_folder)
        
        logging.info(f"Processing data from: {input_folder}")
        logging.info(f"Saving results to: {output_folder}")
        
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"Input folder not found: {input_folder}")
        
        os.makedirs(output_folder, exist_ok=True)
        
        files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
        if not files:
            raise ValueError(f"No CSV files found in {input_folder}")
        
        for file in files:
            try:
                input_path = os.path.join(input_folder, file)
                output_path = os.path.join(output_folder, f"processed_{file}")
                
                # Read data
                logging.info(f"\nProcessing file: {input_path}")
                df = pd.read_csv(input_path)
                
                if 'Date' not in df.columns or 'Close' not in df.columns:
                    raise ValueError(f"Missing required columns in {file}")

                # Convert 'Date' column to datetime format
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)

                # Handle missing values
                df.fillna(method='ffill', inplace=True)
                df.fillna(method='bfill', inplace=True)

                # Calculate Returns
                df['Returns'] = df['Close'].pct_change() * 100

                # Moving Averages
                df['MA5'] = df['Close'].rolling(window=5).mean()
                df['MA20'] = df['Close'].rolling(window=20).mean()

                # Volatility (20-day rolling standard deviation)
                df['Volatility'] = df['Returns'].rolling(window=20).std()

                # MACD (Moving Average Convergence Divergence)
                df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
                df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = df['EMA12'] - df['EMA26']
                df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

                # RSI (Relative Strength Index)
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))

                # Bollinger Bands
                df['Upper_Band'] = df['MA20'] + (df['Close'].rolling(window=20).std() * 2)
                df['Lower_Band'] = df['MA20'] - (df['Close'].rolling(window=20).std() * 2)

                # Drop NaN values that might result from initial calculations
                df.dropna(inplace=True)

                # Save processed data
                df.to_csv(output_path)
                logging.info(f"Saved processed file to: {output_path}")

            except Exception as e:
                logging.error(f"Error processing file {file}: {str(e)}")
                raise
        
        logging.info("Data preprocessing completed successfully.")
        return output_folder
        
    except Exception as e:
        logging.error(f"Critical error in process_and_save_all_data: {str(e)}")
        raise

if __name__ == "__main__":
    # Determine project root and set paths
    project_root = os.getcwd()
    input_folder = os.path.join(project_root, "data", "raw")
    output_folder = os.path.join(project_root, "data", "processed")

    logging.info(f"Data will be processed and saved to: {output_folder}")
    
    # Run the preprocessing
    process_and_save_all_data(input_folder, output_folder)
