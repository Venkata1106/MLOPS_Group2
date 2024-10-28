import pandas as pd
import os

def load_data(file_path):
    """Load stock data from a CSV file."""
    data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    return data

def preprocess_data(data):
    """Perform preprocessing steps on the data."""
    # Fill missing values, if any
    data = data.ffill().bfill()


    # Calculate moving averages (e.g., 5-day and 20-day)
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()

    # Calculate volatility (standard deviation over 5 days)
    data['Volatility'] = data['Close'].rolling(window=5).std()

    # Drop any remaining NaN values created by rolling operations
    data = data.dropna()
    return data

def process_and_save_all_data(input_folder="data/raw", output_folder="data/processed"):
    """Process all CSV files in the raw data folder and save processed data."""
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_folder, file_name)
            print(f"Processing {file_name}...")
            data = load_data(file_path)
            data = preprocess_data(data)
            output_path = os.path.join(output_folder, file_name)
            data.to_csv(output_path)
            print(f"Processed data saved to {output_path}")

# Example usage
if __name__ == "__main__":
    process_and_save_all_data()
