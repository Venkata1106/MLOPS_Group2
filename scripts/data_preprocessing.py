import pandas as pd
import os
import numpy as np

def process_and_save_all_data(input_folder, output_folder):
    """
    Process all stock data files in the input folder
    """
    try:
        # Convert to absolute paths if they're not already
        input_folder = os.path.abspath(input_folder)
        output_folder = os.path.abspath(output_folder)
        
        print(f"Processing data from: {input_folder}")
        print(f"Saving results to: {output_folder}")
        
        # Ensure input folder exists
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"Input folder not found: {input_folder}")
            
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Process files
        files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
        
        if not files:
            raise ValueError(f"No CSV files found in {input_folder}")
            
        for file in files:
            try:
                input_path = os.path.join(input_folder, file)
                output_path = os.path.join(output_folder, f"processed_{file}")
                
                # Read data
                print(f"\nProcessing file: {input_path}")
                df = pd.read_csv(input_path)
                print(f"Initial columns: {df.columns.tolist()}")
                
                # Ensure required columns exist
                required_columns = ['Date', 'Close']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    raise ValueError(f"Missing required columns: {missing_columns}")
                
                # Convert Date to datetime
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Calculate technical indicators
                print("Calculating technical indicators...")
                
                # 1. Returns (percentage change in closing price)
                df['Returns'] = df['Close'].pct_change() * 100
                print("Added Returns column")
                
                # 2. Moving Averages
                df['MA5'] = df['Close'].rolling(window=5).mean()
                df['MA20'] = df['Close'].rolling(window=20).mean()
                print("Added Moving Averages columns")
                
                # 3. Volatility (20-day rolling standard deviation of returns)
                df['Volatility'] = df['Returns'].rolling(window=20).std()
                print("Added Volatility column")
                
                # Fill NaN values that result from calculations
                df = df.ffill().bfill()  # Updated to new syntax
                print("Filled NaN values")
                
                # Verify columns before saving
                print(f"Final columns: {df.columns.tolist()}")
                
                # Save processed data
                df.to_csv(output_path, index=False)
                print(f"Saved processed file to: {output_path}")
                
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
                raise
            
        return output_folder
        
    except Exception as e:
        print(f"Error in process_and_save_all_data: {str(e)}")
        raise

if __name__ == "__main__":
    # Set up the paths relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_folder = os.path.join(project_root, 'data', 'raw')
    output_folder = os.path.join(project_root, 'data', 'processed')
    
    # Run the preprocessing
    process_and_save_all_data(input_folder, output_folder)