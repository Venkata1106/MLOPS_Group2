import os
import pandas as pd
import yaml

def process_and_save_all_data(input_folder, output_folder, params):
    """
    Process all stock data files in the input folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    
    for file in files:
        df = pd.read_csv(os.path.join(input_folder, file))
        
        # Ensure required columns
        required_columns = params['preprocessing']['columns_required']
        if not all(col in df.columns for col in required_columns):
            print(f"Skipping {file} due to missing required columns")
            continue

        # Moving average calculations
        for window in params['preprocessing']['moving_average_windows']:
            df[f'MA{window}'] = df['Close'].rolling(window=window).mean()
        
        # Fill NaN values based on fill_method
        fill_method = params['preprocessing']['fill_method']
        if fill_method == 'ffill':
            df.ffill(inplace=True)
        elif fill_method == 'bfill':
            df.bfill(inplace=True)

        # Save the processed file
        output_file = os.path.join(output_folder, f"processed_{file}")
        df.to_csv(output_file, index=False)
        print(f"Processed file saved to {output_file}")

if __name__ == "__main__":
    # Load parameters
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    input_folder = os.path.join("data", "raw")
    output_folder = os.path.join("data", "processed")
    process_and_save_all_data(input_folder, output_folder, params)
