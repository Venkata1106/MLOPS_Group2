import os
import pytest
import pandas as pd
from src.data.data_preprocessing import process_and_save_all_data  # Adjust import paths as needed

# Test missing input folder
def test_missing_input_folder(tmp_path):
    input_folder = tmp_path / "non_existent"
    output_folder = tmp_path / "output"
    with pytest.raises(FileNotFoundError, match="Input folder not found"):
        process_and_save_all_data(str(input_folder), str(output_folder))

# Test missing required columns
def test_missing_columns(tmp_path):
    input_folder = tmp_path / "input"
    output_folder = tmp_path / "output"
    os.makedirs(input_folder)
    
    # Create a CSV without 'Date' and 'Close' columns
    df = pd.DataFrame({"Open": [100, 101]})
    csv_path = input_folder / "missing_columns.csv"
    df.to_csv(csv_path, index=False)
    
    with pytest.raises(ValueError, match="Missing required columns"):
        process_and_save_all_data(str(input_folder), str(output_folder))

# Test successful preprocessing
def test_successful_preprocessing(tmp_path):
    input_folder = tmp_path / "input"
    output_folder = tmp_path / "output"
    os.makedirs(input_folder)
    
    # Create a valid dummy CSV
    df = pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "Close": [100, 101, 102],
    })
    csv_path = input_folder / "valid_data.csv"
    df.to_csv(csv_path, index=False)
    
    process_and_save_all_data(str(input_folder), str(output_folder))
    
    # Check that processed file is created
    processed_file = output_folder / "processed_valid_data.csv"
    assert os.path.exists(processed_file)

    # Validate technical indicators
    processed_data = pd.read_csv(processed_file)
    assert "MA5" in processed_data.columns
    assert "RSI" in processed_data.columns
