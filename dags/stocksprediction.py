# airflow/dags/stocksprediction.py

import sys
import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# Add the path to your scripts
sys.path.append('/opt/airflow/src/data')

# Import your functions from the scripts
from data_ingestion import fetch_stock_data  # Import the ingestion function
from data_preprocessing import process_and_save_all_data  # Import the preprocessing function

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 10, 1),  # Set your start date
    'retries': 1,
}

# Define the DAG
with DAG('stocksprediction_dag',  # Unique DAG ID
         default_args=default_args,
         schedule_interval='@daily',  # Set your schedule
         catchup=False) as dag:

    # Define the ingestion task
    ingestion_task = PythonOperator(
        task_id='data_ingestion',
        python_callable=fetch_stock_data,  # Call the fetch_stock_data function
        op_kwargs={
            'tickers': ["AAPL", "GOOGL", "MSFT"],  # Pass the tickers as arguments
            'start_date': "2020-01-01",  # Start date
            'output_folder': os.path.join("/opt/airflow/data/raw"),  # Output folder
        },
    )

    # Define the preprocessing task
    preprocessing_task = PythonOperator(
        task_id='data_preprocessing',
        python_callable=process_and_save_all_data,  # Call the process_and_save_all_data function
        op_kwargs={
            'input_folder': os.path.join("/opt/airflow/data/raw"),  # Input folder for raw data
            'output_folder': os.path.join("/opt/airflow/data/processed"),  # Output folder for processed data
        },
    )

    # Set task dependencies
    ingestion_task >> preprocessing_task