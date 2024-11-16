"""
The Airflow DAG for Stock Prediction Pipeline
"""
import logging
import sys
import os
import joblib
import json

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)

logger = logging.getLogger(__name__)

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.email import EmailOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.trigger_rule import TriggerRule
import os
import shutil
from airflow.utils.email import send_email_smtp as send_email
from airflow.hooks.base import BaseHook
from src.model.model_development import StockPredictionModel
from src.model.config import MODEL_CONFIG

# Add scripts directory to path
AIRFLOW_HOME = os.getenv('AIRFLOW_HOME', '/opt/airflow')
SCRIPTS_DIR = os.path.join(AIRFLOW_HOME, 'scripts')
sys.path.insert(0, SCRIPTS_DIR)

# Import pipeline components
try:
    from scripts.data_acquisition import fetch_stock_data
    from scripts.data_preprocessing import process_and_save_all_data
    from scripts.data_validation import DataValidator
    from scripts.bias_detection import BiasAnalyzer
    from scripts.bias_mitigation import BiasMitigator
    from scripts.data_statistics import StockAnalyzer
    from scripts.anomaly_detection import AnomalyDetector
    from scripts.pipeline_optimization import StockPipeline
    from scripts.logger_config import setup_task_logger
except ImportError as e:
    print(f"Import Error: {e}")
    print(f"PYTHONPATH: {os.getenv('PYTHONPATH')}")
    print(f"Current sys.path: {sys.path}")
    raise

# Define project directories
PROJECT_DIR = os.path.join(AIRFLOW_HOME, 'stock_prediction')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
DIRS = {
    'raw': os.path.join(DATA_DIR, 'raw'),
    'processed': os.path.join(DATA_DIR, 'processed'),
    'validated': os.path.join(DATA_DIR, 'validated'),
    'mitigated': os.path.join(DATA_DIR, 'mitigated'),
    'analyzed': os.path.join(DATA_DIR, 'analyzed'),
    'anomalies': os.path.join(DATA_DIR, 'anomalies'),
    'bias': os.path.join(DATA_DIR, 'bias'),
    'models': os.path.join(DATA_DIR, 'models'),
    'model_metrics': os.path.join(DATA_DIR, 'model_metrics'),
}

# Create directories if they don't exist
for dir_path in DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

# Define default arguments
default_args = {
    'owner': 'ananth',
    'start_date': datetime(2024, 1, 1),
    'email': ['ananthareddy12321@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=1),
}

def run_validation(**context):
    """Run data validation with logging"""
    logger.info("Starting data validation")
    processed_folder = DIRS['processed']
    validated_folder = DIRS['validated']
    
    try:
        logger.info(f"Processing data from: {processed_folder}")
        validator = DataValidator(input_folder=processed_folder)
        results = validator.run_all_validations()
        
        # Log validation results
        logger.info("Validation Results:")
        logger.info(f"Schema validation: {results['schema']}")
        logger.info(f"Data types validation: {results['data_types']}")
        logger.info(f"Value ranges validation: {results['value_ranges']}")
        logger.info(f"Date continuity validation: {results['date_continuity']}")
        logger.info(f"Price consistency validation: {results['price_consistency']}")
        
        if not results.get('overall_status', False):
            logger.error("Validation failed")
            raise ValueError("Data validation failed. Check logs for details.")
        
        # Move validated files to validated directory
        logger.info(f"Moving validated files to: {validated_folder}")
        for file in os.listdir(processed_folder):
            if file.endswith('.csv'):
                src = os.path.join(processed_folder, file)
                dst = os.path.join(validated_folder, file.replace('processed_', 'validated_'))
                shutil.copy2(src, dst)
                logger.info(f"Copied {file} to validated directory")
        
        logger.info("Validation completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Error in data validation: {str(e)}", exc_info=True)
        raise

def detect_bias(**context):
    """Run bias detection with logging"""
    logger.info("Starting bias detection")
    try:
        input_folder = DIRS['validated']
        output_folder = DIRS['bias']
        logger.info(f"Processing data from: {input_folder}")
        
        # Initialize analyzer with data
        analyzer = BiasAnalyzer(input_folder=input_folder)
        
        # Load and process data
        analyzer.load_data()
        
        # Generate report
        report = analyzer.generate_report()

        # Save results
        logger.info(f"Saving bias report to: {output_folder}")
        analyzer.save_results(report, output_folder)
        
        logger.info("Bias detection completed successfully")
        return {'status': 'success', 'bias_report': report}
        
    except Exception as e:
        logger.error(f"Error in bias detection: {str(e)}")
        raise

def mitigate_bias(**context):
    """Run bias mitigation with logging"""
    logger.info("Starting bias mitigation")
    try:
        input_folder = DIRS['validated']
        output_folder = DIRS['mitigated']
        
        logger.info(f"Loading data from: {input_folder}")
        mitigator = BiasMitigator()
        mitigator.load_data(input_folder)
        
        # Apply mitigation
        mitigated_data = mitigator.mitigate_bias()
        
        # Save results
        logger.info(f"Saving mitigated data to: {output_folder}")
        mitigator.save_results(mitigated_data, output_folder)
        
        logger.info("Bias mitigation completed successfully")
        return {'status': 'success', 'files_processed': len(os.listdir(input_folder))}
        
    except Exception as e:
        logger.error(f"Error in bias mitigation: {str(e)}")
        raise

def analyze_stock_data(**context):
    """Run statistical analysis with logging"""
    logger.info("Starting statistical analysis")
    try:
        input_folder = DIRS['mitigated']
        output_folder = DIRS['analyzed']
        
        logger.info(f"Loading data from: {input_folder}")
        analyzer = StockAnalyzer()
        analyzer.load_data(input_folder)
        
        # Perform analysis
        analysis = analyzer.analyze_stock_data()
        
        # Save results
        logger.info(f"Saving analysis to: {output_folder}")
        analyzer.save_results(analysis, output_folder)
        
        logger.info("Statistical analysis completed successfully")
        return {'status': 'success', 'analysis': analysis}
        
    except Exception as e:
        logger.error(f"Error in statistical analysis: {str(e)}")
        raise

def detect_anomalies(**context):
    """Run anomaly detection with logging"""
    logger.info("Starting anomaly detection")
    try:
        input_folder = DIRS['mitigated']
        output_folder = DIRS['anomalies']
        
        logger.info(f"Loading data from: {input_folder}")
        detector = AnomalyDetector()
        detector.load_data(input_folder)
        
        # Detect anomalies
        anomalies = detector.detect_anomalies()
        
        # Save results
        logger.info(f"Saving anomalies to: {output_folder}")
        detector.save_results(anomalies, output_folder)
        
        logger.info("Anomaly detection completed successfully")
        return {'status': 'success', 'anomalies': anomalies}
        
    except Exception as e:
        logger.error(f"Error in anomaly detection: {str(e)}")
        raise

# Standard library imports
import os
import logging
from datetime import datetime, timedelta

# Third-party imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Local imports
from src.model.model_development import StockPredictionModel

# Constants for stock symbols
SYMBOLS = [
    'AAPL', 'ADBE', 'AMZN', 'CSCO', 'GOOGL',
    'META', 'MSFT', 'NVDA', 'PEP', 'TSLA'
]

# Constants for file paths
BASE_PATH = '/opt/airflow'
DATA_PATH = os.path.join(BASE_PATH, 'data')
MITIGATED_PATH = os.path.join(DATA_PATH, 'mitigated')
MODEL_PATH = os.path.join(BASE_PATH, 'models')

# Configuration
config = {
    'data_path': DATA_PATH,
    'model_path': MODEL_PATH,
    'model_params': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    },
    'training': {
        'test_size': 0.2,
        'random_state': 42,
        'features': ['Date', 'Open', 'High', 'Low', 'Volume', 'Returns', 'MA5', 'MA20', 'Volatility'],
        'target': 'Close'
    }
}

# Make sure model directory exists
os.makedirs(MODEL_PATH, exist_ok=True)

def train_model(**context):
    """Train models for all stocks"""
    results = {}
    
    # Debug paths
    logging.info(f"MODEL_PATH exists: {os.path.exists(MODEL_PATH)}")
    logging.info(f"MODEL_PATH permissions: {oct(os.stat(MODEL_PATH).st_mode)[-3:]}")
    
    for symbol in SYMBOLS:
        try:
            logging.info(f"Training model for {symbol}")
            file_path = os.path.join(MITIGATED_PATH, f'mitigated_{symbol}.csv')
            model_path = os.path.join(MODEL_PATH, f'{symbol}_model.pkl')
            
            if not os.path.exists(file_path):
                logging.error(f"File not found: {file_path}")
                results[symbol] = {
                    'status': 'failed',
                    'error': f"Data file not found: {file_path}"
                }
                continue
            
            # Load and prepare data
            data = pd.read_csv(file_path)
            logging.info(f"Successfully loaded data for {symbol}: {len(data)} records")
            
            # Train model
            try:
                model = StockPredictionModel(
                    config={
                        'model_params': {
                            'n_estimators': 100,
                            'max_depth': 10,
                            'random_state': 42
                        },
                        'training': {
                            'test_size': 0.2,
                            'random_state': 42,
                            'features': ['Day','Month','Year','Day_of_Week','Open','High','Low','Volume','Returns','MA5','MA20','Volatility'],
                            'target': 'Close'
                        },
                        'output_path': model_path  # Add this line
                    }
                )
                
                result = model.train_model(data)

                logging.info(result['metrics'])
                if result.get('status') == 'success':
                    # Save model
                    try:
                        joblib.dump(result['metrics'], model_path)
                        logging.info(f"Model saved successfully to {model_path}")
                        results[symbol] = {
                            'status': 'success',
                            'metrics': result.get('metrics', {}),
                            'model_path': model_path
                        }
                    except Exception as e:
                        logging.error(f"Error saving model for {symbol}: {str(e)}")
                        results[symbol] = {
                            'status': 'failed',
                            'error': f"Model training succeeded but saving failed: {str(e)}"
                        }
                else:
                    results[symbol] = {
                        'status': 'failed',
                        'error': result.get('error', 'Unknown error during training')
                    }
                    
            except Exception as e:
                logging.error(f"Error training model for {symbol}: {str(e)}")
                logging.error("Full traceback: ", exc_info=True)
                results[symbol] = {
                    'status': 'failed',
                    'error': str(e)
                }
                
        except Exception as e:
            logging.error(f"Error processing {symbol}: {str(e)}")
            logging.error("Full traceback: ", exc_info=True)
            results[symbol] = {
                'status': 'failed',
                'error': str(e)
            }
    
    return results

def evaluate_model_bias(**context):
    """Evaluate model bias"""
    logger.info("Starting model bias evaluation")
    
    # Use the same config as training
    model_config = {
        'model_params': MODEL_CONFIG['model_params'],
        'training': MODEL_CONFIG['training'],
        'data_path': DATA_PATH,
        'output_path': MODEL_PATH
    }
    
    results = {}
    
    for symbol in SYMBOLS:
        try:
            logger.info(f"Evaluating bias for {symbol} model")
            
            # Initialize model with config
            model = StockPredictionModel(config=model_config)
            
            # Evaluate bias
            bias_results = model.evaluate_model_bias(symbol)
            results[symbol] = bias_results
            
        except Exception as e:
            logger.error(f"Error evaluating bias for {symbol}: {str(e)}")
            logger.error("Full traceback: ", exc_info=True)
            results[symbol] = {
                'status': 'failed',
                'error': str(e)
            }
    
    return results

# Email templates
EMAIL_SUCCESS_TEMPLATE = """
<h3>Stock Prediction Pipeline completed successfully!</h3>
<p>
<b>Execution Date:</b> {{ ds }}<br>
<b>DAG:</b> {{ dag.dag_id }}<br>
<b>Run ID:</b> {{ run_id }}<br>
</p>
<h4>Results Summary:</h4>
<ul>
    <li>Data Validation: Completed</li>
    <li>Bias Detection and Mitigation: Completed</li>
    <li>Statistical Analysis: Completed</li>
    <li>Anomaly Detection: Completed</li>
    <li>Model Training: Completed</li>
    <li>Model Bias Evaluation: Completed</li>
</ul>
"""

EMAIL_FAILURE_TEMPLATE = """
<h3>Stock Prediction Pipeline Failed</h3>
<p>
<b>Execution Date:</b> {{ ds }}<br>
<b>DAG:</b> {{ dag.dag_id }}<br>
<b>Run ID:</b> {{ run_id }}<br>
</p>
<p>Please check the Airflow logs for more details.</p>
"""

# Get email connection details
email_conn = BaseHook.get_connection('email_default')

# Email configuration
EMAIL_CONF = {
    'to': 'ananthareddy12321@gmail.com',
    'subject': 'Stock Prediction Pipeline Success - {{ ds }}',
    'html_content': """
    <h3>Stock Prediction Pipeline completed successfully!</h3>
    <p>
    <b>Execution Date:</b> {{ ds }}<br>
    <b>DAG:</b> {{ dag.dag_id }}<br>
    <b>Run ID:</b> {{ run_id }}<br>
    </p>
    """
}

# Create DAG
with DAG(
    'stock_prediction_pipeline',
    default_args=default_args,
    description='Complete Stock Prediction Pipeline',
    schedule_interval='0 2 * * *',
    catchup=False,
    tags=['stock', 'prediction', 'pipeline']
) as dag:

    # Log DAG initialization
    logger.info("Initializing Stock Prediction Pipeline DAG")

    # Start pipeline
    start = DummyOperator(task_id='start_pipeline')

    # Task 1: Data Acquisition
    acquire_data = PythonOperator(
        task_id='acquire_data',
        python_callable=fetch_stock_data,
        op_kwargs={
            'tickers': ["AAPL", "GOOGL", "TSLA", "MSFT", "AMZN", "META", "NVDA", "ADBE", "CSCO", "PEP"],
            'start_date': "2020-01-01",
            'end_date': "{{ ds }}",
            'output_folder': DIRS['raw']
        }
    )

    # Task 2: Data Preprocessing
    preprocess_data = PythonOperator(
        task_id='preprocess_data',
        python_callable=process_and_save_all_data,
        op_kwargs={
            'input_folder': DIRS['raw'],
            'output_folder': DIRS['processed']
        }
    )

    # Task 3: Data Validation
    validate_data = PythonOperator(
        task_id='validate_data',
        python_callable=run_validation,
        provide_context=True
    )

    # Task 4: Bias Detection
    detect_bias_task = PythonOperator(
        task_id='detect_bias',
        python_callable=detect_bias,
        provide_context=True
    )

    # Task 5: Bias Mitigation
    mitigate_bias_task = PythonOperator(
        task_id='mitigate_bias',
        python_callable=mitigate_bias,
        provide_context=True
    )

    # Task 6: Calculate Statistics
    calculate_statistics_task = PythonOperator(
        task_id='calculate_statistics',
        python_callable=analyze_stock_data,
        provide_context=True
    )

    # Task 7: Anomaly Detection
    detect_anomalies_task = PythonOperator(
        task_id='detect_anomalies',
        python_callable=detect_anomalies,
        provide_context=True
    )

    # Task 8: Train Model
    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        provide_context=True
    )

    # Task 9: Evaluate Model Bias
    evaluate_model_bias_task = PythonOperator(
        task_id='evaluate_model_bias',
        python_callable=evaluate_model_bias,
        provide_context=True
    )

    # Success Email
    email_success = EmailOperator(
        task_id='send_success_email',
        to='ananthareddy12321@gmail.com',
        subject='Stock Prediction Pipeline Success - {{ ds }}',
        html_content=EMAIL_SUCCESS_TEMPLATE
    )

    # Failure Email
    email_failure = EmailOperator(
        task_id='send_failure_email',
        to='ananthareddy12321@gmail.com',
        subject='Stock Prediction Pipeline Failed - {{ ds }}',
        html_content=EMAIL_FAILURE_TEMPLATE,
        trigger_rule='one_failed'
    )

    # End pipeline
    end = DummyOperator(
        task_id='end_pipeline',
        trigger_rule=TriggerRule.NONE_FAILED_OR_SKIPPED
    )

    # Define task dependencies
    start >> acquire_data >> preprocess_data >> validate_data >> [detect_bias_task, mitigate_bias_task]
    
    mitigate_bias_task >> [calculate_statistics_task, detect_anomalies_task, train_model_task]
    train_model_task >> evaluate_model_bias_task
    [calculate_statistics_task, detect_anomalies_task, evaluate_model_bias_task] >> email_success
    
    [email_success, email_failure] >> end

    logger.info("DAG initialization completed")

if __name__ == "__main__":
    dag.cli()
