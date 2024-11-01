"""
The Airflow DAG for Stock Prediction Pipeline
"""
import logging
import sys

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
    'anomalies': os.path.join(DATA_DIR, 'anomalies')
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
        logger.info(f"Processing data from: {input_folder}")
        
        # Initialize analyzer with data
        analyzer = BiasAnalyzer(input_folder=input_folder)
        
        # Load and process data
        analyzer.load_data()
        
        # Generate report
        report = analyzer.generate_report()
        
        logger.info("Bias detection completed successfully")
        return report
        
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
            'tickers': ["AAPL", "GOOGL", "TSLA", "MSFT", "AMZN"],
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
    start >> acquire_data >> preprocess_data >> validate_data >> detect_bias_task
    
    detect_bias_task >> mitigate_bias_task >> [calculate_statistics_task, detect_anomalies_task]
    
    [calculate_statistics_task, detect_anomalies_task] >> email_success
    
    [email_success, email_failure] >> end

    logger.info("DAG initialization completed")

if __name__ == "__main__":
    dag.cli()
