import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

def setup_task_logger(dag_id: str, task_id: str, log_dir: str = None) -> logging.Logger:
    """
    Set up a logger for Airflow tasks with both file and console handlers
    
    Parameters:
    -----------
    dag_id : str
        Name of the DAG
    task_id : str
        Name of the task
    log_dir : str
        Directory to store log files (default: AIRFLOW_HOME/logs/dag_id)
    
    Returns:
    --------
    logging.Logger
    """
    # Set up log directory
    if log_dir is None:
        airflow_home = os.getenv('AIRFLOW_HOME', '/opt/airflow')
        log_dir = os.path.join(airflow_home, 'logs', dag_id)
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger_name = f"{dag_id}.{task_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    formatter = logging.Formatter(
        '[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    log_file = os.path.join(log_dir, f"{task_id}_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger 