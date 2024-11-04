# Stock Prediction Pipeline

A comprehensive MLOps pipeline for stock prediction using Airflow for orchestration, implementing data validation, bias detection, anomaly detection, and automated notifications.

## Project Structure


bash
/Project Repo
├── dags/
│ └── stock_prediction_dag.py
├── scripts/
│ ├── data_statistics.py
│ ├── bias_mitigation.py
│ └── anomaly_detection.py
├── tests/
│ └── test_.py
├── logs/
├── data/
└── README.md

## Features

- **Data Acquisition**: Automated stock data fetching using Yahoo Finance API
- **Data Preprocessing**: Comprehensive data cleaning and feature engineering
- **Bias Detection & Mitigation**: Implementation of bias detection through data slicing
- **Anomaly Detection**: Statistical analysis for identifying market anomalies
- **Email Notifications**: Automated alerts for pipeline success/failure
- **Data Validation**: Statistical analysis and data quality checks

## Prerequisites

- Docker
- Python 3.8+
- Apache Airflow 2.x
- Gmail account with App Password for notifications

## Installation & Setup

1. Clone the repository:
bash
git clone [your-repo-url]
cd [your-repo-name]

2. Set up environment variables:
```bash
export AIRFLOW__SMTP__SMTP_HOST=smtp.gmail.com
export AIRFLOW__SMTP__SMTP_PORT=587
export AIRFLOW__SMTP__SMTP_USER=your_email@gmail.com
export AIRFLOW__SMTP__SMTP_PASSWORD=your_app_password
export AIRFLOW__SMTP__SMTP_MAIL_FROM=your_email@gmail.com
export AIRFLOW__SMTP__SMTP_SSL=False
export AIRFLOW__SMTP__SMTP_STARTTLS=True
```

3. Configure Airflow email connection:
```bash
airflow connections add email_default \
    --conn-type smtp \
    --conn-host smtp.gmail.com \
    --conn-login your_email@gmail.com \
    --conn-password your_app_password \
    --conn-port 587 \
    --conn-extra '{"smtp_starttls": true, "smtp_ssl": false}'
```

## Pipeline Components

1. **Data Acquisition**
   - Fetches stock data using Yahoo Finance API
   - Validates data completeness and quality

2. **Data Preprocessing**
   - Handles missing values
   - Calculates technical indicators
   - Normalizes data

3. **Bias Detection & Mitigation**
   - Implements data slicing for bias detection
   - Applies bias mitigation techniques
   - Generates bias reports

4. **Anomaly Detection**
   - Identifies market anomalies
   - Generates alerts for unusual patterns
   - Statistical validation of anomalies

5. **Pipeline Monitoring**
   - Email notifications for success/failure
   - Logging of all operations
   - Performance monitoring