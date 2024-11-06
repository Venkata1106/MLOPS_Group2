# Stock Market Price Prediction

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Compatible-brightgreen.svg)](https://www.docker.com/)
[![Apache Airflow](https://img.shields.io/badge/Airflow-Pipeline-yellowgreen.svg)](https://airflow.apache.org/)

- [Vekata Anantha Reddy Arikatla](https://github.com/Venkata1106)
- [Dheeraj Kumar Goli](https://github.com/dheeraj932)
- [Shubhang Yadav Sandaveni](https://github.com/sandavenishubhang)
- [Manasi Bondalapati](https://github.com/thunderblu)
- [Sachi Hareshkumar Patel](https://github.com/Sachiprogrammer)
- [Pranav Pinjarla](https://github.com/pranav10510)

<p align="center">  
    <br>
	<a href="#">
	      <img src="https://raw.githubusercontent.com/Thomas-George-T/Thomas-George-T/master/assets/python.svg" alt="Python" title="Python" width ="120" />
        <img height=100 src="https://cdn.svgporn.com/logos/airflow-icon.svg" alt="Airflow" title="Airflow" hspace=20 /> 
        <img height=100 src="https://cdn.svgporn.com/logos/docker-icon.svg" alt="Docker" title="Docker" hspace=20 /> 
  </a>	
</p>
<br>

# Introduction 
In today’s rapidly evolving financial landscape, investors seek tools that empower them to make smarter, well-timed investment decisions, optimizing returns while managing risk. This project addresses that need by creating a comprehensive pipeline for stock market analysis and price prediction, offering actionable insights to support investment strategies and enhance financial security. Leveraging historical data from the yfinance library, it builds an advanced forecasting model using Long Short-Term Memory (LSTM) networks and Random Forest to capture subtle patterns in stock price fluctuations. Key metrics—open, close, high, low, and volume—form the foundation of the model, enabling investors to anticipate price trends and make informed choices on when to buy, sell, or hold stocks. Beyond predictions, the pipeline ensures data integrity and reliability through data validation, anomaly detection, and bias mitigation. A DataValidator class verifies schema accuracy, data type consistency, and date continuity, while a BiasAnalyzer and AnomalyDetector guard against hidden biases and detect unusual activity in price, volume, and volatility. The StockAnalyzer further enhances insights by exploring return metrics, price trends, volume fluctuations, and technical indicators, offering a multi-dimensional view of stock performance. This end-to-end solution enables investors to make data-driven choices that reduce risks, align with financial goals, and build wealth strategically in a competitive market. 

# Dataset Information

This repository includes comprehensive historical stock data for all stocks over various periods. The datasets consistently include the following features:

- **Daily Stock Price Details:**
  - Date
  - Open
  - High
  - Low
  - Close
  - Adjusted Close
  - Volume

- **Calculated Metrics:**
  - Daily Returns
  - Moving Averages (e.g., 5-day and 20-day)

- **Technical Indicators:**
  - Volatility

- **Additional Features (in some datasets):**
  - Anomaly Detection Flags

## Data Card
- Size: Varies depending on the ticker, period, and data type requested
- Typical Data Shape: For historical data over one year, e.g., 252 rows × 7 columns for daily prices

## Data Types
  
| Variable Name |Role|Type|Description|
|:--------------|:---|:---|:----------|
|Date |Index	|Date	|The specific date of each observation; serves as the index for time-series data |
|Open |Feature	|Continuous	|Opening price of the stock on that day |
|High|Feature	|Continuous	|Highest price of the stock on that day |
|Low	|Feature	|Continuous	|	Lowest price of the stock on that day |
|Close	|Feature	|Continuous	|Closing price of the stock on that day |
|Adj Close	|Feature	|Continuous	|Adjusted closing price, accounting for corporate actions (e.g., splits and dividends) |
|Volume	|Feature	|Integer	|otal trading volume of the stock on that day |

## Data Sources 
The data is taken from [Yahoo Finance](https://finance.yahoo.com/?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuZ29vZ2xlLmNvbS8&guce_referrer_sig=AQAAADqDhv9AiH713iygoGkr7T_2AkvprC83Wi-4S6lJRfBBgbNwhKsSZxS8uO6DTQaCHttCPJJhSvss5cNPWD1PV7mqEX1hZEdaxwXK9JdfizmtkP025bjoPF3avzduzOzvmEx7GUvZgSDZ7BRiRJfOyjTMZ1s_oAuLiy55hxFJCGBT) by using **yfinance** python library

## Prerequisities
1. Docker and Docker Compose: Required to containerize and orchestrate services.
2. Python 3.8+: Required for development and testing.
3. Airflow 2.x: Required for scheduling and DAG monitoring.
4. Pandas: Requires 2.2.2 version
5. yfinance: Require 0.2.40 version

## User Installation
The steps for User installation are as follows:

1. Clone Repository:
```
git clone https://github.com/Venkata1106/MLOPS_Group2.git
cd MLOPS_Group2
```
2. Run Docker Setup:
```
docker-compose up --build
```
3. Visit localhost:8080 login with credentials

```
user:admin 
password:admin
```

4. Email Alerts Configuration:
```
Connection ID: email_default
Connection Type: Email
Host: smtp.gmail.com
login: "your_email@example.com"
password: "your_password"
Port: 587
Extra:
{
  "smtp_ssl": false,
  "smtp_starttls": true
}
```
5. Run the DAG by clicking on the play button on the right side of the window
6. Stop docker containers
```docker
docker compose down
```

# Tools Used for MLOps

- Docker
- Airflow
- DVC

## Docker and Airflow

The `docker-compose.yml` file is used to set up and run Apache Airflow with all necessary dependencies in Docker containers. By leveraging containerization, it ensures that our data pipeline is portable and runs consistently across various platforms, whether on Windows, macOS, or Linux. This approach simplifies deployment and minimizes compatibility issues, allowing the pipeline to run smoothly on any system.

## DVC 

Data Version Control (DVC) is an open-source tool designed to handle versioning and management of datasets and machine learning models. It facilitates the tracking of data and model changes over time, allowing teams to maintain consistent, reproducible workflows. By storing only metadata in the Git repository while keeping the large data files and models in cloud storage or other remote locations, DVC ensures that the version history remains lightweight and efficient. It integrates seamlessly with Git, enabling easy collaboration between teams, where code is managed with Git and data and models are versioned with DVC. This makes it an essential tool for maintaining traceability, reproducibility, and organization in machine learning projects.

### Folder Structure Overview
```
main/
├── .dvc/                          # DVC folder for data versioning
├── .dockerignore                  # Docker build ignore file
├── .gitignore                     # Git ignore file
├── Dockerfile                     # Docker configuration
├── README.md                      # Project overview and folder structure
├── config/                        # Configuration files
├── dags/                          # Airflow DAGs
│   └── stock_prediction_dag.py    # Main DAG for stock prediction
├── Images/                        # Saved images
├── data/                          # Main data directory
│   ├── raw/                       # Raw data files (e.g., AAPL.csv, AMZN.csv)
│   ├── processed/                 # Preprocessed data (e.g., processed_AAPL.csv)
│   ├── validated/                 # Validated data (e.g., validated_AAPL.csv)
│   ├── analyzed/                  # Analyzed data (e.g., analyzed_processed_AAPL.csv)
│   ├── anomalies/                 # Anomaly detection results (e.g., anomalies.json)
│   └── mitigated/                 # Bias-mitigated data (e.g., mitigated_AAPL.csv)
├── scripts/                       # Utility scripts
│   ├── __init__.py                # Package init file
│   ├── data_acquisition.py        # Data Acquisition script
│   ├── data_preprocessing.py      # Data Preprocessing script
│   └── ...                        # Other utility scripts
├── tests/
│   ├── test_data_acquisition.py   # Test Data Acquisition script
│   ├── test_data_preprocessing.py # Test Data Preprocessing script
│   └── ...                        # Test cases for the project under future implementations
├── dvc.yaml                       # DVC pipeline configuration
├── docker-compose.yml             # Docker Compose configuration
├── requirements.txt               # Required Python dependencies
└── smtp.env                       # SMTP configuration

```


# Data Pipeline

The pipeline is designed to manage stock market data from acquisition to analysis, incorporating several stages to ensure the reliability and accuracy of predictions. Here’s an overview:

![DAG](Images/DAG.jpeg)
![Gantt](Images/Gantt.jpeg)

## Scripts

1. **[Data Acquisition:](https://github.com/Venkata1106/MLOPS_Group2/blob/main/scripts/data_acquisition.py)** 
The pipeline starts by fetching historical stock data from Yahoo Finance using yfinance. This data includes core financial metrics like open, close, high, low, and volume, which serve as essential inputs for further processing.

2. **[Data Preprocessing:](https://github.com/Venkata1106/MLOPS_Group2/blob/main/scripts/data_preprocessing.py)** 
After acquisition, raw data is cleaned and preprocessed. Missing values are filled, and technical indicators such as moving averages, returns, and volatility are calculated to enrich the dataset. This stage ensures that data quality is maintained, creating a solid foundation for subsequent analysis.

3. **[Data Validation:](https://github.com/Venkata1106/MLOPS_Group2/blob/main/scripts/data_validation.py)** 
Ensuring data integrity, this stage verifies schema accuracy, consistent data types, reasonable value ranges, and continuous date records. Validation helps avoid errors in model predictions by detecting inconsistencies or incomplete data early on.

4. **[Bias Detection and Mitigation:](https://github.com/Venkata1106/MLOPS_Group2/blob/main/scripts/bias_detection.py)** 
To improve model fairness, a bias analysis checks for patterns that may lead to biased predictions. If biases are detected, mitigation techniques like winsorization (to handle outliers) and normalization are applied to balance the data distribution.

5. **[Anomaly Detection:](https://github.com/Venkata1106/MLOPS_Group2/blob/main/scripts/anomaly_detection.py)** 
This stage identifies unusual patterns in stock prices, volume, and volatility using methods like Isolation Forest. Detecting anomalies helps flag abnormal market behavior, enabling the model to account for irregularities that might affect prediction accuracy.

6. **[Statistical Analysis:](https://github.com/Venkata1106/MLOPS_Group2/blob/main/scripts/data_statistics.py)** 
To understand deeper patterns, return-based metrics, price trends, and volume analysis are conducted, offering a comprehensive view of stock performance. This stage also includes correlation studies to identify relationships between price movements and volume fluctuations.

7. **Success and Failure Notification:** 
At the end of the pipeline, a notification system sends an email summarizing the pipeline execution status. A success email confirms that the pipeline completed smoothly and shares a brief summary of key insights or metrics. In case of failure, an email is sent detailing the specific error encountered, helping to address issues promptly.

8. **Output and Insights Generation:** 
The processed, validated, and analyzed data is saved, providing actionable insights into stock trends. These insights serve as valuable inputs for predictive models, enabling investors to make data-driven decisions with a clearer understanding of market dynamics.

## Tests

1. **[Data Acquisition:](https://github.com/Venkata1106/MLOPS_Group2/blob/main/tests/test_data_acquisition.py)** 
   - Tests the `fetch_stock_data` function for acquiring stock data, specifically for Apple Inc. (AAPL).
   - Verifies that the function creates the expected output directory and file (AAPL.csv).
   - Checks if the acquired data contains the required columns: Open, High, Low, Close, and Volume.
   - Ensures that the fetched data is not empty.


3. **[Data Preprocessing:](https://github.com/Venkata1106/MLOPS_Group2/blob/main/tests/test_data_preprocessing.py)**
   - Tests the `process_and_save_all_data` function using synthetic stock data.
   - Sets up a test environment with input and output directories, creating test data with known properties.
   - Verifies that the preprocessing adds required columns (Returns, MA5, MA20) and maintains data integrity.
   - Includes setup and teardown methods to create and clean up test data and directories, ensuring a clean test environment for each run.


4. **[Data Validation:](https://github.com/Venkata1106/MLOPS_Group2/blob/main/tests/future_implementations/test_data_validation.py)**
   - Tests the ⁠ `DataValidator` ⁠ class using both valid and invalid stock data.
   - Checks various validation functions including schema validation, data type validation, value range validation, date continuity, and price consistency.
   - Includes setup and teardown methods to create and clean up test data and directories.
   - Tests the full validation process with invalid data to ensure it correctly identifies issues.


5. **[Bias Detection:](https://github.com/Venkata1106/MLOPS_Group2/blob/main/tests/future_implementations/test_bias_detection.py)**
   - Tests the ⁠ `BiasAnalyzer` ⁠ class using sample data with intentional NaN values and gaps.
   - Checks for sampling bias, survivorship bias, and time period bias detection.
   - Verifies data loading functionality and the accuracy of bias detection methods.
   - Includes setup and teardown methods for creating and cleaning test environments.


6. **[Bias Mitigation:](https://github.com/Venkata1106/MLOPS_Group2/blob/main/tests/future_implementations/test_bias_mitigation.py)** 
   - Tests the ⁠ `BiasMitigator` ⁠ class using synthetic stock data with intentional biases and gaps.
   - Verifies various bias mitigation techniques including handling missing dates, normalizing features, and adding technical indicators.
   - ⁠Checks the full bias mitigation process, ensuring successful execution and output generation.
   - Includes setup and teardown methods for creating and cleaning test data and directories.

7. **[Data Statistics:](https://github.com/Venkata1106/MLOPS_Group2/blob/main/tests/future_implementations/test_data_statistics.py)**
   - ⁠Tests the ⁠ `StockAnalyzer` ⁠ class using synthetic stock data.
   - ⁠Checks various analysis functions including returns metrics, price metrics, volume metrics, and technical indicators.
   - Tests correlation metrics calculation and the full analysis process.
   - ⁠Includes class-level setup and teardown methods to create and clean up test data and output directories.

8. **[Anomaly Detection:](https://github.com/Venkata1106/MLOPS_Group2/blob/main/tests/future_implementations/test_anomaly_detection.py)** 
   - Tests the ⁠ `AnomalyDetector` ⁠ class using synthetic stock data with known anomalies.
   - ⁠Verifies feature calculation, anomaly detection process, and output generation.
   - Checks the convenience function ⁠ `run_anomaly_detection` ⁠ and error handling for invalid inputs.
   - Includes methods to create test data with price and volume anomalies.
