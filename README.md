# Stock Price Prediction

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Compatible-green.svg)](https://www.docker.com/)
[![Apache Airflow](https://img.shields.io/badge/Airflow-Pipeline-yellowgreen.svg)](https://airflow.apache.org/)
[![Vertex AI](https://img.shields.io/badge/Vertex%20AI-Deployment-blueviolet.svg)](https://cloud.google.com/vertex-ai)

---

## Video Link
[Click Here to Watch the Project Walkthrough Video](#)  


---

## Introduction
This project provides an end-to-end MLOps pipeline for stock price prediction, integrating data ingestion, preprocessing, validation, bias detection & mitigation, anomaly analysis, model training, hyperparameter tuning, experiment tracking (MLflow), and deployment on Google Cloud’s Vertex AI. The pipeline ensures reproducibility, fairness, scalability, and transparency. A Streamlit-based web interface allows users to interact with the deployed model’s predictions.

---

## Dataset Card

**Data Source:** Yahoo Finance via `yfinance`.

**Data Size:** Approximately ~252 rows × 7 columns per year, per stock.

**Features:**
- **Date, Open, High, Low, Close, Adj Close, Volume:** Core stock metrics per trading day.
- **Derived Metrics:** Returns, Moving Averages (e.g., MA5, MA20), Volatility indicators, and Anomaly flags.

These enriched features enhance the model’s predictive capability.

---

## Prerequisites
- **Python 3.8+**
- **Docker & Docker Compose**
- **Apache Airflow 2.x**
- **DVC** for data version control
- **MLflow** for experiment tracking
- **Vertex AI (GCP)** for model deployment
- **Terraform (Optional)** for Infrastructure as Code
- Core Python libraries: `pandas`, `yfinance`, `scikit-learn`, `xgboost`, `fairlearn`, `shap`, `lime`

---

## Steps to Execute the Data Pipeline

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Venkata1106/MLOPS_Group2.git
   cd MLOPS_Group2
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Docker and Airflow Setup:**
   ```bash
   docker-compose up --build
   ```
   Access Airflow at `http://localhost:8080` (user: `admin`, password: `admin`).

4. **Trigger the DAG:**
   Open Airflow UI → Run `stock_prediction_dag`.

5. **DVC (Optional):**
   ```bash
   dvc pull
   ```
   Ensures correct data versions are used.

6. **MLflow Experiment Tracking:**
   ```bash
   mlflow ui
   ```
   Access at `http://localhost:5000` for comparing runs, metrics, and artifacts.

---

## Data Pipeline Steps

1. **Data Acquisition:** Fetch raw historical stock data from Yahoo Finance.
2. **Data Preprocessing & Validation:** Clean data, handle missing values, compute returns and MAs, and ensure schema correctness.
3. **Bias Detection & Mitigation:** Identify unfair performance on certain slices (e.g., volatility-based) and rebalance the data accordingly.
4. **Anomaly & Slice Analysis:** Detect unusual patterns and ensure the model is fair and robust across different market conditions.
5. **Notifications & Logging:** Inform stakeholders of pipeline completion, successes, or failures.
6. **Data Versioning:** DVC tracks data changes for reproducibility.

---

## Explaining the `src` Directory (Data Pipeline Related)

- **data_acquisition.py:** Fetches raw stock data using `yfinance`.
- **data_preprocessing.py:** Cleans and enriches data (returns, MAs, volatility).
- **data_validation.py:** Ensures data meets schema requirements, checks continuity and no critical nulls.
- **bias_detection.py & bias_mitigation.py:** Identify bias in subsets and apply mitigation strategies.
- **slice_analysis.py:** Analyze performance on subsets (e.g., bull vs. bear markets).
- **utils/** (e.g., `logging_config.py`, `monitoring.py`): Logging, monitoring, and utility functions that support pipeline transparency and debugging.

---

## MODEL DEVELOPMENT

After the data pipeline prepares and verifies data quality, the ML development phase focuses on training, evaluating, and selecting the best model.

### Folder Structure Overview (Main Files Only)
```bash
MLOPS_Group2/
├── README.md
├── requirements.txt
├── modelrequire.txt
├── docker-compose.yml
├── cloudbuild.yaml
├── config/
│   ├── model_config.yml
│   └── pipeline_config.yml
├── credentials/
├── data/
│   ├── raw.dvc
│   ├── processed.dvc
│   ├── mitigated/
│   └── stats/
├── dags/
│   └── stock_prediction_dag.py
├── docker/
│   └── airflow/
│       └── Dockerfile
├── docs/
│   └── bias_mitigation.md
├── images/
│   ├── DAG.jpeg
│   └── Gantt.jpeg
├── models/
│   ├── bias_checker.py
│   ├── bias_detection/
│   ├── data_loader.py
│   ├── experiment_tracker.py
│   ├── hyperparameter_tuner.py
│   ├── model.py
│   ├── model_registry.py
│   ├── model_selector.py
│   ├── model_validator.py
│   ├── sensitivity_analyzer.py
│   ├── train.py
│   └── utils/
│       └── logger.py
├── model_results/
│   ├── metrics_heatmap.png
│   └── performance_comparison.png
├── scripts/
│   ├── check_bias.py
│   ├── deploy_model.py
│   ├── rollback_model.py
│   ├── send_notifications.py
│   ├── test_endpoint.py
│   ├── test_pipeline.py
│   └── validate_model.py
├── src/
│   ├── data_acquisition.py
│   ├── data_preprocessing.py
│   ├── data_validation.py
│   ├── bias_detection.py
│   ├── bias_mitigation.py
│   ├── slice_analysis.py
│   └── utils/
│       ├── logging_config.py
│       └── monitoring.py
└── tests/
    ├── test_basic.py
    ├── test_data_acquisition.py
    ├── test_data_preprocessing.py
    ├── test_data_validation.py
    └── conftest.py
```

### Model Scripts (In `models` Directory)

- **data_loader.py:** Splits data into train/val/test sets, imputes missing values, and ensures the model receives a consistent dataset.
- **experiment_tracker.py:** Integrates with MLflow to log experiments, hyperparameters, metrics, and artifacts, facilitating run comparisons.
- **hyperparameter_tuner.py:** Automates search for optimal hyperparameters (e.g., Grid/Random Search) to improve model accuracy and stability.
- **model.py:** Defines training logic for ML models (e.g., Random Forest, XGBoost).
- **model_registry.py:** Stores and version-controls the best model, ensuring it can be easily deployed.
- **model_selector.py:** Chooses the best model based on validation metrics and fairness scores.
- **model_validator.py:** Tests the selected model on a hold-out test set for unbiased performance estimation.
- **sensitivity_analyzer.py:** Uses SHAP/LIME to explain feature importance, enhancing model interpretability.
- **bias_checker.py:** Evaluates model performance across slices (e.g., volatility quartiles) to ensure fairness.

### Running Model Files
To run model scripts, install additional dependencies:
```bash
pip install -r modelrequire.txt
```

You can then train and evaluate models:
```bash
python models/train.py
```

### Visualizations and Results
- **metrics_heatmap.png & performance_comparison.png:** Compare different model runs or hyperparameters visually.
- **Bias Reports (JSON/PNG):** Show if any subgroups underperform.
- **SHAP/LIME Plots:** Visualize feature importance and model reasoning.
- **MLflow UI:** By running `mlflow ui`, access a browser interface to compare runs, metrics, and artifacts.

*(MLflow screenshot example)*  
![MLflow UI Example](images/mlflow_example.png)  
*(Ensure you have such images ready. If missing, ask ChatGPT for suggestions.)*

---

## MODEL DEPLOYMENT

We trained and built the model using Vertex AI’s AutoML functionality on GCP, which helped automatically select a high-performing model. The model was then deployed to a Vertex AI endpoint for real-time predictions.

### Steps for Deployment

1. **Set Up GCP Environment:** Configure GCP authentication, set required IAM permissions, and ensure Vertex AI and required APIs are enabled.

2. **From `vertex_ai` Directory:**
   ```bash
   pip install -r requirement.txt
   # Authenticate GCP
   gcloud auth login
   # Run prediction script
   python predict.py
   # Run Streamlit app
   streamlit run app.py
   ```
   The Streamlit interface allows end-users to interact with the model endpoint seamlessly.

3. **Model Endpoint & Serving:**  
   Vertex AI hosts the model and provides a scalable endpoint. Users can send requests to get predictions. Model monitoring tools track performance over time, detecting model drift, skew, or degradation in accuracy. Threshold-based triggers can initiate retraining or rolling back to a previous model version.

4. **Model Monitoring, Data Drift, and Skew:**
   - **Model Monitoring (GCP Monitoring):** Keeps track of prediction quality, latency, and errors.
   - **Data Drift & Skew Detection:** When new data patterns emerge (e.g., stock market regime change), monitored metrics can signal the need for retraining.
   - **Continuous Improvement:** If performance drops below thresholds, CI/CD pipelines integrated with GitHub Actions can trigger automatic retraining using Vertex AI AutoML, ensuring the model remains up-to-date and accurate.

*(Deployment architecture diagram or screenshots can be placed here)*  
![Vertex AI Diagram](images/vertex_ai_deployment.png)  
*(If missing images, request ChatGPT for placeholders or conceptual diagrams.)*

---

## Full Project Flowchart

```mermaid
flowchart TD

A[Start] --> B[Data Acquisition]
B --> C[Preprocessing & Validation]
C --> D[Bias Detection & Mitigation]
D --> E[Anomaly & Slice Analysis]
E --> F[Model Training & Hyperparam Tuning]
F --> G[Model Validation & Selection]
G --> H[Model Registry]
H --> I[Vertex AI Deployment]
I --> J[Monitoring & Retraining]
J --> K[Front-End (Streamlit)]
K --> L[End]
```

---

## Tools and Technologies Used

| Tool/Technology | Purpose                                        |
|-----------------|------------------------------------------------|
| Python 3.8+     | Core language for data pipelines & ML          |
| Apache Airflow   | Orchestrate & schedule data pipelines          |
| Docker           | Containerization for reproducible envs         |
| DVC              | Data versioning for reproducibility            |
| MLflow           | Track experiments, metrics, and artifacts      |
| Vertex AI        | AutoML model training & endpoint deployment    |
| Terraform (Opt.) | Infrastructure as Code for cloud resources     |
| Pandas, yfinance | Data manipulation & acquisition                |
| Scikit-learn, XGBoost | Core modeling frameworks                 |
| SHAP, LIME       | Model explainability & feature importance      |
| Fairlearn        | Bias detection & mitigation                    |
| GCP Monitoring   | Observing model performance & triggering retraining |
| Streamlit        | User-friendly front-end interface              |

---

## Cost and Resource Estimation

- **Local Environment:** Minimal additional cost, primarily developer time and local resources.
- **Cloud (GCP Vertex AI):**  
  - Compute costs for training and hosting endpoints.
  - Storage for datasets, model artifacts, and logs.
  - Artifact Registry for Docker images.
  
A small-scale setup might cost around ~$30-$60/month, escalating with increased data, complexity, and usage.

---

## Conclusion

This MLOps solution delivers a robust, scalable, and fair pipeline for stock price prediction. By integrating top-tier tools—Airflow, Docker, DVC, MLflow, Vertex AI, and Fairlearn—the pipeline ensures reliable data handling, transparent model experimentation, thorough fairness checks, automated deployment, and continuous monitoring. The Streamlit interface makes predictive insights accessible and actionable, enabling data-driven decision-making with confidence.
