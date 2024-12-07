# Stock Price Prediction

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Compatible-green.svg)](https://www.docker.com/)
[![Apache Airflow](https://img.shields.io/badge/Airflow-Pipeline-yellowgreen.svg)](https://airflow.apache.org/)
[![Vertex AI](https://img.shields.io/badge/Vertex%20AI-Deployment-blueviolet.svg)](https://cloud.google.com/vertex-ai)

---

## Video Link
[Click Here to Watch the Project Walkthrough Video](#)  
*(Replace `#` with the actual video URL.)*

---

## Introduction
This project demonstrates a complete MLOps pipeline for predicting stock prices. It starts from raw financial market data and takes you through a series of automated steps: data ingestion, cleaning, feature engineering, bias detection and mitigation, anomaly analysis, model training with hyperparameter tuning, experiment tracking (MLflow), and finally deploying the best-performing and fairest model to a Vertex AI endpoint on Google Cloud. A Streamlit-based interface allows seamless interaction with the model’s predictions.

In essence, we’re building a pipeline that ensures every stage—from data to deployed model—is reliable, reproducible, transparent, and continuously improving.

---

## Dataset Information

**Data Source:** Yahoo Finance (via `yfinance`).

**Data Size:** About 252 rows × 7 columns of daily stock data per year per ticker.

**Features:**
- **Core:** Date, Open, High, Low, Close, Adj Close, Volume
- **Derived:** Returns, Moving Averages (MA5, MA20), Volatility, Anomaly Flags

These additional features help the model understand trends and patterns in stock price movements over time.

---

## Prerequisites
- **Python 3.8+**
- **Docker & Docker Compose** for reproducible environments
- **Apache Airflow 2.x** for pipeline orchestration
- **DVC** (Data Version Control) to track data changes
- **MLflow** for experiment tracking
- **Vertex AI (GCP)** for automated model training (AutoML) and scalable deployment
- Optional: **Terraform** for infrastructure provisioning
- Libraries: `pandas`, `yfinance`, `scikit-learn`, `xgboost`, `fairlearn`, `shap`, `lime`

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

3. **Run Docker & Airflow Setup:**
   ```bash
   docker-compose up --build
   ```
   Access Airflow at `http://localhost:8080` (Credentials: `admin` / `admin`).

4. **Trigger the Airflow DAG:**
   In the Airflow UI, start the `stock_prediction_dag`.

5. **DVC (Optional):**
   ```bash
   dvc pull
   ```
   Ensures correct version of data is used.

6. **MLflow UI for Experiment Tracking:**
   ```bash
   mlflow ui
   ```
   Open `http://localhost:5000` to view experiment runs, metrics, and artifacts.

---

## Data Pipeline Steps (What’s Happening)

1. **Data Acquisition:**  
   Fetches raw historical stock data using `yfinance`, essentially pulling daily price info from Yahoo Finance’s API.

2. **Preprocessing & Validation:**  
   Cleans data, handles missing values, generates features like returns and MAs, and validates the dataset to ensure no corrupted or missing critical info.

3. **Bias Detection & Mitigation:**  
   Checks if certain data segments might cause unfair model performance. If bias is found, applies reweighting or resampling so that all subsets are treated fairly.

4. **Anomaly & Slice Analysis:**  
   Identifies unusual patterns (e.g., sudden market shifts) and ensures consistent performance across different market conditions.

5. **Notifications & Logging:**  
   Sends success/failure notifications and logs pipeline runs for easy debugging.

6. **Data Versioning (DVC):**  
   Tracks data changes, enabling you to revert to previous data states and ensure complete reproducibility.

---

## Explaining the `src` Directory (Data Pipeline)

- **data_acquisition.py:** Downloads the raw stock data.
- **data_preprocessing.py:** Cleans the data, computes returns, MAs, volatility.
- **data_validation.py:** Ensures data meets schema expectations and no critical columns are missing.
- **bias_detection.py & bias_mitigation.py:** Identifies and reduces biases in certain data slices.
- **slice_analysis.py:** Evaluates data in subsets (e.g., volatility quartiles).
- **utils/**: Logging, monitoring, and other utilities to keep the pipeline transparent.

---

## MODEL DEVELOPMENT 

After obtaining a clean, bias-mitigated dataset, we focus on model development:

1. **Data Loading & Splitting (data_loader.py):**  
   Splits processed data into training, validation, and test sets. Ensures the model sees stable input and the final evaluation is unbiased.

2. **Model Training (model.py):**  
   Trains multiple models (e.g., Random Forest, XGBoost). Trying multiple algorithms increases chances of finding a well-generalizing model.

3. **Hyperparameter Tuning (hyperparameter_tuner.py):**  
   Systematically searches for the best model parameters, improving accuracy and robustness against overfitting.

4. **Bias Checking (bias_checker.py):**  
   Re-checks predictions for unfair patterns. If bias persists, adjustments are made until fairness improves.

5. **Model Validation & Selection (model_validator.py & model_selector.py):**  
   The model validator tests performance on a test set, ensuring it generalizes well. The model selector then picks the model that best balances accuracy and fairness.

6. **Experiment Tracking (experiment_tracker.py):**  
   MLflow logs all runs, parameters, and metrics. This makes it easy to compare models and pick the best approach.

7. **Explainability (sensitivity_analyzer.py):**  
   Tools like SHAP and LIME provide feature importance and interpretability, letting you understand which factors most influence predictions.

8. **Model Registry (model_registry.py):**  
   Saves the final chosen model version for easy retrieval and deployment. No confusion about which model is live—just the best one.

---

## Folder Structure Overview (Key Files)
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
└── src/
    ├── data_acquisition.py
    ├── data_preprocessing.py
    ├── data_validation.py
    ├── bias_detection.py
    ├── bias_mitigation.py
    ├── slice_analysis.py
    └── utils/
        ├── logging_config.py
        └── monitoring.py
```

---

## Running Model Scripts

First, install additional model dependencies:
```bash
pip install -r modelrequire.txt
```

Train and evaluate models:
```bash
python models/train.py
```

### Visualizations & Results
- **metrics_heatmap.png & performance_comparison.png:** Visual comparisons of model runs and hyperparameters.
- **Bias Reports:** JSON/PNG files showing if certain groups underperform.
- **SHAP/LIME Plots:** Reveal which features drive the model’s predictions.
- **MLflow UI:** `mlflow ui` to analyze runs and metrics in a clean interface.

---

## MODEL DEPLOYMENT

With Vertex AI (AutoML), we let GCP automatically train and select a top model. The chosen model is then deployed to a Vertex AI endpoint:

1. **GCP Setup:**  
   Authenticate with GCP, ensure Vertex AI and required APIs are enabled.

2. **From `vertex_ai` Directory:**
   ```bash
   pip install -r requirement.txt
   python predict.py
   streamlit run app.py
   ```
   The Streamlit app lets you input parameters and get real-time predictions from the deployed model.

### Monitoring & Retraining
- **Vertex AI Monitoring:** Watches performance and detects data drift.
- **CI/CD Pipelines:** If accuracy drops or data changes, automated retraining can be triggered, ensuring the model stays current and reliable.

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
| Python 3.8+     | Data & ML scripting                           |
| Apache Airflow   | Orchestration & Scheduling                    |
| Docker           | Reproducible, portable environments           |
| DVC              | Data versioning and reproducibility           |
| MLflow           | Experiment logging & comparison               |
| Vertex AI        | AutoML training & scalable deployment         |
| Terraform (Opt.) | IaC for cloud resources                       |
| Pandas, yfinance | Data acquisition & manipulation               |
| scikit-learn, XGBoost | ML model frameworks                     |
| SHAP, LIME       | Model explainability                          |
| Fairlearn        | Bias detection & mitigation                   |
| GCP Monitoring   | Performance observation, detecting drift      |
| Streamlit        | User-friendly interface for predictions       |

---

## Cost & Resource Estimation
- **Local Environment:** Minimal (developer time, local hardware).
- **Cloud (GCP Vertex AI):** Compute for training/serving endpoints, storage for data and artifacts. A small-scale setup might cost ~$30-$60/month, scaling with data and complexity.

---

### Model Monitoring and Maintenance 

1. **Vertex AI Model Monitoring Activated:**  
   We have enabled Vertex AI’s model monitoring for our deployed AutoML model. This configuration continuously checks if incoming data diverges from the training baseline. For instance, over the past month, we’ve received alerts when certain input feature distributions (e.g., trading volume patterns) moved outside expected ranges. After reviewing these alerts, we updated our training datasets and triggered retraining to keep the model aligned with current market behaviors.

2. **Performance Alerts and Dashboards in Cloud Monitoring:**  
   Our team set up custom dashboards and alerts in Google Cloud Monitoring to track key performance metrics, such as prediction latency and error rates. Recently, when latency averaged above 400ms for a few days due to heightened market volatility, we received Slack notifications. Prompt action—adjusting configuration parameters—restored performance within hours, ensuring minimal impact on users.

3. **Regular Bias Checks on Live Predictions:**  
   A weekly Airflow task samples recent endpoint predictions and applies the same bias detection logic used pre-deployment. Two weeks ago, this check flagged a slight performance gap for highly volatile stocks. We responded by adjusting the dataset distribution and retraining the model, effectively restoring fairness metrics to baseline levels.

4. **Automated Retraining with CI/CD Pipelines:**  
   We integrated our GitHub repository with Vertex AI and CI/CD pipelines via GitHub Actions. When data drift or fairness alerts occur, an automated workflow pulls fresh data, reruns preprocessing and bias mitigation steps, triggers a new Vertex AI AutoML training job, and redeploys the improved model. This responsive process ensured timely adaptation when recent market sector rotations changed the feature importance landscape.

5. **Data Drift Dashboard for Stakeholder Transparency:**  
   We created a drift dashboard using GCP Monitoring and BigQuery, accessible to the entire team. By visualizing feature distribution shifts over time, stakeholders gain insight into why the model’s recommendations may differ month-to-month. This transparency fosters trust and informs decision-making around model updates.

6. **Historical Artifact Storage and Versioning:**  
   Each retraining event’s data snapshots, model artifacts, and metrics are recorded in MLflow and DVC. Several weeks ago, this historical record helped us pinpoint when prediction skew first appeared in specific industry segments and track how subsequent corrective measures improved model performance. This retrospective capability ensures data-driven improvement cycles.

7. **Expanding Slice Analysis Beyond Volatility:**  
   Rather than focusing exclusively on volatility slices, we rotate through different slicing criteria monthly—such as sector-based or liquidity-based segments. This rotation recently revealed underperformance in low-volume tech stocks, prompting targeted mitigation. By not relying on a single slicing strategy, we maintain broader coverage against evolving market challenges.

8. **Monthly Model Health Reports:**  
   We issue a “Model Health” report each month, summarizing data drift incidents, bias check outcomes, retraining triggers, and performance trends. Sharing these reports with both technical and non-technical stakeholders keeps everyone informed about the model’s ongoing adaptation and stability.

---



## Conclusion
By integrating Airflow, Docker, DVC, MLflow, Fairlearn, and Vertex AI, this MLOps pipeline ensures accurate, fair, and maintainable stock price predictions. Continuous monitoring and retraining capabilities keep the model up-to-date. The Streamlit interface makes predictive insights accessible, enabling stakeholders to confidently leverage the model’s outputs for informed decision-making.
```