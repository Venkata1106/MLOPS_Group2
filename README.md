

---

```markdown
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
This project demonstrates a complete MLOps pipeline for predicting stock prices. It starts from raw financial market data and takes you through a series of automated steps: data ingestion, cleaning, feature engineering, bias detection and mitigation, anomaly analysis, model training with hyperparameter tuning, experiment tracking using MLflow, and finally deploying the best-performing and fairest model to a Vertex AI endpoint on Google Cloud. A Streamlit-based front-end gives users a friendly interface to interact with the model’s predictions.

In essence, we’re building a pipeline that ensures every stage—**from data to deployed model**—is reliable, reproducible, transparent, and continuously monitored for performance and fairness.

---

## Dataset Information

**Data Source:** Yahoo Finance (via `yfinance`).

**Data Size:** About 252 rows × 7 columns of daily stock data per year per ticker.

**Features:**
- **Core:** Date, Open, High, Low, Close, Adj Close, Volume
- **Derived:** Returns, Moving Averages (MA5, MA20), Volatility, and Anomaly Flags.

These additional features help the model understand trends and patterns in stock price movements over time.

---

## Prerequisites
- **Python 3.8+**
- **Docker & Docker Compose** for reproducible environments
- **Apache Airflow 2.x** for pipeline orchestration
- **DVC** (Data Version Control) to track data changes
- **MLflow** for logging experiments, parameters, and metrics
- **Vertex AI (GCP)** for automated model training (AutoML) and scalable deployment
- Optionally, **Terraform** for infrastructure provisioning
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
   Access Airflow at `http://localhost:8080` (Credentials: admin/admin).

4. **Trigger Airflow DAG:**
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
   Open `http://localhost:5000` to compare experiment runs.

---

## Data Pipeline Steps (What’s Happening at Each Stage)

1. **Data Acquisition:**  
   The pipeline fetches raw historical stock data using `yfinance`. Think of it as downloading a CSV of daily prices directly from Yahoo Finance’s API.

2. **Preprocessing & Validation:**  
   We clean the data, handle missing values, and create new features like daily returns or moving averages. Validation steps ensure no corrupted rows or missing critical fields are passed along.

3. **Bias Detection & Mitigation:**  
   We check if the model might later treat certain segments unfairly (e.g., certain volatility conditions) by looking at data distributions. If bias is detected, we apply strategies (like reweighting samples) so that every subset of the data is fairly represented.

4. **Anomaly & Slice Analysis:**  
   The data is examined for outliers or unusual market conditions. This ensures the model can handle edge cases (like sudden market crashes) and perform consistently across different market states (e.g., bull vs. bear).

5. **Notifications & Logging:**  
   At the end of each pipeline run, notifications (e.g., emails or Slack messages) let you know if the run succeeded or failed. Logging helps debug issues if something goes wrong.

6. **Data Versioning (DVC):**  
   If you need to roll back to a previous dataset version, DVC makes it easy. This ensures reproducibility: if a model performed well last month, you can reconstruct the exact data used.

---

## Explaining the `src` Directory (Data Pipeline)

- **data_acquisition.py:** Downloads the raw stock data.
- **data_preprocessing.py:** Cleans the data, computes returns, moving averages, and volatility.
- **data_validation.py:** Checks that the dataset meets our quality standards (no unexpected missing columns, proper data types, continuous dates).
- **bias_detection.py & bias_mitigation.py:** Identifies if any data segments might cause unfair model performance and fixes it by adjusting sample distributions.
- **slice_analysis.py:** Analyzes the dataset across different dimensions (like volatility ranges) to ensure no subset is overlooked.
- **utils/**: Contains logging and monitoring utilities that keep track of what’s happening at each pipeline stage.

---

## MODEL DEVELOPMENT (More Detail)

After the pipeline has prepared a high-quality, bias-mitigated dataset, we move into ML development. Here’s what happens:

1. **Data Loading & Splitting:**  
   The `data_loader.py` uses the processed dataset and splits it into training (to learn patterns), validation (to tune parameters), and test sets (to unbiasedly judge final performance).

2. **Model Training with Multiple Algorithms:**  
   We train different models (e.g., Random Forest, XGBoost). By trying multiple algorithms, we increase the chances of finding one that adapts best to stock market patterns. Each model sees the training data and learns how price and volume patterns translate into future prices.

3. **Hyperparameter Tuning:**  
   The `hyperparameter_tuner.py` systematically searches for the best combination of settings (like tree depth, learning rate) that improves accuracy and stability. Instead of guessing parameters, this automated search makes the model more robust.

4. **Bias Evaluation & Mitigation in ML Stage:**  
   Even if we balanced the data, the model might learn unfair patterns. The `bias_checker.py` re-examines predictions on different slices. If bias remains, we iterate: adjust data or training procedures and retrain until the model treats all segments more fairly.

5. **Validation & Model Selection:**  
   The `model_validator.py` checks how well the model does on unseen test data. The `model_selector.py` picks the best among candidates based on accuracy, fairness, and generalization ability. This step ensures we don’t deploy a model that just got lucky on training data.

6. **Experiment Tracking with MLflow:**  
   The `experiment_tracker.py` logs runs, parameters, metrics, and generated plots in MLflow. We can revisit any experiment, see what hyperparameters were used, and compare performance over time. It’s like keeping a detailed lab notebook of every model trial.

7. **Explainability (SHAP/LIME):**  
   The `sensitivity_analyzer.py` uses these tools to show which features strongly influence predictions. This builds trust: you’ll know if the model relies heavily on certain recent volatility signals or a particular moving average.

8. **Model Registry & Finalization:**  
   Once satisfied, we push the chosen model to `model_registry.py`, storing a final version ready for deployment. No confusion about which model is going live—only the best one is registered.

---

## Folder Structure Overview (Key Files Only)

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

## To Run Model Files
First, ensure extra model dependencies are installed:
```bash
pip install -r modelrequire.txt
```

Run model training:
```bash
python models/train.py
```

### Visualizations & Results
- **metrics_heatmap.png & performance_comparison.png:** Compare different model variants visually.
- **Bias Reports:** Check whether certain groups get worse predictions.
- **SHAP/LIME Plots:** Highlight how features affect individual predictions.
- **MLflow UI:** Run `mlflow ui` to see all experiment logs, metrics, and artifacts in a clean dashboard.

---

## MODEL DEPLOYMENT

We use Vertex AI (AutoML) to train and build a strong model automatically. Once AutoML selects a top model, we deploy it to a Vertex AI endpoint. This endpoint provides real-time predictions at scale.

### Deployment Steps
1. Set up GCP: `gcloud auth login`, enable Vertex AI.
2. In the `vertex_ai` directory:
   ```bash
   pip install -r requirement.txt
   python predict.py
   streamlit run app.py
   ```
   The Streamlit interface lets you input stock parameters and see the model’s predictions.

### Monitoring & Retraining
- **Vertex AI Monitoring:** Watches performance over time. If new market conditions emerge (data drift) or accuracy drops (model decay), alerts can trigger automatic retraining via CI/CD pipelines.
- **Continuous Improvement:** If the model’s performance declines, the pipeline can spin up AutoML training again with the latest data, ensuring your predictions remain current and reliable.

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
| DVC              | Data versioning and management                |
| MLflow           | Experiment logging & comparison               |
| Vertex AI        | AutoML training & scalable deployment         |
| Terraform (Opt.) | Infrastructure as Code                       |
| Pandas, yfinance | Data acquisition & manipulation               |
| scikit-learn, XGBoost | Core ML modeling frameworks             |
| SHAP, LIME       | Explainability for model decisions            |
| Fairlearn        | Bias detection & mitigation                   |
| GCP Monitoring   | Observing performance, detecting data drift   |
| Streamlit        | User-friendly interface for predictions       |

---

## Cost & Resource Estimation
- **Local Setup:** Mainly developer time and hardware resources.
- **Cloud (GCP Vertex AI):**
  - Compute costs for training and serving.
  - Storage for datasets and models.
  - Artifact Registry for container images.
  
A minimal setup might cost ~$30-$60/month, growing with data volume, complexity, and usage frequency.

---

## Conclusion
This pipeline demonstrates a modern MLOps approach: automated data preparation, fairness checks, model experimentation, explainability, and seamless deployment with Vertex AI. Continuous monitoring and retraining keep the model relevant in changing markets. By integrating tools like Airflow, Docker, DVC, MLflow, Fairlearn, and Vertex AI, we ensure trustworthy, maintainable, and impactful stock price predictions—accessible through a Streamlit UI for effortless user interaction.
```