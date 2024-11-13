FROM apache/airflow:2.7.1

# Switch to root for system dependencies
USER root

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch to airflow user for Python packages
USER airflow

# Install Python packages
COPY requirements.txt /opt/airflow/requirements.txt
RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt

# Create necessary directories
RUN mkdir -p /opt/airflow/stock_prediction/data/{raw,processed,validated,mitigated,analyzed,anomalies}

# Set working directory
WORKDIR /opt/airflow

# Set environment variables
ENV PYTHONPATH=/opt/airflow
