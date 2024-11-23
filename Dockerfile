# Use the official Apache Airflow image as the base
FROM apache/airflow:2.7.1

# Switch to root for installing system dependencies
USER root

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch to the airflow user for installing Python dependencies
USER airflow

# Copy the requirements file into the container
COPY requirements.txt /opt/airflow/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt

# Create necessary directories for the project
RUN mkdir -p /opt/airflow/project/data/{raw,processed,validated,mitigated,analyzed,anomalies,bias} \
    && mkdir -p /opt/airflow/project/src/model \
    && mkdir -p /opt/airflow/dags \
    && mkdir -p /opt/airflow/plugins \
    && mkdir -p /opt/airflow/logs

# Copy your source code, DAGs, and plugins into the container
COPY src /opt/airflow/project/src
COPY dags /opt/airflow/dags
COPY plugins /opt/airflow/plugins

# Set the working directory
WORKDIR /opt/airflow

# Set environment variables
ENV PYTHONPATH=/opt/airflow/project
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# Expose the default Airflow port (optional)
EXPOSE 8080
