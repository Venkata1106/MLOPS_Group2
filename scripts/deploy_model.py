import mlflow
import yaml
import logging
from google.cloud import storage
from google.cloud import aiplatform
from pathlib import Path
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelDeployer:
    def __init__(self, config_path: str = 'config/pipeline_config.yml'):
        self.config = self._load_config(config_path)
        self.registry_config = self.config['model_registry']
        self.gcp_project = self.registry_config['gcp_project']
        self.region = self.registry_config['region']
        
    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_latest_run(self) -> Optional[mlflow.entities.Run]:
        """Get the latest successful MLflow run"""
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            experiment_ids=[self.config['experiment_id']],
            filter_string="metrics.validation_passed = 1 AND metrics.bias_check_passed = 1",
            order_by=["attributes.start_time DESC"],
            max_results=1
        )
        return runs[0] if runs else None
    
    def push_to_registry(self) -> bool:
        """Push model to GCP Model Registry"""
        try:
            # Get latest successful run
            latest_run = self._get_latest_run()
            if not latest_run:
                logger.error("No valid model runs found")
                return False
            
            # Initialize Vertex AI
            aiplatform.init(
                project=self.gcp_project,
                location=self.region
            )
            
            # Get model artifacts path
            model_uri = f"runs:/{latest_run.info.run_id}/model"
            
            # Upload model to GCS first
            bucket_name = self.registry_config['gcs_bucket']
            model_dir = f"models/{latest_run.info.run_id}"
            
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            
            # Upload model files
            mlflow.sklearn.save_model(
                mlflow.sklearn.load_model(model_uri),
                model_dir
            )
            
            for root, _, files in Path(model_dir).walk():
                for file in files:
                    local_file = Path(root) / file
                    blob_name = f"{model_dir}/{file}"
                    blob = bucket.blob(blob_name)
                    blob.upload_from_filename(str(local_file))
            
            # Create model in Vertex AI Model Registry
            model = aiplatform.Model.upload(
                display_name=self.registry_config['model_name'],
                artifact_uri=f"gs://{bucket_name}/{model_dir}",
                serving_container_image_uri=self.registry_config['serving_container'],
                parent_model=self.registry_config.get('parent_model'),
                version_description=f"Model version from run {latest_run.info.run_id}"
            )
            
            # Log registry information back to MLflow
            with mlflow.start_run(run_id=latest_run.info.run_id, nested=True):
                mlflow.log_dict(
                    {
                        'model_registry': {
                            'model_id': model.resource_name,
                            'version': model.version_id,
                            'gcs_path': f"gs://{bucket_name}/{model_dir}"
                        }
                    },
                    "model_registry_info.json"
                )
            
            logger.info(f"Successfully pushed model to registry: {model.resource_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error pushing model to registry: {str(e)}")
            return False

def main():
    deployer = ModelDeployer()
    success = deployer.push_to_registry()
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 