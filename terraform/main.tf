provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_artifact_registry_repository" "model_registry" {
  location      = var.region
  repository_id = "vertex-ai-models"
  format        = "docker"
}

resource "google_vertex_ai_endpoint" "prediction_endpoint" {
  name         = "stock-prediction-endpoint"
  location     = var.region
  display_name = "Stock Price Prediction Endpoint"
} 