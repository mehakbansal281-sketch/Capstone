import os
import pytest
import pandas as pd
from fastapi.testclient import TestClient

from src.ingestion import generate_synthetic_data
from src.training import ModelTrainer
from src.api import app, load_artifacts

@pytest.fixture(scope="module", autouse=True)
def setup_test_environment(tmp_path_factory):
    """
    Sets up isolated test environment variables and builds temporary model artifacts.
    """
    tmp_dir = tmp_path_factory.mktemp("api_test_env")
    test_data = tmp_dir / "test_sales.csv"
    test_models = tmp_dir / "models"
    test_logs = tmp_dir / "logs"

    df = generate_synthetic_data(num_samples=200, seed=999)
    df.to_csv(test_data, index=False)

    trainer = ModelTrainer(data_path=str(test_data), models_dir=str(test_models))
    trainer.run_training_pipeline()

    os.environ["DATA_PATH"] = str(test_data)
    os.environ["MODEL_PATH"] = str(test_models / "best_model.joblib")
    os.environ["PREPROCESSOR_PATH"] = str(test_models / "preprocessing_pipeline.joblib")
    os.environ["LOG_DIR"] = str(test_logs)

    # Force API to reload artifacts from isolated test paths
    load_artifacts()

@pytest.fixture
def client():
    return TestClient(app)

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    json_resp = response.json()
    assert json_resp["status"] == "healthy"
    assert json_resp["model_loaded"] is True
    assert json_resp["version"] == "1.0.0"

def test_predict_country_valid(client):
    response = client.get("/predict?country=USA")
    assert response.status_code == 200
    data = response.json()
    assert data["country"] == "USA"
    assert "total_transactions" in data
    assert "total_predicted_sales" in data
    assert "average_predicted_sales" in data
    assert len(data["sample_predictions"]) > 0

def test_predict_country_invalid(client):
    response = client.get("/predict?country=NonExistentCountry123")
    assert response.status_code == 404
    data = response.json()
    assert data["error"] is True
    assert "not found" in data["message"]

def test_predict_all_countries(client):
    response = client.get("/predict/all")
    assert response.status_code == 200
    data = response.json()
    assert "total_transactions" in data
    assert "global_total_predicted_sales" in data
    assert "country_breakdown" in data
    assert isinstance(data["country_breakdown"], dict)
    assert len(data["country_breakdown"]) > 0

def test_predict_post_valid_payload(client):
    payload = {
        "Country": "USA",
        "Product_Category": "Electronics",
        "Unit_Price": 199.99,
        "Quantity_Ordered": 3,
        "Discount_Percent": 0.05,
        "Marketing_Spend": 400.0,
        "Customer_Rating": 4.5,
        "Store_Size_SqFt": 12000.0,
        "Is_Holiday_Season": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "predicted_sales_amount" in data
    assert data["predicted_sales_amount"] > 0

def test_predict_post_invalid_payload(client):
    bad_payload = {
        "Country": "USA",
        "Unit_Price": -50.0  # Invalid negative price
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422

def test_metrics_endpoint(client):
    client.get("/health")
    client.get("/predict?country=USA")
    
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "uptime_seconds" in data
    assert "total_requests" in data
    assert "system_metrics" in data
    assert "memory_usage_mb" in data["system_metrics"]
