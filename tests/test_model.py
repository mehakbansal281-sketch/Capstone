import os
import joblib
import pytest
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from src.ingestion import generate_synthetic_data, DataIngestionPipeline
from src.preprocessing import DataPreprocessor, TARGET_COLUMN
from src.training import evaluate_predictions, ModelTrainer

@pytest.fixture
def synthetic_df():
    return generate_synthetic_data(num_samples=100, seed=123)

@pytest.fixture
def trained_model_and_preprocessor(tmp_path, synthetic_df):
    """
    Creates an isolated model and preprocessor artifact inside tmp_path for unit testing.
    Ensures production artifacts are not touched.
    """
    data_file = tmp_path / "test_sales.csv"
    synthetic_df.to_csv(data_file, index=False)

    models_dir = tmp_path / "models"
    models_dir.mkdir()

    trainer = ModelTrainer(data_path=str(data_file), models_dir=str(models_dir))
    results, best_name = trainer.run_training_pipeline()

    model_path = models_dir / "best_model.joblib"
    prep_path = models_dir / "preprocessing_pipeline.joblib"

    return str(model_path), str(prep_path), results

def test_data_ingestion_and_validation(synthetic_df, tmp_path):
    csv_path = tmp_path / "sales.csv"
    synthetic_df.to_csv(csv_path, index=False)

    pipeline = DataIngestionPipeline(str(csv_path))
    loaded_df = pipeline.load_data()

    assert not loaded_df.empty
    assert len(loaded_df) == 100
    assert "Sales_Amount" in loaded_df.columns
    assert pipeline.validate_data(loaded_df) is True

def test_data_ingestion_invalid_schema(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"Wrong_Col": [1, 2, 3]}).to_csv(bad_csv, index=False)

    pipeline = DataIngestionPipeline(str(bad_csv))
    with pytest.raises(ValueError, match="missing required columns"):
        pipeline.load_data(auto_generate=False)

def test_preprocessor_fit_transform(synthetic_df):
    X = synthetic_df.drop(columns=["Transaction_ID", "Date", TARGET_COLUMN])
    preprocessor = DataPreprocessor()
    X_proc = preprocessor.fit_transform(X)

    assert isinstance(X_proc, np.ndarray)
    assert X_proc.shape[0] == len(synthetic_df)
    assert not np.isnan(X_proc).any()

def test_model_loading_and_prediction(trained_model_and_preprocessor, synthetic_df):
    model_path, prep_path, _ = trained_model_and_preprocessor

    assert os.path.exists(model_path)
    assert os.path.exists(prep_path)

    loaded_model = joblib.load(model_path)
    loaded_prep = DataPreprocessor.load(prep_path)

    X = synthetic_df.drop(columns=["Transaction_ID", "Date", TARGET_COLUMN])
    X_proc = loaded_prep.transform(X)
    preds = loaded_model.predict(X_proc)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(synthetic_df)
    assert not np.isnan(preds).any()
    assert (preds > 0).all()

def test_evaluation_metrics_computation():
    y_true = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    y_pred = np.array([110.0, 190.0, 310.0, 390.0, 510.0])

    metrics = evaluate_predictions(y_true, y_pred)
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics
    assert metrics["R2"] > 0.95
    assert metrics["MAE"] == 10.0
