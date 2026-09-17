import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.ingestion import DataIngestionPipeline
from src.preprocessing import DataPreprocessor, TARGET_COLUMN

logger = logging.getLogger("ModelTraining")

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-5))) * 100)
    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "MAPE": round(mape, 4)
    }

class ModelTrainer:
    def __init__(self, data_path: str = "data/sales_data.csv", models_dir: str = "models"):
        self.data_path = data_path
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self.preprocessor = DataPreprocessor()

    def run_training_pipeline(self):
        logger.info("Starting end-to-end model training pipeline...")
        ingestion = DataIngestionPipeline(self.data_path)
        df = ingestion.load_data()

        X = df.drop(columns=["Transaction_ID", "Date", TARGET_COLUMN])
        y = df[TARGET_COLUMN].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logger.info(f"Split data into train shape {X_train.shape} and test shape {X_test.shape}.")

        # Fit preprocessor on train set
        X_train_proc = self.preprocessor.fit_transform(X_train)
        X_test_proc = self.preprocessor.transform(X_test)
        self.preprocessor.save(os.path.join(self.models_dir, "preprocessing_pipeline.joblib"))

        # Define Models
        models = {
            "Baseline (Mean)": DummyRegressor(strategy="mean"),
            "Linear Regression (Ridge)": Ridge(alpha=1.0, random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            "Hist Gradient Boosting": HistGradientBoostingRegressor(max_iter=100, random_state=42)
        }

        try:
            from xgboost import XGBRegressor
            models["XGBoost"] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
        except ImportError:
            logger.warning("XGBoost module not installed. Proceeding with HistGradientBoosting.")

        results = {}
        fitted_models = {}

        logger.info("Training and evaluating all models...")
        for name, model in models.items():
            model.fit(X_train_proc, y_train)
            preds = model.predict(X_test_proc)
            metrics = evaluate_predictions(y_test, preds)
            results[name] = metrics
            fitted_models[name] = model
            logger.info(f"[{name}] RMSE: {metrics['RMSE']}, R2: {metrics['R2']}, MAE: {metrics['MAE']}")

        # Save Baseline model explicitly
        baseline_model = fitted_models["Baseline (Mean)"]
        joblib.dump(baseline_model, os.path.join(self.models_dir, "baseline_model.joblib"))

        # Identify best non-baseline model based on R2 / RMSE
        advanced_results = {k: v for k, v in results.items() if k != "Baseline (Mean)"}
        best_model_name = max(advanced_results, key=lambda k: advanced_results[k]["R2"])
        best_model = fitted_models[best_model_name]
        logger.info(f"Selected Best Model: '{best_model_name}' with R2={results[best_model_name]['R2']}")

        # Save Best Model
        joblib.dump(best_model, os.path.join(self.models_dir, "best_model.joblib"))

        # Save metadata JSON
        metadata = {
            "best_model_name": best_model_name,
            "feature_names": list(self.preprocessor.get_feature_names()),
            "evaluation_results": results,
            "test_sample_count": len(y_test)
        }
        with open(os.path.join(self.models_dir, "model_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Model training pipeline executed successfully.")
        return results, best_model_name

if __name__ == "__main__":
    trainer = ModelTrainer()
    results, best_name = trainer.run_training_pipeline()
    print("--- Model Benchmark Results ---")
    for model_name, metrics in results.items():
        print(f"{model_name:25s} -> RMSE: {metrics['RMSE']:8.2f} | R2: {metrics['R2']:6.4f}")
    print(f"\nBest Model: {best_name}")
