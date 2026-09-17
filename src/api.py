import os
import time
import logging
import joblib
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.ingestion import DataIngestionPipeline, COUNTRIES, CATEGORIES
from src.preprocessing import DataPreprocessor, TARGET_COLUMN
from src.monitoring import monitor

# Configure logger
LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("FastAPIApp")

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.joblib")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "models/preprocessing_pipeline.joblib")
DATA_PATH = os.getenv("DATA_PATH", "data/sales_data.csv")

model = None
preprocessor = None

def load_artifacts():
    global model, preprocessor
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
            model = joblib.load(MODEL_PATH)
            preprocessor = DataPreprocessor.load(PREPROCESSOR_PATH)
            logger.info("Successfully loaded ML model and preprocessor artifacts.")
        else:
            logger.warning("Artifacts missing. Running automatic ingestion and model training...")
            from src.training import ModelTrainer
            trainer = ModelTrainer(data_path=DATA_PATH, models_dir=os.path.dirname(MODEL_PATH))
            trainer.run_training_pipeline()
            model = joblib.load(MODEL_PATH)
            preprocessor = DataPreprocessor.load(PREPROCESSOR_PATH)
            logger.info("Automatic model training completed and loaded.")
    except Exception as e:
        logger.error(f"Error loading model artifacts: {str(e)}", exc_info=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield

app = FastAPI(
    title="ML Sales & Demand Forecasting API",
    description="Production-Ready REST API for Business Sales Metrics & Demand Forecasting",
    version="1.0.0",
    lifespan=lifespan
)

# Async Middleware for performance monitoring & logging
@app.middleware("http")
async def monitor_and_log_middleware(request: Request, call_next):
    start_time = time.time()
    response = None
    is_error = False
    try:
        response = await call_next(request)
        if response.status_code >= 400:
            is_error = True
        return response
    except Exception as exc:
        is_error = True
        raise exc
    finally:
        duration_ms = (time.time() - start_time) * 1000.0
        monitor.record_request(duration_ms, is_error=is_error)
        status_code = response.status_code if response else 500
        logger.info(f"Method: {request.method} Path: {request.url.path} Status: {status_code} Duration: {duration_ms:.2f}ms")

# Pydantic Input Validation Schema
class TransactionPayload(BaseModel):
    Country: str = Field(..., json_schema_extra={"example": "USA"}, description="Target market country")
    Product_Category: str = Field(..., json_schema_extra={"example": "Electronics"}, description="Category of product")
    Unit_Price: float = Field(..., gt=0, json_schema_extra={"example": 299.99}, description="Unit price in USD")
    Quantity_Ordered: int = Field(..., gt=0, json_schema_extra={"example": 5}, description="Number of items ordered")
    Discount_Percent: float = Field(0.0, ge=0.0, le=1.0, json_schema_extra={"example": 0.10}, description="Discount fraction (0.0 to 1.0)")
    Marketing_Spend: float = Field(..., ge=0.0, json_schema_extra={"example": 500.0}, description="Marketing spend in USD")
    Customer_Rating: float = Field(..., ge=1.0, le=5.0, json_schema_extra={"example": 4.5}, description="Rating scale 1 to 5")
    Store_Size_SqFt: float = Field(..., gt=0, json_schema_extra={"example": 10000.0}, description="Store area in sq ft")
    Is_Holiday_Season: int = Field(0, ge=0, le=1, json_schema_extra={"example": 1}, description="Binary flag 0 or 1")

    @field_validator("Country")
    def validate_country(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Country cannot be empty.")
        return v.strip()

# Exception Handlers
@app.exception_handler(HTTPException)
def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP Exception on {request.url.path}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": True, "message": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Server Error on {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": True, "message": "An internal server error occurred.", "details": str(exc)}
    )

# Endpoints
@app.get("/health", summary="API Health Check")
def health_check():
    """
    Returns API health status and model availability.
    """
    model_loaded = model is not None and preprocessor is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "version": "1.0.0"
    }

@app.get("/predict", summary="Predict Sales for a Specific Country")
def predict_by_country(country: str = Query(..., description="Country code (e.g. USA, UK, Germany)")):
    """
    Generates predicted sales metrics for transactions in a specific country.
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded.")

    try:
        ingestion = DataIngestionPipeline(DATA_PATH)
        df = ingestion.load_data()

        # Filter by country (case-insensitive search)
        country_df = df[df["Country"].str.upper() == country.strip().upper()].copy()

        if country_df.empty:
            valid_countries = sorted(df["Country"].unique().tolist())
            raise HTTPException(
                status_code=404,
                detail=f"Country '{country}' not found. Available countries: {valid_countries}"
            )

        X = country_df.drop(columns=["Transaction_ID", "Date", TARGET_COLUMN], errors="ignore")
        
        start_inf = time.time()
        X_proc = preprocessor.transform(X)
        predictions = model.predict(X_proc)
        inf_duration = (time.time() - start_inf) * 1000.0
        monitor.record_inference(inf_duration)

        total_predicted_sales = float(np.sum(predictions))
        avg_predicted_sales = float(np.mean(predictions))
        count = len(predictions)

        logger.info(f"Prediction generated for country '{country}'. Total: ${total_predicted_sales:.2f}")

        return {
            "country": country.upper(),
            "total_transactions": count,
            "total_predicted_sales": round(total_predicted_sales, 2),
            "average_predicted_sales": round(avg_predicted_sales, 2),
            "sample_predictions": [round(float(p), 2) for p in predictions[:5]]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing GET /predict for country '{country}': {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/predict/all", summary="Predict Sales for All Countries Combined")
def predict_all_countries():
    """
    Generates global sales predictions aggregated across all countries with per-country breakdown.
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded.")

    try:
        ingestion = DataIngestionPipeline(DATA_PATH)
        df = ingestion.load_data()

        X = df.drop(columns=["Transaction_ID", "Date", TARGET_COLUMN], errors="ignore")

        start_inf = time.time()
        X_proc = preprocessor.transform(X)
        predictions = model.predict(X_proc)
        monitor.record_inference((time.time() - start_inf) * 1000.0)

        df["Predicted_Sales"] = predictions
        
        breakdown = {}
        for cntry, group in df.groupby("Country"):
            cntry_preds = group["Predicted_Sales"].values
            breakdown[cntry] = {
                "transaction_count": len(cntry_preds),
                "total_predicted_sales": round(float(np.sum(cntry_preds)), 2),
                "avg_predicted_sales": round(float(np.mean(cntry_preds)), 2)
            }

        total_global_sales = float(np.sum(predictions))
        avg_global_sales = float(np.mean(predictions))

        logger.info(f"Global predictions generated across {len(df)} transactions.")

        return {
            "total_transactions": len(predictions),
            "global_total_predicted_sales": round(total_global_sales, 2),
            "global_avg_predicted_sales": round(avg_global_sales, 2),
            "country_breakdown": breakdown
        }
    except Exception as e:
        logger.error(f"Error executing GET /predict/all: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict", summary="Predict Sales for Custom Single Payload")
def predict_custom_payload(payload: TransactionPayload):
    """
    Accepts custom transaction json input and returns predicted sales amount.
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded.")

    try:
        data_dict = payload.model_dump()
        input_df = pd.DataFrame([data_dict])

        start_inf = time.time()
        X_proc = preprocessor.transform(input_df)
        pred = model.predict(X_proc)[0]
        monitor.record_inference((time.time() - start_inf) * 1000.0)

        predicted_val = round(float(pred), 2)
        logger.info(f"Custom payload prediction generated: ${predicted_val}")

        return {
            "status": "success",
            "predicted_sales_amount": predicted_val,
            "input_summary": data_dict
        }
    except Exception as e:
        logger.error(f"Error executing POST /predict: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Custom prediction error: {str(e)}")

@app.get("/metrics", summary="Performance & Operational Metrics")
def get_metrics():
    """
    Exposes API response times, model inference latencies, memory and CPU usage metrics.
    """
    return monitor.get_metrics_summary()
