import os
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

logger = logging.getLogger("DataPreprocessing")

NUMERICAL_FEATURES = [
    "Unit_Price",
    "Quantity_Ordered",
    "Discount_Percent",
    "Marketing_Spend",
    "Customer_Rating",
    "Store_Size_SqFt",
    "Is_Holiday_Season"
]

CATEGORICAL_FEATURES = [
    "Country",
    "Product_Category"
]

TARGET_COLUMN = "Sales_Amount"

class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.num_features = NUMERICAL_FEATURES
        self.cat_features = CATEGORICAL_FEATURES
        self.pipeline = None
        self.feature_names_out = None

    def _build_pipeline(self):
        num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ("num", num_pipeline, self.num_features),
            ("cat", cat_pipeline, self.cat_features)
        ])
        return preprocessor

    def fit(self, X: pd.DataFrame, y=None):
        logger.info("Fitting DataPreprocessor pipeline...")
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X[self.num_features + self.cat_features])
        
        # Capture transformed feature names
        num_cols = self.num_features
        cat_encoder = self.pipeline.named_transformers_["cat"].named_steps["onehot"]
        cat_cols = list(cat_encoder.get_feature_names_out(self.cat_features))
        self.feature_names_out = num_cols + cat_cols
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("DataPreprocessor must be fitted before calling transform().")
        
        # Ensure input DataFrame has all expected columns or missing as NaN
        df_proc = X.copy()
        for col in self.num_features + self.cat_features:
            if col not in df_proc.columns:
                df_proc[col] = np.nan
                
        transformed_array = self.pipeline.transform(df_proc[self.num_features + self.cat_features])
        return transformed_array

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names(self):
        return self.feature_names_out

    def save(self, filepath: str = "models/preprocessing_pipeline.joblib"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Saved preprocessor pipeline to '{filepath}'.")

    @classmethod
    def load(cls, filepath: str = "models/preprocessing_pipeline.joblib"):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor file '{filepath}' not found.")
        preprocessor = joblib.load(filepath)
        logger.info(f"Loaded preprocessor pipeline from '{filepath}'.")
        return preprocessor
