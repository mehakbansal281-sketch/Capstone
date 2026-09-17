import os
import logging
import pandas as pd
import numpy as np

# Configure logging
LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DataIngestion")

REQUIRED_COLUMNS = [
    "Transaction_ID",
    "Date",
    "Country",
    "Product_Category",
    "Unit_Price",
    "Quantity_Ordered",
    "Discount_Percent",
    "Marketing_Spend",
    "Customer_Rating",
    "Store_Size_SqFt",
    "Is_Holiday_Season",
    "Sales_Amount"
]

COUNTRIES = ["USA", "UK", "Germany", "France", "Canada", "Japan", "Australia"]
CATEGORIES = ["Electronics", "Apparel", "Home & Kitchen", "Health & Beauty", "Automotive", "Books"]

def generate_synthetic_data(num_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic e-commerce retail dataset with continuous Sales_Amount target.
    Includes intentional missing values to allow EDA missing value analysis.
    """
    np.random.seed(seed)
    logger.info(f"Generating synthetic business dataset with {num_samples} records...")
    
    dates = pd.date_range(start="2024-01-01", periods=num_samples, freq="h")
    country_choices = np.random.choice(COUNTRIES, size=num_samples, p=[0.35, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05])
    category_choices = np.random.choice(CATEGORIES, size=num_samples)
    
    unit_prices = np.round(np.random.uniform(15.0, 850.0, size=num_samples), 2)
    quantities = np.random.randint(1, 15, size=num_samples)
    discounts = np.round(np.random.uniform(0.0, 0.30, size=num_samples), 2)
    marketing_spends = np.round(np.random.exponential(scale=500.0, size=num_samples) + 50.0, 2)
    ratings = np.round(np.random.uniform(2.5, 5.0, size=num_samples), 1)
    store_sizes = np.random.choice([2500, 5000, 10000, 15000, 25000], size=num_samples)
    holidays = np.random.choice([0, 1], size=num_samples, p=[0.82, 0.18])
    
    # Base target formula with realistic correlation and noise
    raw_sales = (
        (unit_prices * quantities * (1.0 - discounts)) +
        (marketing_spends * 0.45) +
        (store_sizes * 0.02) +
        (ratings * 45.0) +
        (holidays * 250.0) +
        np.random.normal(loc=0.0, scale=80.0, size=num_samples)
    )
    sales_amount = np.round(np.maximum(raw_sales, 10.0), 2)
    
    df = pd.DataFrame({
        "Transaction_ID": [f"TXN-{100000 + i}" for i in range(num_samples)],
        "Date": dates,
        "Country": country_choices,
        "Product_Category": category_choices,
        "Unit_Price": unit_prices,
        "Quantity_Ordered": quantities,
        "Discount_Percent": discounts,
        "Marketing_Spend": marketing_spends,
        "Customer_Rating": ratings,
        "Store_Size_SqFt": store_sizes,
        "Is_Holiday_Season": holidays,
        "Sales_Amount": sales_amount
    })
    
    # Inject ~2% missing values into select numerical and categorical columns for EDA demo
    mask_spend = np.random.rand(num_samples) < 0.02
    df.loc[mask_spend, "Marketing_Spend"] = np.nan
    
    mask_rating = np.random.rand(num_samples) < 0.015
    df.loc[mask_rating, "Customer_Rating"] = np.nan

    mask_category = np.random.rand(num_samples) < 0.01
    df.loc[mask_category, "Product_Category"] = np.nan

    logger.info("Dataset generation completed.")
    return df

class DataIngestionPipeline:
    def __init__(self, data_path: str = "data/sales_data.csv"):
        self.data_path = data_path

    def load_data(self, auto_generate: bool = True) -> pd.DataFrame:
        """
        Loads data from CSV. Automatically generates dataset if missing.
        """
        if not os.path.exists(self.data_path):
            if auto_generate:
                logger.warning(f"Data file '{self.data_path}' not found. Generating default dataset...")
                os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
                df = generate_synthetic_data()
                df.to_csv(self.data_path, index=False)
                logger.info(f"Dataset saved to '{self.data_path}'.")
            else:
                raise FileNotFoundError(f"Data file '{self.data_path}' does not exist.")
        else:
            logger.info(f"Loading data from '{self.data_path}'...")
            df = pd.read_csv(self.data_path)
            
        self.validate_data(df)
        return df

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validates schema, required columns, and basic integrity.
        """
        logger.info("Validating dataset integrity...")
        if df.empty:
            raise ValueError("Ingested dataset is empty.")
        
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Dataset is missing required columns: {missing_cols}")

        null_counts = df[REQUIRED_COLUMNS].isnull().sum().to_dict()
        logger.info(f"Validation successful. Rows: {len(df)}, Columns: {len(df.columns)}")
        logger.info(f"Missing Value Counts: {null_counts}")
        return True

if __name__ == "__main__":
    pipeline = DataIngestionPipeline()
    df = pipeline.load_data()
    print(f"Data Ingestion Successful! Shape: {df.shape}")
