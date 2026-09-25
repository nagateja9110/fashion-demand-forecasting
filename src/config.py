from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

TRANSACTIONS_CSV = RAW_DIR / "transactions.csv"
PRODUCTS_CSV = RAW_DIR / "products.csv"
DISCOUNTS_CSV = RAW_DIR / "discounts.csv"
STORES_CSV = RAW_DIR / "stores.csv"

DAILY_DEMAND_PARQUET = PROCESSED_DIR / "daily_demand.parquet"
FEATURES_PARQUET = PROCESSED_DIR / "features.parquet"

TARGET_COL = "units_sold"

# Time-aware split: last N days held out for testing, N before that for validation.
TEST_DAYS = 60
VAL_DAYS = 60

# Feature engineering windows (in days).
LAG_DAYS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]

RANDOM_SEED = 42
