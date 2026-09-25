"""Time-aware feature engineering for the daily demand panel.

Every lag/rolling feature is shifted so that day t only sees information
available before day t, which is what makes the evaluation an honest
simulation of forecasting future demand rather than curve-fitting.
"""
import pandas as pd

from . import config

GROUP_COLS = ["category", "sub_category"]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = df["date"].dt
    df["day_of_week"] = dt.dayofweek
    df["day_of_month"] = dt.day
    df["day_of_year"] = dt.dayofyear
    df["week_of_year"] = dt.isocalendar().week.astype("int32")
    df["month"] = dt.month
    df["quarter"] = dt.quarter
    df["year"] = dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["is_month_start"] = dt.is_month_start.astype("int8")
    df["is_month_end"] = dt.is_month_end.astype("int8")
    return df


def add_lag_features(df: pd.DataFrame, target_col: str = config.TARGET_COL) -> pd.DataFrame:
    df = df.copy()
    grouped = df.groupby(GROUP_COLS, observed=True)[target_col]
    for lag in config.LAG_DAYS:
        df[f"lag_{lag}"] = grouped.shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target_col: str = config.TARGET_COL) -> pd.DataFrame:
    df = df.copy()
    # Shift by 1 first so the rolling window never includes the current day's own demand.
    shifted = df.groupby(GROUP_COLS, observed=True)[target_col].shift(1)
    for window in config.ROLLING_WINDOWS:
        rolled = shifted.groupby([df["category"], df["sub_category"]]).rolling(window).mean()
        df[f"rolling_mean_{window}"] = rolled.reset_index(level=GROUP_COLS, drop=True)
        rolled_std = shifted.groupby([df["category"], df["sub_category"]]).rolling(window).std()
        df[f"rolling_std_{window}"] = rolled_std.reset_index(level=GROUP_COLS, drop=True)
    return df


def add_price_discount_lag(df: pd.DataFrame) -> pd.DataFrame:
    """Same-day price/discount are known in advance (set by the retailer),
    so they are safe to use directly rather than lagged."""
    return df


def build_feature_table(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["category", "sub_category", "date"]).reset_index(drop=True)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_price_discount_lag(df)

    # Category dtype gives XGBoost native categorical support and a stable
    # integer code for the PyTorch embedding lookup.
    for col in GROUP_COLS:
        df[col] = df[col].astype("category")

    # Rows in each group's burn-in period have NaN lag/rolling features by
    # construction (not enough history yet) -- drop rather than impute them.
    feature_cols_with_nan = [f"lag_{lag}" for lag in config.LAG_DAYS] + [
        f"rolling_mean_{w}" for w in config.ROLLING_WINDOWS
    ]
    df = df.dropna(subset=feature_cols_with_nan).reset_index(drop=True)
    return df


def main() -> None:
    panel = pd.read_parquet(config.DAILY_DEMAND_PARQUET)
    features = build_feature_table(panel)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(config.FEATURES_PARQUET, index=False)
    print(f"Saved feature table: {features.shape} -> {config.FEATURES_PARQUET}")


if __name__ == "__main__":
    main()
