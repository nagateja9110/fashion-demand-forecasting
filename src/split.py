"""Time-aware train/val/test split.

Retail demand is autocorrelated in time, so a random split would leak
adjacent-day information between train and test and produce an overly
optimistic estimate of forecast accuracy. Instead the most recent block of
days is held out entirely, mirroring how the model would actually be used:
trained on the past, evaluated on days it has never seen.
"""
import pandas as pd

from . import config


def time_aware_split(df: pd.DataFrame, date_col: str = "date"):
    max_date = df[date_col].max()
    test_start = max_date - pd.Timedelta(days=config.TEST_DAYS - 1)
    val_start = test_start - pd.Timedelta(days=config.VAL_DAYS)

    train = df[df[date_col] < val_start].reset_index(drop=True)
    val = df[(df[date_col] >= val_start) & (df[date_col] < test_start)].reset_index(drop=True)
    test = df[df[date_col] >= test_start].reset_index(drop=True)
    return train, val, test
