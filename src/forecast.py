"""Recursive multi-day-ahead demand forecasting for the live API.

Training evaluates one-step-ahead accuracy on real held-out days. Serving a
"forecast the next N days" feature needs another step: recursively feed each
day's prediction back in as pseudo-history for the next day's lag/rolling
features, exactly as a production forecasting service would when the actual
future outcome isn't known yet.
"""
import json
from datetime import timedelta

import pandas as pd
import xgboost as xgb

from . import config
from .features import add_calendar_features

HISTORY_DAYS = max(config.LAG_DAYS + config.ROLLING_WINDOWS)


def group_key(category: str, sub_category: str) -> str:
    return f"{category}||{sub_category}"


def build_promo_calendar(discounts_path=config.DISCOUNTS_CSV) -> dict:
    """Map (category, sub_category) -> {"MM-DD": discount_pct} from the
    recurring annual promo calendar, so a forecast for any future date can
    look up whether it falls in a planned campaign."""
    discounts = pd.read_csv(discounts_path, parse_dates=["Start", "End"])
    discounts.columns = ["start", "end", "discount_pct", "description", "category", "sub_category"]

    calendar: dict = {}
    for _, row in discounts.iterrows():
        key = group_key(row["category"], row["sub_category"])
        calendar.setdefault(key, {})
        for date in pd.date_range(row["start"], row["end"], freq="D"):
            month_day = date.strftime("%m-%d")
            calendar[key][month_day] = max(calendar[key].get(month_day, 0.0), float(row["discount_pct"]))
    return calendar


def build_recent_history(panel: pd.DataFrame) -> dict:
    history = {}
    for (category, sub_category), g in panel.groupby(["category", "sub_category"], observed=True):
        g = g.sort_values("date").tail(HISTORY_DAYS)
        records = g[
            ["date", "units_sold", "avg_unit_price", "avg_realized_discount", "planned_discount_pct", "is_promo_day"]
        ].copy()
        records["date"] = records["date"].dt.strftime("%Y-%m-%d")
        history[group_key(category, sub_category)] = records.to_dict("records")
    return history


def export_forecast_context(panel: pd.DataFrame, category_categories: list, subcategory_categories: list) -> dict:
    context = {
        "category_categories": category_categories,
        "subcategory_categories": subcategory_categories,
        "groups": [
            {"category": c, "sub_category": s}
            for c, s in panel[["category", "sub_category"]].drop_duplicates().itertuples(index=False)
        ],
        "last_date": panel["date"].max().strftime("%Y-%m-%d"),
        "history": build_recent_history(panel),
        "promo_calendar": build_promo_calendar(),
    }
    return context


def load_forecast_context(path=config.FORECAST_CONTEXT_JSON) -> dict:
    with open(path) as f:
        return json.load(f)


def _row_features(buffer: list, day_idx: int, calendar_row: dict) -> dict:
    """Compute lag/rolling features for buffer[day_idx] from the preceding
    entries in buffer (the same definition as src/features.py, just computed
    incrementally instead of vectorized over a full dataframe)."""
    history_before = [r["units_sold"] for r in buffer[:day_idx]]
    features = dict(calendar_row)

    for lag in config.LAG_DAYS:
        features[f"lag_{lag}"] = history_before[-lag] if len(history_before) >= lag else history_before[0]

    for window in config.ROLLING_WINDOWS:
        recent = history_before[-window:] if len(history_before) >= window else history_before
        series = pd.Series(recent, dtype=float)
        features[f"rolling_mean_{window}"] = series.mean()
        features[f"rolling_std_{window}"] = series.std() if len(series) > 1 else 0.0

    return features


def recursive_forecast(
    context: dict,
    booster: xgb.Booster,
    category: str,
    sub_category: str,
    horizon_days: int,
    avg_unit_price: float | None = None,
    discount_override_pct: float | None = None,
) -> list[dict]:
    key = group_key(category, sub_category)
    if key not in context["history"]:
        raise ValueError(f"Unknown category/sub_category combination: {category} / {sub_category}")

    history = [dict(r) for r in context["history"][key]]
    last_price = history[-1]["avg_unit_price"] if avg_unit_price is None else avg_unit_price
    promo_calendar = context["promo_calendar"].get(key, {})
    last_date = pd.Timestamp(history[-1]["date"])

    buffer = list(history)
    predictions = []

    for step in range(1, horizon_days + 1):
        forecast_date = last_date + timedelta(days=step)
        month_day = forecast_date.strftime("%m-%d")

        if discount_override_pct is not None:
            planned_discount_pct = discount_override_pct
        else:
            planned_discount_pct = promo_calendar.get(month_day, 0.0)

        day_row = {
            "date": forecast_date.strftime("%Y-%m-%d"),
            "avg_unit_price": last_price,
            "avg_realized_discount": planned_discount_pct,
            "planned_discount_pct": planned_discount_pct,
            "is_promo_day": int(planned_discount_pct > 0),
        }
        buffer.append(day_row)

        calendar_row = add_calendar_features(pd.DataFrame([{"date": forecast_date}])).iloc[0].to_dict()
        feature_row = _row_features(buffer, len(buffer) - 1, calendar_row)
        feature_row.update(
            {
                "avg_unit_price": day_row["avg_unit_price"],
                "avg_realized_discount": day_row["avg_realized_discount"],
                "planned_discount_pct": day_row["planned_discount_pct"],
                "is_promo_day": day_row["is_promo_day"],
            }
        )

        feature_row["category"] = category
        feature_row["sub_category"] = sub_category

        feature_df = pd.DataFrame([feature_row])
        feature_df["category"] = pd.Categorical(feature_df["category"], categories=context["category_categories"])
        feature_df["sub_category"] = pd.Categorical(
            feature_df["sub_category"], categories=context["subcategory_categories"]
        )
        ordered_cols = config.CATEGORICAL_FEATURES + config.NUMERIC_FEATURES
        dmatrix = xgb.DMatrix(feature_df[ordered_cols], enable_categorical=True)
        predicted_units = max(0.0, float(booster.predict(dmatrix)[0]))

        buffer[-1]["units_sold"] = predicted_units
        predictions.append(
            {
                "date": day_row["date"],
                "predicted_units_sold": round(predicted_units, 1),
                "planned_discount_pct": planned_discount_pct,
                "avg_unit_price": last_price,
            }
        )

    return predictions
