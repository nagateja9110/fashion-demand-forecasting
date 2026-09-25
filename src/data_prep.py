"""Build a continuous daily (Category, Sub Category) demand panel from the raw
transactional exports (transactions, products, discounts).

Raw grain is one line per invoice line item; individual SKU-day series are too
sparse/zero-inflated for reliable forecasting, so demand is aggregated to the
product-category level, which is the standard granularity for retail demand
planning and inventory decisions.
"""
import pandas as pd

from . import config

TRANSACTION_COLS = ["Product ID", "Unit Price", "Quantity", "Date", "Discount", "Transaction Type"]
TRANSACTION_DTYPES = {
    "Product ID": "int32",
    "Unit Price": "float32",
    "Quantity": "int16",
    "Discount": "float32",
    "Transaction Type": "category",
}


def load_products() -> pd.DataFrame:
    products = pd.read_csv(config.PRODUCTS_CSV, usecols=["Product ID", "Category", "Sub Category"])
    products.columns = ["product_id", "category", "sub_category"]
    return products


def load_transactions() -> pd.DataFrame:
    tx = pd.read_csv(
        config.TRANSACTIONS_CSV,
        usecols=TRANSACTION_COLS,
        dtype=TRANSACTION_DTYPES,
        parse_dates=["Date"],
    )
    tx.columns = ["product_id", "unit_price", "quantity", "date", "discount", "transaction_type"]
    tx["date"] = tx["date"].dt.normalize()
    # Returns reduce net demand; a "Return" row's Quantity is stored positive, so subtract it.
    sign = tx["transaction_type"].map({"Sale": 1, "Return": -1}).astype("int8")
    tx["net_quantity"] = tx["quantity"] * sign
    return tx


def aggregate_daily_sales(tx: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    tx = tx.merge(products, on="product_id", how="left")
    daily = (
        tx.groupby(["category", "sub_category", "date"], observed=True)
        .agg(
            units_sold=("net_quantity", "sum"),
            avg_unit_price=("unit_price", "mean"),
            avg_realized_discount=("discount", "mean"),
            n_transactions=("net_quantity", "size"),
        )
        .reset_index()
    )
    return daily


def make_full_panel(daily: pd.DataFrame) -> pd.DataFrame:
    """Reindex every (category, sub_category) group over the full date range so
    days with zero sales are explicit rows, not missing ones -- required for
    correct lag/rolling features later."""
    full_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    groups = daily[["category", "sub_category"]].drop_duplicates()

    panels = []
    for _, group in groups.iterrows():
        mask = (daily["category"] == group["category"]) & (daily["sub_category"] == group["sub_category"])
        g = daily.loc[mask].set_index("date").reindex(full_dates)
        g["category"] = group["category"]
        g["sub_category"] = group["sub_category"]
        g["units_sold"] = g["units_sold"].fillna(0)
        g["n_transactions"] = g["n_transactions"].fillna(0)
        g["avg_unit_price"] = g["avg_unit_price"].ffill().bfill()
        g["avg_realized_discount"] = g["avg_realized_discount"].fillna(0)
        panels.append(g)

    panel = pd.concat(panels).rename_axis("date").reset_index()
    return panel


def load_promo_calendar() -> pd.DataFrame:
    """Expand the (sparse) discount campaign calendar into one row per
    (category, sub_category, date) with the planned discount rate -- known in
    advance from the promo calendar, so safe to use as a forecasting feature."""
    discounts = pd.read_csv(config.DISCOUNTS_CSV, parse_dates=["Start", "End"])
    discounts.columns = ["start", "end", "planned_discount_pct", "description", "category", "sub_category"]

    rows = []
    for _, row in discounts.iterrows():
        dates = pd.date_range(row["start"], row["end"], freq="D")
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "category": row["category"],
                    "sub_category": row["sub_category"],
                    "planned_discount_pct": row["planned_discount_pct"],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def build_daily_demand() -> pd.DataFrame:
    products = load_products()
    tx = load_transactions()
    daily = aggregate_daily_sales(tx, products)
    panel = make_full_panel(daily)

    promo = load_promo_calendar()
    panel = panel.merge(promo, on=["category", "sub_category", "date"], how="left")
    panel["planned_discount_pct"] = panel["planned_discount_pct"].fillna(0.0)
    panel["is_promo_day"] = (panel["planned_discount_pct"] > 0).astype("int8")

    panel = panel.sort_values(["category", "sub_category", "date"]).reset_index(drop=True)
    return panel


def main() -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel = build_daily_demand()
    panel.to_parquet(config.DAILY_DEMAND_PARQUET, index=False)
    print(f"Saved daily demand panel: {panel.shape} -> {config.DAILY_DEMAND_PARQUET}")
    print(panel.head())


if __name__ == "__main__":
    main()
