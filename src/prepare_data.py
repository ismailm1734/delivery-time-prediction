"""
Prepare the Olist e-commerce dataset for delivery time prediction.

Merges raw Olist CSV files, engineers features, and saves a clean
training dataset to data/processed.csv.

Usage:
    python src/prepare_data.py --data-dir data/raw --out data/processed.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def haversine(lat1, lng1, lat2, lng2):
    """Great-circle distance between two points, in kilometers (vectorized)."""
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def load_raw(data_dir: Path) -> dict[str, pd.DataFrame]:
    files = {
        "orders": "olist_orders_dataset.csv",
        "items": "olist_order_items_dataset.csv",
        "products": "olist_products_dataset.csv",
        "customers": "olist_customers_dataset.csv",
        "sellers": "olist_sellers_dataset.csv",
        "geo": "olist_geolocation_dataset.csv",
    }
    missing = [f for f in files.values() if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing files in {data_dir}: {missing}. "
            "Download the Olist dataset from Kaggle first (see data/README.md)."
        )
    return {name: pd.read_csv(data_dir / fname) for name, fname in files.items()}


def build_dataset(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    orders = raw["orders"].copy()

    # Keep only delivered orders (we need the actual delivery date as target)
    orders = orders[orders["order_status"] == "delivered"]

    date_cols = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")
    orders = orders.dropna(subset=date_cols)

    # ---- Target: actual delivery time in days ----
    orders["delivery_days"] = (
        orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    # Olist's own promised estimate (a strong baseline feature)
    orders["estimated_days"] = (
        orders["order_estimated_delivery_date"] - orders["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    # Remove obvious data errors / extreme outliers
    orders = orders[(orders["delivery_days"] > 0) & (orders["delivery_days"] <= 60)]

    # ---- Aggregate order items (an order can contain multiple items) ----
    items = raw["items"]
    items_agg = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        price=("price", "sum"),
        freight_value=("freight_value", "sum"),
        seller_id=("seller_id", "first"),
        product_id=("product_id", "first"),
    )

    df = orders.merge(items_agg, on="order_id", how="inner")

    # ---- Product size / weight ----
    products = raw["products"][
        [
            "product_id",
            "product_category_name",
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
        ]
    ].copy()
    products["volume_cm3"] = (
        products["product_length_cm"]
        * products["product_height_cm"]
        * products["product_width_cm"]
    )
    df = df.merge(products, on="product_id", how="left")

    # ---- Customer & seller location ----
    geo = (
        raw["geo"]
        .groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
        .mean()
        .reset_index()
    )

    customers = raw["customers"][
        ["customer_id", "customer_zip_code_prefix", "customer_state"]
    ]
    sellers = raw["sellers"][["seller_id", "seller_zip_code_prefix", "seller_state"]]

    df = df.merge(customers, on="customer_id", how="left")
    df = df.merge(sellers, on="seller_id", how="left")

    df = df.merge(
        geo.rename(
            columns={
                "geolocation_zip_code_prefix": "customer_zip_code_prefix",
                "geolocation_lat": "cust_lat",
                "geolocation_lng": "cust_lng",
            }
        ),
        on="customer_zip_code_prefix",
        how="left",
    )
    df = df.merge(
        geo.rename(
            columns={
                "geolocation_zip_code_prefix": "seller_zip_code_prefix",
                "geolocation_lat": "sell_lat",
                "geolocation_lng": "sell_lng",
            }
        ),
        on="seller_zip_code_prefix",
        how="left",
    )

    df["distance_km"] = haversine(
        df["cust_lat"], df["cust_lng"], df["sell_lat"], df["sell_lng"]
    )
    df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)

    # ---- Time-based features ----
    df["purchase_weekday"] = df["order_purchase_timestamp"].dt.weekday
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    df["is_weekend"] = (df["purchase_weekday"] >= 5).astype(int)

    feature_cols = [
        "distance_km",
        "same_state",
        "freight_value",
        "price",
        "n_items",
        "product_weight_g",
        "volume_cm3",
        "estimated_days",
        "purchase_weekday",
        "purchase_month",
        "purchase_hour",
        "is_weekend",
        "customer_state",
    ]
    target_col = "delivery_days"

    out = df[feature_cols + [target_col]].dropna(
        subset=["distance_km", "estimated_days", target_col]
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--out", default="data/processed.csv", type=Path)
    args = parser.parse_args()

    raw = load_raw(args.data_dir)
    df = build_dataset(raw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df):,} rows -> {args.out}")


if __name__ == "__main__":
    main()
