"""
Train delivery time prediction models.

Compares a Linear Regression baseline against XGBoost,
prints evaluation metrics, and saves the best model.

Usage:
    python src/train.py --data data/processed.csv
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

NUMERIC_FEATURES = [
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
]
CATEGORICAL_FEATURES = ["customer_state"]
TARGET = "delivery_days"


def build_pipelines() -> dict[str, Pipeline]:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return {
        "linear_regression": Pipeline(
            [("prep", preprocess), ("model", LinearRegression())]
        ),
        "xgboost": Pipeline(
            [
                ("prep", preprocess),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=400,
                        max_depth=7,
                        learning_rate=0.08,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    return {
        "mae": round(float(mean_absolute_error(y_test, preds)), 3),
        "rmse": round(float(root_mean_squared_error(y_test, preds)), 3),
        "r2": round(float(r2_score(y_test, preds)), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed.csv", type=Path)
    parser.add_argument("--model-dir", default="models", type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df["product_weight_g"] = df["product_weight_g"].fillna(df["product_weight_g"].median())
    df["volume_cm3"] = df["volume_cm3"].fillna(df["volume_cm3"].median())

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Naive baseline: always predict the mean delivery time
    naive_pred = np.full(len(y_test), y_train.mean())
    results = {
        "naive_mean": {
            "mae": round(float(mean_absolute_error(y_test, naive_pred)), 3),
            "rmse": round(float(root_mean_squared_error(y_test, naive_pred)), 3),
            "r2": 0.0,
        }
    }

    best_name, best_model, best_rmse = None, None, float("inf")
    for name, pipeline in build_pipelines().items():
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        results[name] = metrics
        print(f"{name:20s} MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  R2={metrics['r2']:.3f}")
        if metrics["rmse"] < best_rmse:
            best_name, best_model, best_rmse = name, pipeline, metrics["rmse"]

    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, args.model_dir / "model.joblib")
    with open(args.model_dir / "metrics.json", "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    print(f"\nBest model: {best_name} (RMSE={best_rmse:.2f})")
    print(f"Saved -> {args.model_dir / 'model.joblib'}")


if __name__ == "__main__":
    main()
