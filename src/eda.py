"""
Exploratory Data Analysis for the delivery time dataset.

Generates key visualizations and saves them to reports/figures/.

Usage:
    python src/eda.py --data data/processed.csv
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed.csv", type=Path)
    parser.add_argument("--out-dir", default="reports/figures", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)

    # 1. Distribution of delivery times
    fig, ax = plt.subplots(figsize=(8, 5))
    df["delivery_days"].hist(bins=60, ax=ax, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Delivery time (days)")
    ax.set_ylabel("Number of orders")
    ax.set_title("Distribution of Delivery Times")
    fig.tight_layout()
    fig.savefig(args.out_dir / "delivery_time_distribution.png", dpi=150)

    # 2. Distance vs delivery time
    fig, ax = plt.subplots(figsize=(8, 5))
    sample = df.sample(min(5000, len(df)), random_state=42)
    ax.scatter(sample["distance_km"], sample["delivery_days"], alpha=0.15, s=8, color="#4C72B0")
    ax.set_xlabel("Seller-customer distance (km)")
    ax.set_ylabel("Delivery time (days)")
    ax.set_title("Distance vs Delivery Time")
    fig.tight_layout()
    fig.savefig(args.out_dir / "distance_vs_delivery.png", dpi=150)

    # 3. Average delivery time by customer state (top 15)
    fig, ax = plt.subplots(figsize=(10, 5))
    state_avg = (
        df.groupby("customer_state")["delivery_days"].mean().sort_values(ascending=False).head(15)
    )
    state_avg.plot(kind="bar", ax=ax, color="#DD8452", edgecolor="white")
    ax.set_xlabel("Customer state")
    ax.set_ylabel("Avg delivery time (days)")
    ax.set_title("Average Delivery Time by State (Top 15 Slowest)")
    fig.tight_layout()
    fig.savefig(args.out_dir / "delivery_by_state.png", dpi=150)

    # 4. Weekend vs weekday purchases
    fig, ax = plt.subplots(figsize=(6, 5))
    df.groupby("is_weekend")["delivery_days"].mean().rename(
        {0: "Weekday", 1: "Weekend"}
    ).plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"], edgecolor="white")
    ax.set_ylabel("Avg delivery time (days)")
    ax.set_title("Purchase Day: Weekday vs Weekend")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(args.out_dir / "weekend_effect.png", dpi=150)

    print(f"Saved 4 figures -> {args.out_dir}")


if __name__ == "__main__":
    main()
