"""
Streamlit app for delivery time prediction.

Run with:
    streamlit run app/streamlit_app.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/model.joblib")
METRICS_PATH = Path("models/metrics.json")

BRAZIL_STATES = [
    "SP", "RJ", "MG", "RS", "PR", "SC", "BA", "DF", "ES", "GO",
    "PE", "CE", "PA", "MT", "MA", "MS", "PB", "PI", "RN", "AL",
    "SE", "TO", "RO", "AM", "AC", "AP", "RR",
]

st.set_page_config(page_title="Delivery Time Predictor", page_icon="📦")

st.title("📦 Delivery Time Predictor")
st.caption(
    "Predicts how many days an e-commerce order will take to arrive, "
    "trained on 90k+ real orders from the Olist Brazilian E-Commerce dataset."
)

if not MODEL_PATH.exists():
    st.error(
        "Model not found. Train it first:\n\n"
        "```\npython src/prepare_data.py\npython src/train.py\n```"
    )
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

if METRICS_PATH.exists():
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    best = metrics["results"][metrics["best_model"]]
    col1, col2, col3 = st.columns(3)
    col1.metric("Model", metrics["best_model"])
    col2.metric("MAE (days)", best["mae"])
    col3.metric("R²", best["r2"])

st.divider()
st.subheader("Order details")

c1, c2 = st.columns(2)
with c1:
    distance_km = st.slider("Seller → customer distance (km)", 0, 3000, 500)
    customer_state = st.selectbox("Customer state", BRAZIL_STATES)
    same_state = st.checkbox("Seller in same state as customer", value=False)
    n_items = st.number_input("Number of items", 1, 20, 1)
    price = st.number_input("Order price (R$)", 1.0, 5000.0, 120.0)
with c2:
    freight_value = st.number_input("Freight cost (R$)", 0.0, 500.0, 20.0)
    product_weight_g = st.number_input("Product weight (g)", 50, 30000, 800)
    volume_cm3 = st.number_input("Product volume (cm³)", 100, 300000, 5000)
    estimated_days = st.slider("Platform's promised estimate (days)", 1, 45, 15)

c3, c4 = st.columns(2)
with c3:
    purchase_weekday = st.selectbox(
        "Purchase day",
        options=list(range(7)),
        format_func=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d],
    )
with c4:
    purchase_hour = st.slider("Purchase hour", 0, 23, 14)

purchase_month = st.slider("Purchase month", 1, 12, 6)

if st.button("Predict delivery time", type="primary", use_container_width=True):
    row = pd.DataFrame(
        [
            {
                "distance_km": distance_km,
                "same_state": int(same_state),
                "freight_value": freight_value,
                "price": price,
                "n_items": n_items,
                "product_weight_g": product_weight_g,
                "volume_cm3": volume_cm3,
                "estimated_days": estimated_days,
                "purchase_weekday": purchase_weekday,
                "purchase_month": purchase_month,
                "purchase_hour": purchase_hour,
                "is_weekend": int(purchase_weekday >= 5),
                "customer_state": customer_state,
            }
        ]
    )
    pred = float(model.predict(row)[0])
    st.success(f"### Estimated delivery time: **{pred:.1f} days**")
    diff = estimated_days - pred
    if diff > 2:
        st.info(f"Model predicts arrival ~{diff:.0f} days earlier than the platform's promise.")
    elif diff < -2:
        st.warning(f"Model predicts a delay of ~{-diff:.0f} days vs the platform's promise.")
