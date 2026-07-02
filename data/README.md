# Data

This project uses the **Olist Brazilian E-Commerce Public Dataset** (~100k real orders).

1. Download it from Kaggle: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
2. Extract all CSV files into `data/raw/`
3. Run `python src/prepare_data.py` to generate `data/processed.csv`

Required files:
- olist_orders_dataset.csv
- olist_order_items_dataset.csv
- olist_products_dataset.csv
- olist_customers_dataset.csv
- olist_sellers_dataset.csv
- olist_geolocation_dataset.csv
