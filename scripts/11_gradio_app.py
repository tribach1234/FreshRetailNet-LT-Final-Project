"""FreshRetailNet-LT | Gradio demo of the frozen seven-day LightGBM model.

Uses ONLY train.parquet sale history through 2025-07-13 for forecast features.
Uses eval.parquet ONLY for identifying eligible city/store/product/category keys;
never reads evaluation sale_amount, stockout status, or other future outcomes.
"""

from datetime import date, timedelta
from pathlib import Path

import gradio as gr
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "raw" / "train.parquet"
EVAL = ROOT / "data" / "raw" / "eval.parquet"
MODEL = ROOT / "models" / "07_lightgbm_model.txt"
ORIGIN = date(2025, 7, 14)
HISTORY_START = ORIGIN - timedelta(days=28)
FEATURE_NAMES = [
    "city_id", "store_id", "product_id", "first_category_id",
    "day_of_week", "is_weekend", "horizon", "mean28", "mean7",
    "lag7", "std28", "count28", "count7",
]

for path in [TRAIN, EVAL, MODEL]:
    if not path.exists():
        raise FileNotFoundError(f"Required project file not found: {path}")

print("Loading saved model and recent sales history...", flush=True)
model = lgb.Booster(model_file=str(MODEL))
if model.num_feature() != len(FEATURE_NAMES):
    raise ValueError("Saved model does not have 13 expected features")

history = (
    pl.scan_parquet(TRAIN)
    .filter(pl.col("dt").is_between(pl.lit(str(HISTORY_START)), pl.lit(str(ORIGIN - timedelta(days=1)))))
    .select(["dt", "store_id", "product_id", "sale_amount"])
    .with_columns(pl.col("dt").str.strptime(pl.Date, "%Y-%m-%d").alias("date"))
    .drop("dt")
    .collect(engine="streaming")
)
if history.is_empty():
    raise ValueError("History is empty")

global_mean = float(history["sale_amount"].mean())
last_week_start = ORIGIN - timedelta(days=7)

stats = history.group_by(["store_id", "product_id"]).agg([
    pl.col("sale_amount").mean().alias("mean28"),
    pl.col("sale_amount").std().alias("std28"),
    pl.len().alias("count28"),
    pl.col("sale_amount").filter(pl.col("date") >= last_week_start).mean().alias("mean7"),
    pl.col("sale_amount").filter(pl.col("date") >= last_week_start).len().alias("count7"),
])
stat_lookup = {
    (int(r["store_id"]), int(r["product_id"])): r
    for r in stats.iter_rows(named=True)
}
lag_lookup = {
    (int(r["store_id"]), int(r["product_id"]), r["date"]): float(r["sale_amount"])
    for r in history.filter(pl.col("date") >= last_week_start).iter_rows(named=True)
}

# Metadata only: never access target or stockout columns from eval.parquet.
catalog = (
    pl.scan_parquet(EVAL)
    .select(["city_id", "store_id", "product_id", "first_category_id"])
    .unique()
    .collect(engine="streaming")
)
catalog_lookup = {}
city_to_stores = {}
store_to_products = {}
for r in catalog.iter_rows(named=True):
    city, store, product, category = (
        int(r["city_id"]), int(r["store_id"]),
        int(r["product_id"]), int(r["first_category_id"]),
    )
    catalog_lookup[(city, store, product)] = category
    city_to_stores.setdefault(city, set()).add(store)
    store_to_products.setdefault((city, store), set()).add(product)

cities = sorted(city_to_stores)
if not cities:
    raise ValueError("No valid city/store/product combinations found")


def stores_for(city):
    return sorted(city_to_stores.get(int(city), [])) if city is not None else []


def products_for(city, store):
    if city is None or store is None:
        return []
    return sorted(store_to_products.get((int(city), int(store)), []))


def change_city(city):
    stores = stores_for(city)
    store = stores[0] if stores else None
    products = products_for(city, store)
    return (gr.update(choices=stores, value=store),
            gr.update(choices=products, value=products[0] if products else None))


def change_store(city, store):
    products = products_for(city, store)
    return gr.update(choices=products, value=products[0] if products else None)


def predict(city, store, product):
    if any(v is None for v in (city, store, product)):
        raise gr.Error("Please select City, Store and Product")
    city, store, product = int(city), int(store), int(product)
    key = (city, store, product)
    if key not in catalog_lookup:
        raise gr.Error("This combination was not found in the evaluation catalog")
    category = catalog_lookup[key]
    st = stat_lookup.get((store, product), {})
    mean28 = float(st.get("mean28") if st.get("mean28") is not None else global_mean)
    mean7 = float(st.get("mean7") if st.get("mean7") is not None else mean28)
    std28 = float(st.get("std28") if st.get("std28") is not None else 0)
    count28, count7 = int(st.get("count28", 0)), int(st.get("count7", 0))

    days = [ORIGIN + timedelta(days=i) for i in range(7)]
    features = []
    for i, day in enumerate(days):
        lag7 = lag_lookup.get((store, product, day - timedelta(days=7)), mean28)
        features.append([
            city, store, product, category,
            day.isoweekday(), int(day.isoweekday() >= 6), i + 1,
            mean28, mean7, lag7, std28, count28, count7,
        ])

    matrix = np.asarray(features, dtype=np.float64)
    forecasts = np.maximum(0.0, model.predict(matrix))
    dates = [d.isoformat() for d in days]
    table = pd.DataFrame({
        "Date": dates,
        "LightGBM Forecast": np.round(forecasts, 4),
        "Mean28 Baseline": np.repeat(round(mean28, 4), 7),
    })

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(days, forecasts, marker="o", label="LightGBM forecast")
    ax.plot(days, [mean28] * 7, linestyle="--", label="Mean28 baseline")
    ax.set_title("Seven-day normalized sales forecast")
    ax.set_ylabel("Normalized sales (not currency)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()

    summary = (
        f"**City:** {city}  |  **Store:** {store}  |  **Product:** {product}  "
        f"|  **Category:** {category}  \n\n"
        f"**Forecast week:** 14–20 July 2025  \n"
        f"**7-day predicted total (normalized):** {float(forecasts.sum()):,.3f}  \n"
        f"**Baseline total (normalized):** {7 * mean28:,.3f}  \n\n"
        "This demo uses the frozen LightGBM model and sales history up to 13 July 2025. "
        "Evaluation targets are not inputs. It does not forecast the current week."
    )
    return summary, table, fig


initial_city = cities[0]
initial_store = stores_for(initial_city)[0]
initial_product = products_for(initial_city, initial_store)[0]

with gr.Blocks(title="FreshRetailNet-LT Forecast") as demo:
    gr.Markdown("# FreshRetailNet-LT | Seven-day Sales Forecast")
    gr.Markdown(
        "**Project demo:** predict normalized sales for **14–20 July 2025** "
        "using the saved LightGBM model and data available through 13 July 2025. "
        "City, store and product identifiers are anonymized."
    )
    with gr.Row():
        city = gr.Dropdown(choices=cities, value=initial_city, label="City ID")
        store = gr.Dropdown(choices=stores_for(initial_city), value=initial_store, label="Store ID")
        product = gr.Dropdown(
            choices=products_for(initial_city, initial_store),
            value=initial_product, label="Product ID",
        )
    button = gr.Button("Predict 7 days", variant="primary")
    summary = gr.Markdown()
    result = gr.Dataframe(label="Forecast table", interactive=False)
    plot = gr.Plot(label="Forecast chart")

    city.change(change_city, inputs=city, outputs=[store, product])
    store.change(change_store, inputs=[city, store], outputs=product)
    button.click(predict, inputs=[city, store, product], outputs=[summary, result, plot])

if __name__ == "__main__":
    print("App ready: http://127.0.0.1:7860", flush=True)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)
