r"""Step 3.4: Evaluate the frozen LightGBM model on 14-20 July 2025.

Run from the project root:
    .\.venv\Scripts\python.exe scripts\09_official_eval.py

IMPORTANT: eval sale_amount and stockout fields are never used for features;
they are loaded separately only after predictions have been generated.
"""

from datetime import date, timedelta
from pathlib import Path
import json
import math

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "data" / "raw" / "train.parquet"
EVAL_FILE = ROOT / "data" / "raw" / "eval.parquet"
MODEL_FILE = ROOT / "models" / "07_lightgbm_model.txt"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

ORIGIN = date(2025, 7, 14)
END = ORIGIN + timedelta(days=6)
HISTORY_START = ORIGIN - timedelta(days=28)
FEATURES = [
    "city_id", "store_id", "product_id", "first_category_id",
    "day_of_week", "is_weekend", "horizon",
    "mean28", "mean7", "lag7", "std28", "count28", "count7",
]
KEYS = ["date", "store_id", "product_id"]

print("STEP 3.4 - OFFICIAL EVAL (FROZEN MODEL)", flush=True)
for path in (TRAIN_FILE, EVAL_FILE, MODEL_FILE):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

# 1. Historical features: train data THROUGH 13 JULY ONLY.
print("1/4: Reading historical features...", flush=True)
history = (
    pl.scan_parquet(TRAIN_FILE)
    .filter(pl.col("dt").is_between(
        pl.lit(str(HISTORY_START)), pl.lit(str(ORIGIN - timedelta(days=1)))
    ))
    .select(["dt", "store_id", "product_id", "sale_amount"])
    .with_columns(pl.col("dt").str.strptime(pl.Date, "%Y-%m-%d").alias("date"))
    .drop("dt")
    .collect(engine="streaming")
)
if history.is_empty():
    raise ValueError("The 28-day history window is empty")

# 2. EVAL: read identifiers and calendar ONLY, not future actuals.
print("2/4: Building 7-day forecast...", flush=True)
forecast_keys = (
    pl.scan_parquet(EVAL_FILE)
    .select(["dt", "city_id", "first_category_id", "store_id", "product_id"])
    .with_columns(pl.col("dt").str.strptime(pl.Date, "%Y-%m-%d").alias("date"))
    .drop("dt")
    .collect(engine="streaming")
)
if forecast_keys.height != 70000:
    raise ValueError(f"Expected 70,000 eval rows, received {forecast_keys.height}")
if forecast_keys["date"].min() != ORIGIN or forecast_keys["date"].max() != END:
    raise ValueError("Unexpected eval date range")
if forecast_keys.select(KEYS).unique().height != forecast_keys.height:
    raise ValueError("Duplicate eval date/store/product keys")

pair = history.group_by(["store_id", "product_id"]).agg([
    pl.col("sale_amount").mean().alias("raw_mean28"),
    pl.col("sale_amount").std().alias("std28"),
    pl.len().alias("count28"),
    pl.col("sale_amount")
      .filter(pl.col("date") >= ORIGIN - timedelta(days=7))
      .mean().alias("mean7"),
    pl.col("sale_amount")
      .filter(pl.col("date") >= ORIGIN - timedelta(days=7))
      .len().alias("count7"),
])
last_week = (
    history.filter(pl.col("date") >= ORIGIN - timedelta(days=7))
    .select(["date", "store_id", "product_id",
             pl.col("sale_amount").alias("raw_lag7")])
    .with_columns((pl.col("date") + timedelta(days=7)).alias("date"))
)
global_avg = float(history["sale_amount"].mean())

features = (
    forecast_keys
    .join(pair, on=["store_id", "product_id"], how="left")
    .join(last_week, on=KEYS, how="left")
    .with_columns([
        pl.coalesce([pl.col("raw_mean28"), pl.lit(global_avg)]).alias("mean28"),
        pl.coalesce([pl.col("mean7"), pl.col("raw_mean28"),
                     pl.lit(global_avg)]).alias("mean7"),
        pl.coalesce([pl.col("raw_lag7"), pl.col("raw_mean28"),
                     pl.lit(global_avg)]).alias("lag7"),
        pl.col("std28").fill_null(0),
        pl.col("count28").fill_null(0),
        pl.col("count7").fill_null(0),
        pl.col("date").dt.weekday().alias("day_of_week"),
        (pl.col("date").dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend"),
        ((pl.col("date") - ORIGIN).dt.total_days() + 1).alias("horizon"),
    ])
)
if features.height != forecast_keys.height:
    raise ValueError("Feature generation changed the number of eval rows")
if any(features[c].null_count() > 0 for c in FEATURES):
    raise ValueError("Missing feature values before prediction")

model = lgb.Booster(model_file=str(MODEL_FILE))
if model.num_feature() != len(FEATURES):
    raise ValueError("Feature count does not match saved model")
pred = np.maximum(0, model.predict(features.select(FEATURES).to_numpy()))
features = features.with_columns(pl.Series("ml_prediction", pred))

# 3. Only now, use eval targets for scoring and diagnostic reports.
print("3/4: Scoring predictions with actuals...", flush=True)
actuals = (
    pl.scan_parquet(EVAL_FILE)
    .select(["dt", "store_id", "product_id", "sale_amount",
             "stock_hour6_22_cnt"])
    .with_columns(pl.col("dt").str.strptime(pl.Date, "%Y-%m-%d").alias("date"))
    .drop("dt")
    .rename({"sale_amount": "actual", "stock_hour6_22_cnt": "stockout_hours"})
    .collect(engine="streaming")
)
scored = features.join(actuals, on=KEYS, how="left")
if scored.height != 70000 or scored["actual"].null_count() != 0:
    raise ValueError("Scoring join failed or contains missing actual values")


def metrics(frame: pl.DataFrame, pred_col: str):
    if frame.is_empty():
        return None
    y = frame["actual"].to_numpy()
    p = frame[pred_col].to_numpy()
    error = np.abs(y - p)
    denominator = float(np.abs(y).sum())
    return {
        "MAE": round(float(error.mean()), 4),
        "RMSE": round(math.sqrt(float(np.square(y-p).mean())), 4),
        "WAPE_pct": round(100 * float(error.sum()) / denominator, 2)
                    if denominator > 0 else None,
    }

no_stockout = scored.filter(pl.col("stockout_hours") == 0)
report = {
    "forecast_window": [str(ORIGIN), str(END)],
    "history_window": [str(HISTORY_START), str(ORIGIN - timedelta(days=1))],
    "model_file": MODEL_FILE.name,
    "eval_rows": scored.height,
    "no_stockout_rows": no_stockout.height,
    "pair_history_available_pct": round(100 * features["raw_mean28"].is_not_null().mean(), 2),
    "lag7_available_pct": round(100 * features["raw_lag7"].is_not_null().mean(), 2),
    "results": {
        "Mean28_Baseline": {
            "all": metrics(scored, "mean28"),
            "no_stockout": metrics(no_stockout, "mean28"),
        },
        "LightGBM": {
            "all": metrics(scored, "ml_prediction"),
            "no_stockout": metrics(no_stockout, "ml_prediction"),
        },
    },
}

# Keep a compact prediction file for later Power BI and report preparation.
scored.select([
    "date", "city_id", "first_category_id", "store_id", "product_id",
    "actual", "stockout_hours", "mean28", "ml_prediction",
]).write_parquet(REPORTS / "09_official_eval_predictions.parquet")
with open(REPORTS / "09_official_eval_metrics.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("4/4: Results saved.", flush=True)
print(json.dumps(report, ensure_ascii=False, indent=2))
print("Saved: reports/09_official_eval_metrics.json")
print("STEP 3.4 COMPLETED")
