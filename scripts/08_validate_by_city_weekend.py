r"""Step 3.3: LightGBM validation diagnostics by city and weekday/weekend.

Run from the FreshRetailNet project root:
    .\.venv\Scripts\python.exe scripts\08_validate_by_city_weekend.py

Uses the SAVED 07_lightgbm_model.txt. Does not retrain the model.
"""

from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "train.parquet"
MODEL_FILE = ROOT / "models" / "07_lightgbm_model.txt"
OUTPUT = ROOT / "reports" / "08_city_weekend_model_errors.csv"

VALID_START = date(2025, 7, 7)
VALID_END = date(2025, 7, 13)
HISTORY_START = VALID_START - timedelta(days=28)

FEATURES = [
    "city_id", "store_id", "product_id", "first_category_id",
    "day_of_week", "is_weekend", "horizon",
    "mean28", "mean7", "lag7", "std28", "count28", "count7",
]

print("STEP 3.3 - CITY & WEEKEND MODEL ERROR ANALYSIS", flush=True)
if not SOURCE.exists():
    raise FileNotFoundError(f"Dataset not found: {SOURCE}")
if not MODEL_FILE.exists():
    raise FileNotFoundError(f"Saved model not found: {MODEL_FILE}. Run Step 3.2 first.")

# Keep the precise feature logic of 07_train_lightgbm.py for a fair comparison.
data = (
    pl.scan_parquet(SOURCE)
    .filter(pl.col("dt").is_between(
        pl.lit(str(HISTORY_START)), pl.lit(str(VALID_END))
    ))
    .select([
        "dt", "city_id", "store_id", "product_id", "first_category_id",
        "sale_amount", "stock_hour6_22_cnt"
    ])
    .with_columns(pl.col("dt").str.strptime(pl.Date, "%Y-%m-%d").alias("date"))
    .drop("dt")
    .collect(engine="streaming")
)

history = data.filter(pl.col("date") < VALID_START)
actuals = data.filter(pl.col("date") >= VALID_START)
print(f"History rows: {history.height:,}; validation rows: {actuals.height:,}", flush=True)
if actuals.height != 70765:
    raise ValueError(f"Unexpected validation rows: {actuals.height} (expected 70,765)")

pair = history.group_by(["store_id", "product_id"]).agg([
    pl.col("sale_amount").mean().alias("mean28"),
    pl.col("sale_amount").std().alias("std28"),
    pl.len().alias("count28"),
    pl.col("sale_amount")
      .filter(pl.col("date") >= VALID_START - timedelta(days=7))
      .mean().alias("mean7"),
    pl.col("sale_amount")
      .filter(pl.col("date") >= VALID_START - timedelta(days=7))
      .len().alias("count7"),
])

last_week = (
    history.filter(pl.col("date") >= VALID_START - timedelta(days=7))
    .select("date", "store_id", "product_id",
            pl.col("sale_amount").alias("lag7"))
    .with_columns((pl.col("date") + timedelta(days=7)).alias("date"))
)

global_avg = history["sale_amount"].mean()

valid = (
    actuals
    .join(pair, on=["store_id", "product_id"], how="left")
    .join(last_week, on=["date", "store_id", "product_id"], how="left")
    .with_columns([
        pl.coalesce([pl.col("mean28"), pl.lit(global_avg)]).alias("mean28"),
        pl.coalesce([pl.col("mean7"), pl.col("mean28"), pl.lit(global_avg)]).alias("mean7"),
        pl.coalesce([pl.col("lag7"), pl.col("mean28"), pl.lit(global_avg)]).alias("lag7"),
        pl.col("std28").fill_null(0),
        pl.col("count28").fill_null(0),
        pl.col("count7").fill_null(0),
        pl.col("date").dt.weekday().alias("day_of_week"),
        (pl.col("date").dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend"),
        ((pl.col("date") - VALID_START).dt.total_days() + 1).alias("horizon"),
    ])
)

if valid.height != actuals.height:
    raise RuntimeError("Validation row count changed during feature engineering")
if any(valid[col].null_count() for col in FEATURES):
    raise RuntimeError("Some modeling features are NULL; check feature engineering")

print("Loading saved LightGBM and predicting (no retraining)...", flush=True)
model = lgb.Booster(model_file=str(MODEL_FILE))
if model.num_feature() != len(FEATURES):
    raise RuntimeError(f"Expected {len(FEATURES)} model features, found {model.num_feature()}")
pred = np.maximum(0, model.predict(valid.select(FEATURES).to_numpy()))
valid = valid.with_columns(pl.Series("prediction", pred))

valid = valid.with_columns([
    (pl.col("prediction") - pl.col("sale_amount")).abs().alias("ml_abs_error"),
    (pl.col("mean28") - pl.col("sale_amount")).abs().alias("baseline_abs_error"),
    pl.col("sale_amount").abs().alias("abs_actual"),
    (pl.col("stock_hour6_22_cnt") > 0).cast(pl.Float64).alias("is_stockout"),
])

# Reproducibility check: our output should match Step 3.2 reported overall WAPE.
actual_total = valid["abs_actual"].sum()
ml_wape = 100 * valid["ml_abs_error"].sum() / actual_total
baseline_wape = 100 * valid["baseline_abs_error"].sum() / actual_total
print(f"Overall WAPE: LightGBM={ml_wape:.2f}%; Mean28={baseline_wape:.2f}%", flush=True)
if abs(ml_wape - 31.17) > 0.15 or abs(baseline_wape - 36.86) > 0.15:
    raise RuntimeError(
        "WAPE does not match Step 3.2. Check the saved model, feature order, and date ranges."
    )

# One row per city and day type; these are post-hoc metrics (not training inputs).
summary = (
    valid.group_by(["city_id", "is_weekend"])
    .agg([
        pl.len().alias("n_rows"),
        pl.col("store_id").n_unique().alias("n_stores"),
        pl.col("product_id").n_unique().alias("n_products"),
        pl.col("abs_actual").sum().alias("abs_actual_sum"),
        pl.col("baseline_abs_error").sum().alias("baseline_abs_error_sum"),
        pl.col("ml_abs_error").sum().alias("ml_abs_error_sum"),
        (pl.col("ml_abs_error").mean()).alias("ml_mae"),
        (pl.col("baseline_abs_error").mean()).alias("baseline_mae"),
        (pl.col("is_stockout").mean() * 100).alias("stockout_pct"),
    ])
    .with_columns([
        pl.when(pl.col("abs_actual_sum") > 0)
          .then(100 * pl.col("baseline_abs_error_sum") / pl.col("abs_actual_sum"))
          .otherwise(None).alias("baseline_wape_pct"),
        pl.when(pl.col("abs_actual_sum") > 0)
          .then(100 * pl.col("ml_abs_error_sum") / pl.col("abs_actual_sum"))
          .otherwise(None).alias("ml_wape_pct"),
    ])
    .with_columns(
        (pl.col("baseline_wape_pct") - pl.col("ml_wape_pct"))
        .alias("improvement_pp")
    )
    .select([
        "city_id", "is_weekend", "n_rows", "n_stores", "n_products",
        "stockout_pct", "baseline_wape_pct", "ml_wape_pct",
        "improvement_pp", "baseline_mae", "ml_mae"
    ])
    .sort(["city_id", "is_weekend"])
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
summary.write_csv(OUTPUT, float_precision=4)
print(f"Groups: {summary.height} (expected 36)")
print(f"Cities where LightGBM improved, city-day groups: {(summary['improvement_pp'] > 0).sum()}/{summary.height}")
print(f"Saved: {OUTPUT}")
print("STEP 3.3 COMPLETED")
