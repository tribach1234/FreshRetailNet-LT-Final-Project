
from pathlib import Path
from datetime import date
import json
import math
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "train.parquet"
REPORTS = ROOT / "reports"

REPORTS.mkdir(parents=True, exist_ok=True)

HISTORY_START = date(2025, 6, 9)
HISTORY_END = date(2025, 7, 6)

VAL_START = date(2025, 7, 7)
VAL_END = date(2025, 7, 13)

print("STEP 3.1 - BASELINE FORECASTING", flush=True)

# 1. Read necessary columns
data = (
    pl.scan_parquet(SOURCE)
    .select([
        "dt", "city_id", "first_category_id",
        "store_id", "product_id",
        "sale_amount", "stock_hour6_22_cnt"
    ])
    .with_columns(
        pl.col("dt").str.strptime(
            pl.Date, "%Y-%m-%d"
        ).alias("date")
    )
)

# 2. Strict time separation
print("1/5: Loading historical data...", flush=True)

history = (
    data.filter(
        pl.col("date").is_between(
            HISTORY_START, HISTORY_END
        )
    )
    .select([
        "date", "store_id",
        "product_id", "sale_amount"
    ])
    .collect(engine="streaming")
)

print("2/5: Loading validation period...", flush=True)

validation = (
    data.filter(
        pl.col("date").is_between(
            VAL_START, VAL_END
        )
    )
    .select([
        "date", "city_id", "first_category_id",
        "store_id", "product_id",
        "sale_amount", "stock_hour6_22_cnt"
    ])
    .collect(engine="streaming")
)

if history.is_empty() or validation.is_empty():
    raise ValueError("History or validation is empty.")

KEYS = ["date", "store_id", "product_id"]

if validation.select(KEYS).unique().height != validation.height:
    raise ValueError("Duplicate validation series-date keys.")

# 3. Calculate historical statistics
print("3/5: Building baseline predictions...", flush=True)

pair_mean = (
    history.group_by(["store_id", "product_id"])
    .agg(
        pl.col("sale_amount").mean().alias("mean28")
    )
)

product_mean = (
    history.group_by("product_id")
    .agg(
        pl.col("sale_amount").mean().alias("product_mean")
    )
)

global_mean = history["sale_amount"].mean()

last_week = history.select([
    "store_id",
    "product_id",
    pl.col("date").alias("reference_date"),
    pl.col("sale_amount").alias("lag7")
])

# Only identifiers and dates from validation
# are used to construct predictions.
forecast_keys = (
    validation.select([
        "date", "city_id", "first_category_id",
        "store_id", "product_id"
    ])
    .with_columns(
        pl.col("date")
        .dt.offset_by("-7d")
        .alias("reference_date")
    )
)

predictions = (
    forecast_keys
    .join(
        last_week,
        on=["store_id", "product_id", "reference_date"],
        how="left"
    )
    .join(
        pair_mean,
        on=["store_id", "product_id"],
        how="left"
    )
    .join(
        product_mean,
        on="product_id",
        how="left"
    )
    .with_columns([
        pl.coalesce([
            pl.col("mean28"),
            pl.col("product_mean"),
            pl.lit(global_mean)
        ]).clip(lower_bound=0).alias("pred_mean28"),

        pl.coalesce([
            pl.col("lag7"),
            pl.col("mean28"),
            pl.col("product_mean"),
            pl.lit(global_mean)
        ]).clip(lower_bound=0).alias("pred_lag7")
    ])
)

# 4. Join actual values ONLY for evaluation
results = (
    predictions.join(
        validation.select([
            "date", "store_id", "product_id",
            pl.col("sale_amount").alias("actual"),
            pl.col("stock_hour6_22_cnt").alias(
                "stockout_hours"
            )
        ]),
        on=KEYS,
        how="left"
    )
)

if results.height != validation.height:
    raise ValueError("Prediction row count mismatch.")

if results["actual"].null_count() > 0:
    raise ValueError("Missing validation targets.")

# 5. Evaluation metrics
print("4/5: Evaluating both baselines...", flush=True)

def evaluate(df, prediction_col):
    if df.is_empty():
        return None

    actual = df["actual"]
    predicted = df[prediction_col]

    error = actual - predicted
    absolute_error = error.abs()

    denominator = actual.abs().sum()

    return {
        "MAE": round(float(absolute_error.mean()), 4),
        "RMSE": round(
            math.sqrt(float((error * error).mean())), 4
        ),
        "WAPE_pct": (
            round(
                float(
                    absolute_error.sum() / denominator * 100
                ), 2
            )
            if denominator > 0 else None
        )
    }

no_stockout = results.filter(
    pl.col("stockout_hours") == 0
)

models = {
    "Mean_28_Days": "pred_mean28",
    "Seasonal_Naive_7_Days": "pred_lag7"
}

report = {
    "history_window": [
        str(HISTORY_START), str(HISTORY_END)
    ],
    "validation_window": [
        str(VAL_START), str(VAL_END)
    ],
    "history_rows": history.height,
    "validation_rows": validation.height,
    "validation_no_stockout_rows": no_stockout.height,
    "lag7_available_pct": round(
        100 * predictions["lag7"].is_not_null().sum()
        / predictions.height, 2
    ),
    "pair_mean_available_pct": round(
        100 * predictions["mean28"].is_not_null().sum()
        / predictions.height, 2
    ),
    "models": {}
}

for model_name, column in models.items():
    report["models"][model_name] = {
        "all_rows": evaluate(results, column),
        "no_stockout_rows": evaluate(no_stockout, column)
    }

# Save prediction results for later analysis
results.select([
    "date", "city_id", "first_category_id",
    "store_id", "product_id",
    "actual", "stockout_hours",
    "pred_mean28", "pred_lag7"
]).write_csv(
    REPORTS / "06_baseline_validation_predictions.csv"
)

# Save compact report
report_path = REPORTS / "06_baseline_validation.json"

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("5/5: Results saved.", flush=True)

print("\n===== BASELINE EVALUATION =====")
print(json.dumps(report, indent=2))
print("\nSaved:", report_path)
print("STEP 3.1 COMPLETED")
