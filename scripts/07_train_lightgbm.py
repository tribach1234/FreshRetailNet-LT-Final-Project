from pathlib import Path
from datetime import date, timedelta
import json
import numpy as np
import polars as pl
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "train.parquet"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
REPORTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

TRAIN_STARTS = [date(2025, 5, 12) + timedelta(days=7*i) for i in range(8)]
VALID_START = date(2025, 7, 7)
FEATURES = ["city_id", "store_id", "product_id", "first_category_id",
            "day_of_week", "is_weekend", "horizon",
            "mean28", "mean7", "lag7", "std28", "count28", "count7"]

print("STEP 3.2 - LIGHTGBM FORECASTING", flush=True)
all_data = (
    pl.scan_parquet(SOURCE)
    .filter(pl.col("dt").is_between(
        pl.lit("2025-04-14"),
        pl.lit("2025-07-13")
    ))
    .select(["dt", "city_id", "store_id", "product_id", "first_category_id",
             "sale_amount", "stock_hour6_22_cnt"])
    .with_columns(pl.col("dt").str.strptime(pl.Date, "%Y-%m-%d").alias("date"))
    .drop("dt")
    .collect(engine="streaming")
)


def make_week(origin):
    history = all_data.filter(
        pl.col("date").is_between(origin-timedelta(days=28), origin-timedelta(days=1))
    )
    target = all_data.filter(
        pl.col("date").is_between(origin, origin+timedelta(days=6))
    ).select("date", "city_id", "first_category_id", "store_id", "product_id", "sale_amount", "stock_hour6_22_cnt")

    pair = history.group_by(["store_id", "product_id"]).agg([
        pl.col("sale_amount").mean().alias("mean28"),
        pl.col("sale_amount").std().alias("std28"),
        pl.len().alias("count28"),
        pl.col("sale_amount")
          .filter(pl.col("date") >= origin-timedelta(days=7))
          .mean().alias("mean7"),
        pl.col("sale_amount")
          .filter(pl.col("date") >= origin-timedelta(days=7))
          .len().alias("count7")
    ])
    last_week = (
        history.filter(pl.col("date") >= origin-timedelta(days=7))
        .select("date", "store_id", "product_id", pl.col("sale_amount").alias("lag7"))
        .with_columns((pl.col("date")+timedelta(days=7)).alias("date"))
    )
    global_avg = history["sale_amount"].mean()

    week = (
        target.join(pair, on=["store_id", "product_id"], how="left")
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
            ((pl.col("date")-origin).dt.total_days()+1).alias("horizon")
        ])
    )
    if week.select([pl.col(c).null_count() for c in FEATURES]).row(0) != (0,)*len(FEATURES):
        raise RuntimeError(f"Missing modeling features for week {origin}")
    if week.select(["date", "store_id", "product_id"]).unique().height != week.height:
        raise RuntimeError(f"Duplicate keys for week {origin}")
    return week

print("Creating training windows...", flush=True)
train_frames = []
for start in TRAIN_STARTS:
    frame = make_week(start)
    train_frames.append(frame)
    print(f" {start}: {frame.height:,} records", flush=True)
train = pl.concat(train_frames)
print("Preparing validation week...", flush=True)
valid = make_week(VALID_START)

X_train = train.select(FEATURES).to_numpy()
y_train = train["sale_amount"].to_numpy()
X_valid = valid.select(FEATURES).to_numpy()

print(f"Training LightGBM on {train.height:,} rows...", flush=True)
model = LGBMRegressor(
    objective="regression", n_estimators=200, learning_rate=0.05,
    num_leaves=31, min_child_samples=50, random_state=42,
    n_jobs=4, verbosity=-1
)
model.fit(X_train, y_train, categorical_feature=[0, 1, 2, 3, 4])
valid = valid.with_columns(pl.Series("ml_prediction", np.maximum(0, model.predict(X_valid))))


def evaluate(frame, prediction_col):
    if not frame.height:
        return None
    actual = frame["sale_amount"].to_numpy()
    pred = frame[prediction_col].to_numpy()
    diff = actual - pred
    denom = np.abs(actual).sum()
    return {
        "MAE": round(float(np.abs(diff).mean()), 4),
        "RMSE": round(float(np.sqrt(np.square(diff).mean())), 4),
        "WAPE_pct": round(float(100*np.abs(diff).sum()/denom), 2) if denom else None,
    }

no_stockout = valid.filter(pl.col("stock_hour6_22_cnt") == 0)
report = {
    "training_week_starts": [str(d) for d in TRAIN_STARTS],
    "validation_dates": ["2025-07-07", "2025-07-13"],
    "training_rows": train.height,
    "validation_rows": valid.height,
    "validation_no_stockout_rows": no_stockout.height,
    "feature_columns": FEATURES,
    "results": {
        "Mean28_Baseline": {"all": evaluate(valid,"mean28"), "no_stockout": evaluate(no_stockout,"mean28")},
        "LightGBM": {"all": evaluate(valid,"ml_prediction"), "no_stockout": evaluate(no_stockout,"ml_prediction")},
    },
}
with open(REPORTS / "07_ml_validation.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
model.booster_.save_model(str(MODELS / "07_lightgbm_model.txt"))
print(json.dumps(report["results"], ensure_ascii=False, indent=2))
print("Saved: reports/07_ml_validation.json")
print("STEP 3.2 COMPLETED")
