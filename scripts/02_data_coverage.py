
from pathlib import Path
import json
import polars as pl

# Project paths
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "train.parquet"
OUTPUT = ROOT / "reports" / "02_coverage_summary.json"

print("STEP 2.1 - DATA COVERAGE CHECK", flush=True)

# Read only the columns needed for coverage analysis
df = (
    pl.scan_parquet(SOURCE)
    .select([
        "dt", "city_id", "store_id", "product_id"
    ])
    .with_columns(
        pl.col("dt").cast(pl.Date, strict=False).alias("date")
    )
)

# 1. Global date coverage
print("1/4: Checking overall date coverage...", flush=True)

overview = df.select(
    pl.len().alias("total_rows"),
    pl.col("date").min().alias("min_date"),
    pl.col("date").max().alias("max_date"),
    pl.col("date").n_unique().alias("active_days"),
    pl.col("date").null_count().alias("invalid_dates")
).collect(engine="streaming").to_dicts()[0]

if overview["invalid_dates"] > 0:
    raise ValueError("Invalid dates found in train.parquet")

calendar_days = (
    overview["max_date"] - overview["min_date"]
).days + 1

overview["calendar_days"] = calendar_days
overview["days_without_records"] = (
    calendar_days - overview["active_days"]
)

# 2. Monthly coverage
print("2/4: Checking monthly coverage...", flush=True)

monthly = (
    df.with_columns(
        pl.col("date").dt.strftime("%Y-%m").alias("month")
    )
    .group_by("month")
    .agg(
        pl.len().alias("rows"),
        pl.col("date").n_unique().alias("active_days"),
        pl.col("city_id").n_unique().alias("cities")
    )
    .sort("month")
    .collect(engine="streaming")
)

# 3. Coverage by city
print("3/4: Checking coverage by city...", flush=True)

cities = (
    df.group_by("city_id")
    .agg(
        pl.len().alias("rows"),
        pl.col("date").min().alias("min_date"),
        pl.col("date").max().alias("max_date"),
        pl.col("date").n_unique().alias("active_days"),
        pl.col("store_id").n_unique().alias("stores"),
        pl.col("product_id").n_unique().alias("products")
    )
    .sort("city_id")
    .collect(engine="streaming")
)

# 4. Coverage by store-product pair
print("4/4: Checking store-product time series...", flush=True)

series = (
    df.group_by(["store_id", "product_id"])
    .agg(
        pl.col("date").n_unique().alias("active_days"),
        pl.col("date").min().alias("first_date"),
        pl.col("date").max().alias("last_date")
    )
    .collect(engine="streaming")
    .with_columns(
        (
            (pl.col("last_date") - pl.col("first_date"))
            .dt.total_days() + 1
        ).alias("span_days")
    )
)

series_stats = series.select(
    pl.len().alias("total_series"),
    pl.col("active_days").min().alias("min_active_days"),
    pl.col("active_days").median().alias("median_active_days"),
    pl.col("active_days").max().alias("max_active_days"),
    pl.col("active_days").quantile(0.9).alias("p90_active_days"),
    (
        (pl.col("active_days") < pl.col("span_days"))
        .cast(pl.Float64)
        .mean()
    ).alias("fraction_series_with_gaps")
).to_dicts()[0]

# Save everything to one report
report = {
    "overview": overview,
    "monthly_coverage": monthly.to_dicts(),
    "city_coverage": cities.to_dicts(),
    "series_coverage": series_stats
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w", encoding="utf-8") as file:
    json.dump(
        report, file,
        indent=2,
        ensure_ascii=False,
        default=str
    )

print("\n========== RESULTS ==========")
print("Total rows:", overview["total_rows"])
print("Active days:", overview["active_days"])
print("Calendar days:", overview["calendar_days"])
print("Days without records:", overview["days_without_records"])
print("Number of cities:", cities.height)
print("Store-product series:", series_stats["total_series"])
print("Median active days:", series_stats["median_active_days"])
print("\nReport saved:", OUTPUT)
print("STEP 2.1 COMPLETED")
