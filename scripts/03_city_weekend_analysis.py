
from pathlib import Path
import polars as pl

# Paths
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "train.parquet"
OUTPUT = ROOT / "reports" / "03_city_weekend_summary.csv"

print("STEP 2.2A - CITY & WEEKEND ANALYSIS", flush=True)

# 1. Read only necessary columns
df = (
    pl.scan_parquet(SOURCE)
    .select([
        "dt",
        "city_id",
        "store_id",
        "product_id",
        "sale_amount"
    ])
    .with_columns(
        pl.col("dt").cast(pl.Date).alias("date")
    )
    .with_columns(
        (pl.col("date").dt.weekday() >= 6)
        .alias("is_weekend")
    )
)

print("1/3: Aggregating by store and product...", flush=True)

# 2. Compute weekday/weekend means for each store-product
pairs = (
    df.group_by([
        "city_id",
        "store_id",
        "product_id",
        "is_weekend"
    ])
    .agg([
        pl.col("sale_amount").mean().alias("avg_sales"),
        pl.col("date").n_unique().alias("n_days")
    ])
    .collect(engine="streaming")
)

keys = ["city_id", "store_id", "product_id"]

weekday = (
    pairs.filter(~pl.col("is_weekend"))
    .select(
        keys + [
            pl.col("avg_sales").alias("weekday_avg"),
            pl.col("n_days").alias("weekday_days")
        ]
    )
)

weekend = (
    pairs.filter(pl.col("is_weekend"))
    .select(
        keys + [
            pl.col("avg_sales").alias("weekend_avg"),
            pl.col("n_days").alias("weekend_days")
        ]
    )
)

# 3. Compare the same store-product pairs
matched = (
    weekday.join(weekend, on=keys, how="inner")
    .filter(
        (pl.col("weekday_days") >= 20) &
        (pl.col("weekend_days") >= 8)
    )
)

print("2/3: Computing city-level comparisons...", flush=True)

# Count all store-product pairs for coverage checking
all_pairs = (
    pairs.select(keys).unique()
    .group_by("city_id")
    .agg(pl.len().alias("total_pairs"))
)

city_results = (
    matched.group_by("city_id")
    .agg([
        pl.len().alias("matched_pairs"),
        pl.col("weekday_avg").mean()
          .alias("weekday_avg_sales"),
        pl.col("weekend_avg").mean()
          .alias("weekend_avg_sales"),
        (
            pl.col("weekend_avg") -
            pl.col("weekday_avg")
        ).mean().alias("mean_difference")
    ])
)

summary = (
    all_pairs.join(city_results, on="city_id", how="left")
    .with_columns([
        (
            100 * pl.col("matched_pairs") /
            pl.col("total_pairs")
        ).round(2).alias("pair_coverage_pct"),

        pl.when(pl.col("weekday_avg_sales") > 0)
        .then(
            100 * (
                pl.col("weekend_avg_sales") /
                pl.col("weekday_avg_sales") - 1
            )
        )
        .otherwise(None)
        .round(2)
        .alias("weekend_change_pct")
    ])
    .sort("city_id")
)

print("3/3: Saving results...", flush=True)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
summary.write_csv(OUTPUT)

print("\n========== RESULTS ==========")
print(summary)
print("\nTotal cities:", summary.height)
print("Total matched pairs:",
      matched.height)
print("Saved:", OUTPUT)
print("STEP 2.2A COMPLETED")
