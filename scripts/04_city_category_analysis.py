
from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "data" / "raw" / "train.parquet"
OUTPUT = ROOT / "reports" / "04_city_category_summary.csv"

print("STEP 2.2B - CITY & CATEGORY ANALYSIS", flush=True)

# 1. Read only required columns
df = (
    pl.scan_parquet(SOURCE)
    .select([
        "city_id",
        "store_id",
        "product_id",
        "first_category_id",
        "dt",
        "sale_amount",
        "stock_hour6_22_cnt"
    ])
)

print("1/3: Aggregating category sales...", flush=True)

# 2. Aggregate by city and product category
summary = (
    df.group_by(["city_id", "first_category_id"])
    .agg([
        pl.len().alias("n_records"),

        pl.col("sale_amount")
        .sum()
        .alias("total_sales"),

        pl.col("sale_amount")
        .mean()
        .alias("avg_sales_per_record"),

        pl.col("store_id")
        .n_unique()
        .alias("n_stores"),

        pl.col("product_id")
        .n_unique()
        .alias("n_products"),

        pl.col("dt")
        .n_unique()
        .alias("active_days"),

        (
            (pl.col("stock_hour6_22_cnt") > 0)
            .cast(pl.Float64)
            .mean()
        ).alias("stockout_rate")
    ])
    .with_columns(
        (
            100 * pl.col("total_sales") /
            pl.col("total_sales").sum().over("city_id")
        )
        .round(2)
        .alias("city_sales_share_pct")
    )
    .sort(
        ["city_id", "city_sales_share_pct"],
        descending=[False, True]
    )
    .collect(engine="streaming")
)

# 3. Write CSV
print("2/3: Saving results...", flush=True)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
summary.write_csv(OUTPUT)

# 4. Show top 3 categories in each city
print("3/3: Showing top categories...", flush=True)

top3 = (
    summary.group_by("city_id", maintain_order=True)
    .head(3)
)

print(top3.select([
    "city_id",
    "first_category_id",
    "city_sales_share_pct",
    "n_products",
    "n_stores",
    "avg_sales_per_record"
]))

print("\nTotal city-category groups:", summary.height)
print("Total cities:", summary["city_id"].n_unique())
print("Saved:", OUTPUT)
print("STEP 2.2B COMPLETED")
