
from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "train.parquet"
OUTPUT = ROOT / "reports" / "05_stockout_promotion_summary.csv"

print("STEP 2.3 - STOCKOUT & PROMOTION ANALYSIS", flush=True)

# Read only the columns needed
df = (
    pl.scan_parquet(SOURCE)
    .select([
        "city_id",
        "first_category_id",
        "sale_amount",
        "stock_hour6_22_cnt",
        "discount",
        "activity_flag"
    ])
    .with_columns([
        (pl.col("stock_hour6_22_cnt") > 0)
        .alias("has_stockout"),

        (pl.col("discount") < 1.0)
        .alias("is_discounted")
    ])
)

print("1/3: Aggregating by city and category...", flush=True)

summary = (
    df.group_by(["city_id", "first_category_id"])
    .agg([
        pl.len().alias("n_records"),

        # Stockout indicators
        (pl.col("has_stockout").mean() * 100)
        .round(2).alias("stockout_pct"),

        pl.col("stock_hour6_22_cnt")
        .mean().round(2).alias("avg_stockout_hours"),

        # Promotion indicators
        (pl.col("is_discounted").mean() * 100)
        .round(2).alias("discounted_pct"),

        (pl.col("activity_flag").eq(1).mean() * 100)
        .round(2).alias("activity_pct"),

        # Average normalized sales
        pl.col("sale_amount")
        .mean().round(4).alias("avg_sales"),

        # Sales on stockout vs non-stockout days
        pl.col("sale_amount")
        .filter(pl.col("has_stockout"))
        .mean().round(4).alias("avg_sales_stockout"),

        pl.col("sale_amount")
        .filter(~pl.col("has_stockout"))
        .mean().round(4).alias("avg_sales_no_stockout"),

        # Sales with vs without a discount
        pl.col("sale_amount")
        .filter(pl.col("is_discounted"))
        .mean().round(4).alias("avg_sales_discounted"),

        pl.col("sale_amount")
        .filter(~pl.col("is_discounted"))
        .mean().round(4).alias("avg_sales_no_discount")
    ])
    .sort(["city_id", "first_category_id"])
    .collect(engine="streaming")
)

print("2/3: Saving results...", flush=True)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
summary.write_csv(OUTPUT)

print("3/3: Showing high-stockout groups...", flush=True)

top_stockout = (
    summary
    .filter(pl.col("n_records") >= 10000)
    .sort("stockout_pct", descending=True)
    .head(10)
)

print(top_stockout.select([
    "city_id",
    "first_category_id",
    "n_records",
    "stockout_pct",
    "discounted_pct",
    "avg_sales"
]))

print("\nTotal groups:", summary.height)
print("Total rows:", summary["n_records"].sum())
print("Saved:", OUTPUT)
print("STEP 2.3 COMPLETED")
