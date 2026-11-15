r"""STEP 4.1 - Prepare compact, Power BI-ready facts from FreshRetailNet-LT.

Run from project root:
    .\.venv\Scripts\python.exe scripts\10_prepare_powerbi_data.py

This script never retrains or re-scores the model. It keeps historical
EDA and official evaluation data in separate tables to avoid mixing targets.
"""
from pathlib import Path
import json
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / 'data' / 'raw' / 'train.parquet'
EVAL_PRED = ROOT / 'reports' / '09_official_eval_predictions.parquet'
OUT_DIR = ROOT / 'reports' / 'powerbi'
CHECK = ROOT / 'reports' / '10_powerbi_export_check.json'

for path in [TRAIN, EVAL_PRED]:
    if not path.exists():
        raise FileNotFoundError(f'Input file not found: {path}')
OUT_DIR.mkdir(parents=True, exist_ok=True)
print('STEP 4.1 - PREPARE POWER BI DATA', flush=True)

# 1. Historical EDA: one row per date x city x first-level category.
# This is NOT revenue. sale_amount is normalized sales quantity.
print('1/3: Aggregating historical sales...', flush=True)
historical = (
    pl.scan_parquet(TRAIN)
    .select([
        'dt', 'city_id', 'first_category_id',
        'sale_amount', 'stock_hour6_22_cnt', 'discount',
    ])
    .with_columns(
        pl.col('dt').str.strptime(pl.Date, '%Y-%m-%d').alias('date')
    )
    .with_columns(
        (pl.col('date').dt.weekday() >= 6).cast(pl.Int8).alias('is_weekend')
    )
    .group_by(['date', 'city_id', 'first_category_id', 'is_weekend'])
    .agg([
        pl.len().alias('n_records'),
        pl.col('sale_amount').sum().alias('total_normalized_sales'),
        (pl.col('stock_hour6_22_cnt') > 0).sum().alias('stockout_records'),
        (pl.col('discount') < 1.0).sum().alias('discounted_records'),
    ])
    .collect(engine='streaming')
    .sort(['date', 'city_id', 'first_category_id'])
)
if historical['n_records'].sum() != 7_869_549:
    raise ValueError('Historical aggregation row count is not 7,869,549')
history_path = OUT_DIR / 'fact_sales_daily.csv'
historical.write_csv(history_path)

# 2. Official evaluation: aggregate previously FROZEN model predictions.
# Keep this in a separate fact table, never blend it into historical sales.
print('2/3: Aggregating frozen evaluation predictions...', flush=True)
evaluated = (
    pl.scan_parquet(EVAL_PRED)
    .with_columns([
        (pl.col('date').dt.weekday() >= 6).cast(pl.Int8).alias('is_weekend'),
        (pl.col('actual') - pl.col('ml_prediction')).abs().alias('ml_abs_error'),
        (pl.col('actual') - pl.col('mean28')).abs().alias('baseline_abs_error'),
        pl.col('actual').abs().alias('abs_actual'),
    ])
    .group_by(['date', 'city_id', 'first_category_id', 'is_weekend'])
    .agg([
        pl.len().alias('n_eval_rows'),
        pl.col('actual').sum().alias('total_actual'),
        pl.col('ml_prediction').sum().alias('total_ml_prediction'),
        pl.col('mean28').sum().alias('total_baseline_prediction'),
        pl.col('abs_actual').sum().alias('abs_actual_sum'),
        pl.col('ml_abs_error').sum().alias('ml_abs_error_sum'),
        pl.col('baseline_abs_error').sum().alias('baseline_abs_error_sum'),
        (pl.col('stockout_hours') > 0).sum().alias('stockout_rows'),
    ])
    .collect(engine='streaming')
    .sort(['date', 'city_id', 'first_category_id'])
)
if evaluated['n_eval_rows'].sum() != 70_000:
    raise ValueError('Evaluation aggregation row count is not 70,000')
eval_path = OUT_DIR / 'fact_eval_daily.csv'
evaluated.write_csv(eval_path)

# 3. Recalculate WAPE from the exported table, checking reproducibility.
print('3/3: Validating exported files...', flush=True)
denominator = float(evaluated['abs_actual_sum'].sum())
if denominator <= 0:
    raise ValueError('WAPE denominator is zero')
ml_wape = 100 * float(evaluated['ml_abs_error_sum'].sum()) / denominator
base_wape = 100 * float(evaluated['baseline_abs_error_sum'].sum()) / denominator
if abs(ml_wape - 29.21) > 0.15 or abs(base_wape - 32.57) > 0.15:
    raise ValueError(
        f'WAPE mismatch: expected ML=29.21%, baseline=32.57%; '
        f'actual ML={ml_wape:.2f}%, baseline={base_wape:.2f}%'
    )

report = {
    'historical_export': str(history_path.relative_to(ROOT)),
    'historical_aggregate_rows': historical.height,
    'historical_original_rows': int(historical['n_records'].sum()),
    'historical_start': str(historical['date'].min()),
    'historical_end': str(historical['date'].max()),
    'historical_cities': historical['city_id'].n_unique(),
    'historical_categories': historical['first_category_id'].n_unique(),
    'eval_export': str(eval_path.relative_to(ROOT)),
    'eval_aggregate_rows': evaluated.height,
    'eval_original_rows': int(evaluated['n_eval_rows'].sum()),
    'eval_start': str(evaluated['date'].min()),
    'eval_end': str(evaluated['date'].max()),
    'eval_ml_wape_pct': round(ml_wape, 2),
    'eval_baseline_wape_pct': round(base_wape, 2),
    'notes': [
        'Sales are normalized quantities, not monetary revenue.',
        'Use SUM of error totals / SUM of absolute actuals for Power BI WAPE.',
        'Evaluation actuals are for reporting only, never training or tuning.',
    ],
}
with CHECK.open('w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
print(f'CHECK FILE: {CHECK}', flush=True)
print('STEP 4.1 COMPLETED', flush=True)
