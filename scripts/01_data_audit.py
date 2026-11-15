"""Produce a reproducible low-memory initial audit of FreshRetailNet-LT.

Reads metadata with PyArrow and scalar columns with Polars lazy queries;
never materializes all 7.9m rows as a pandas DataFrame.
"""
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
REPORTS = ROOT / 'reports'
EXPECTED_ROWS = {'train': 7_869_549, 'eval': 70_000}
EXPECTED_COLUMNS = [
    'city_id', 'store_id', 'management_group_id', 'first_category_id',
    'second_category_id', 'third_category_id', 'product_id', 'dt',
    'sale_amount', 'hours_sale', 'stock_hour6_22_cnt',
    'hours_stock_status', 'activity_flag', 'discount', 'holiday_flag',
    'precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level',
]
EXCLUDED_NESTED = {'hours_sale', 'hours_stock_status'}
SAMPLE_COLUMNS = [
    'dt', 'city_id', 'store_id', 'product_id', 'sale_amount',
    'stock_hour6_22_cnt', 'discount', 'activity_flag', 'holiday_flag',
    'precpt', 'avg_temperature',
]


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect(lazy_frame):
    # Polars 1.25+ streaming query engine: limits peak memory for large scans.
    return lazy_frame.collect(engine='streaming')


def audit_split(name):
    path = RAW / (name + '.parquet')
    if not path.is_file():
        raise FileNotFoundError(f'Cannot find {path}')
    print(f'\n[{name.upper()}] Inspecting {path.name} ...', flush=True)
    pf = pq.ParquetFile(str(path))
    n_rows = int(pf.metadata.num_rows)
    cols = list(pf.schema_arrow.names)
    missing = [c for c in EXPECTED_COLUMNS if c not in cols]
    if missing:
        raise ValueError(f'Missing columns in {name}: {missing}')
    schema_rows = [{'split': name, 'name': field.name, 'dtype': str(field.type)}
                   for field in pf.schema_arrow]
    print(f'  Metadata: {n_rows:,} rows; {len(cols)} columns; '
          f'{pf.metadata.num_row_groups} row groups', flush=True)

    # Only a tiny batch of selected scalar columns is loaded here.
    batches = pf.iter_batches(batch_size=6, columns=[c for c in SAMPLE_COLUMNS if c in cols])
    first_batch = next(batches, None)
    samples = first_batch.to_pylist() if first_batch is not None else []
    write_csv(REPORTS / f'sample_{name}.csv', samples,
              [c for c in SAMPLE_COLUMNS if c in cols])

    lf = pl.scan_parquet(str(path))
    basic = collect(lf.select([
        pl.col('dt').min().alias('min_date'),
        pl.col('dt').max().alias('max_date'),
        pl.col('store_id').n_unique().alias('unique_stores'),
        pl.col('product_id').n_unique().alias('unique_products'),
        pl.col('city_id').n_unique().alias('unique_cities'),
        pl.col('sale_amount').is_null().sum().alias('missing_target'),
        pl.col('stock_hour6_22_cnt').gt(0).cast(pl.Float64).mean().alias('row_stockout_rate'),
    ])).to_dicts()[0]
    print(f'  Dates: {basic["min_date"]} ... {basic["max_date"]}', flush=True)
    print(f'  Stores: {basic["unique_stores"]:,}; '
          f'products: {basic["unique_products"]:,}; '
          f'cities: {basic["unique_cities"]:,}', flush=True)

    # Low-memory daily aggregation, useful for the EDA in the next step.
    daily = collect(lf.group_by('dt').agg([
        pl.len().alias('n_records'),
        pl.col('sale_amount').sum().alias('total_normalized_sales'),
        pl.col('stock_hour6_22_cnt').gt(0).cast(pl.Float64).mean().alias('stockout_rate'),
    ]).sort('dt'))
    daily.write_csv(str(REPORTS / f'daily_{name}.csv'))

    scalar_cols = [c for c in EXPECTED_COLUMNS if c not in EXCLUDED_NESTED]
    nulls = collect(lf.select([
        pl.col(c).null_count().alias(c) for c in scalar_cols
    ])).to_dicts()[0]
    missing_rows = [
        {'split': name, 'column': c, 'null_count': int(nulls[c]),
         'null_percent': round(100 * int(nulls[c]) / max(n_rows, 1), 4)}
        for c in scalar_cols
    ]
    write_csv(REPORTS / f'missing_{name}.csv', missing_rows,
              ['split', 'column', 'null_count', 'null_percent'])
    summary = {
        'file': path.name,
        'size_mib': round(path.stat().st_size / (1024 ** 2), 2),
        'rows': n_rows,
        'expected_rows': EXPECTED_ROWS[name],
        'rows_match_source': n_rows == EXPECTED_ROWS[name],
        'n_columns': len(cols),
        'n_row_groups': pf.metadata.num_row_groups,
        **basic,
    }
    print(f'  Saved audit files for {name}', flush=True)
    return summary, schema_rows


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    all_schema = []
    all_summary = {}
    for name in ('train', 'eval'):
        all_summary[name], schema = audit_split(name)
        all_schema.extend(schema)
    write_csv(REPORTS / 'schema.csv', all_schema, ['split', 'name', 'dtype'])
    train_max, eval_min = all_summary['train']['max_date'], all_summary['eval']['min_date']
    chronological = bool(str(train_max) < str(eval_min))
    all_summary['validation'] = {
        'train_max_strictly_before_eval_min': chronological,
        'expected_forecast_days': '2025-07-14 to 2025-07-20',
        'eval_date_range_matches_benchmark': (
            str(all_summary['eval']['min_date']) == '2025-07-14' and
            str(all_summary['eval']['max_date']) == '2025-07-20'
        ),
        'important': 'Do not use eval target or eval-day stockout labels as model inputs.',
    }
    (REPORTS / 'audit_summary.json').write_text(
        json.dumps(all_summary, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    report_lines = [
        '# Initial data audit - FreshRetailNet-LT', '',
        f'Generated: {datetime.now():%Y-%m-%d %H:%M}', '',
        '| Split | Rows | Columns | Date range | Stores | Products |',
        '|---|---:|---:|---|---:|---:|',
    ]
    for name in ('train', 'eval'):
        info = all_summary[name]
        report_lines.append(f'| {name} | {info["rows"]:,} | {info["n_columns"]} | '
                            f'{info["min_date"]} to {info["max_date"]} | '
                            f'{info["unique_stores"]:,} | {info["unique_products"]:,} |')
    report_lines += ['',
        '## Split validation',
        f'- Train strictly earlier than eval: **{chronological}**',
        f'- Eval period matches official 7-day window: '
        f'**{all_summary["validation"]["eval_date_range_matches_benchmark"]}**',
        '- No inference from eval target or eval-day actual stockout fields.',
        '', '## Output files',
        '- `schema.csv`: file schemas',
        '- `sample_train.csv`, `sample_eval.csv`: six rows for manual inspection',
        '- `missing_train.csv`, `missing_eval.csv`: scalar null profiles',
        '- `daily_train.csv`, `daily_eval.csv`: daily aggregates',
        '- `audit_summary.json`: machine-readable summary',
        '', 'NOTE: Actual sales are normalized, NOT monetary revenue.',
    ]
    (REPORTS / '01_initial_audit.md').write_text(
        '\n'.join(report_lines) + '\n', encoding='utf-8')
    print('\n' + '=' * 64)
    print('STEP 1 FINISHED. Files saved in:', REPORTS)
    print('Open: reports\\01_initial_audit.md')
    print('Open: reports\\audit_summary.json')
    print('=' * 64)
    if not chronological:
        print('WARNING: train dates overlap eval dates; investigate before modeling!')
    if not all_summary['validation']['eval_date_range_matches_benchmark']:
        print('WARNING: evaluation date range differs from official benchmark!')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'\nERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise
