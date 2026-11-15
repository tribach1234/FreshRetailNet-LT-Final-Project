
# FreshRetailNet-LT | Sales Analysis & 7-Day Forecasting

## Overview

This final project analyzes fresh retail sales data and develops a machine learning model to forecast normalized daily sales for the next 7 days.

The project combines:
- Exploratory Data Analysis (EDA)
- City, category, and weekend analysis
- Stockout and promotion analysis
- Regression and time-series forecasting
- Power BI dashboards
- An interactive Gradio application

## Dataset

**Source:** [Dingdong-Inc / FreshRetailNet-LT](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-LT)

| Property | Value |
|---|---|
| Training records | 7,869,549 |
| Evaluation records | 70,000 |
| Features in source data | 19 columns |
| Cities | 18 |
| Stores | 1,057 |
| Products | 576 |
| Training period | 2023-06-05 to 2025-07-13 |
| Evaluation period | 2025-07-14 to 2025-07-20 |
| Target | `sale_amount` |

The target represents normalized sales quantity, not monetary revenue or uncensored customer demand.

### Download the data

Download `train.parquet` and `eval.parquet` from the dataset page.

Place the files in:

```text
data/
  raw/
    train.parquet
    eval.parquet
```

The original datasets are excluded from this GitHub repository.

## Business Questions

1. How does sales behavior differ between weekdays and weekends?
2. Which product categories contribute most to each city's sales?
3. How are stockout events and promotions associated with sales?
4. Can machine learning improve 7-day sales forecasting?
5. How does forecasting performance differ across cities?

## Exploratory Data Analysis

Key findings:

- Weekend average sales were higher than weekday averages across all 18 cities.
- Product category composition varied across cities.
- Categories 3, 19, and 25 were among the prominent categories.
- Approximately 48.20% of training records indicated at least one stockout hour.
- Individual store-product time series may contain missing observation dates.

These are descriptive findings, not causal conclusions.

## Forecasting Methodology

**Task:** Supervised regression for 7-day normalized sales forecasting.

Models:
- Mean 28-Day Baseline
- Seasonal Naive Baseline
- LightGBM Regressor

The LightGBM model uses 13 engineered features derived from identifiers, calendar information, and historical sales.

Temporal separation is used to prevent target leakage.

### Evaluation periods

- Validation: July 7-13, 2025
- Final evaluation: July 14-20, 2025

The final evaluation period is not used for model training.

### Final Evaluation Results

| Metric | Mean28 Baseline | LightGBM |
|---|---:|---:|
| MAE | 0.5945 | 0.5332 |
| RMSE | 0.8899 | 0.8125 |
| WAPE | 32.57% | 29.21% |
| WAPE (No Stockout) | 28.51% | 25.25% |

LightGBM reduced overall WAPE by 3.36 percentage points compared with the Mean28 baseline.

WAPE is calculated as the sum of absolute prediction errors divided by the sum of absolute actual values.

## Project Structure

```text
FINAL_PROJECT_FRESHRETAIL/
├── data/
│   ├── raw/             # Downloaded Parquet files (not tracked)
│   └── processed/
├── scripts/             # Audit, EDA, forecasting, and Gradio code
├── models/
│   └── 07_lightgbm_model.txt
├── reports/             # Analysis and evaluation outputs
├── final_kit/           # Report, presentation, speaking script
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

Recommended environment: Python 3.11 on Windows.

Create a virtual environment:

```powershell
py -3.11 -m venv .venv
```

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the Project

After downloading the dataset, execute the project scripts from the repository root.

### Data audit

```powershell
.\.venv\Scripts\python.exe scripts\01_data_audit.py
```

### Baseline forecasting

```powershell
.\.venv\Scripts\python.exe scripts\06_baseline_forecasting.py
```

### Train LightGBM

```powershell
.\.venv\Scripts\python.exe scripts\07_train_lightgbm.py
```

### Official evaluation

```powershell
.\.venv\Scripts\python.exe scripts\09_official_eval.py
```

### Run the Gradio App

A trained LightGBM model is provided in `models/`.

```powershell
.\.venv\Scripts\python.exe scripts\11_gradio_app.py
```

Open the local URL displayed in the terminal, typically:

http://127.0.0.1:7860

The demo forecasts July 14-20, 2025, using historical information available through July 13, 2025.

It is a retrospective demonstration, not a live forecast for the current date.

## Deliverables

### Power BI Dashboard

Three pages:
1. Sales Overview
2. Product & City Analysis
3. Forecast Performance

A PDF export is provided in `reports/FreshRetailNet_Dashboard.pdf`.

### Gradio Application

The application allows users to select City, Store, and Product identifiers and display:
- Seven-day LightGBM forecasts
- Mean28 baseline forecasts
- Forecast tables and charts

### Final Project Documents

The `final_kit/` folder contains:
- Final Report (Word)
- Final Presentation (PowerPoint)
- Presentation Speaking Script (Word)

## Limitations

- Product and location identifiers are anonymized.
- Sales quantities are globally normalized.
- Observed sales may be affected by stockouts.
- The trained model is evaluated on a historical 2025 forecasting window.
- Model performance may vary across cities and products.

## Data Attribution

Dataset: FreshRetailNet-LT by Dingdong-Inc.

Source:
https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-LT

Dataset license: CC BY 4.0.

## Project Status

Final project completed:
EDA, Baseline Forecasting, LightGBM, Evaluation, Power BI, and Gradio.
