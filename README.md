# Fashion Demand Forecasting System

A demand forecasting pipeline for a fashion retailer: predicts daily unit
demand per product category from historical sales, pricing, discounts, and
seasonality, to support short-term inventory planning.

## Data

[Global Fashion Retail Stores Dataset](https://www.kaggle.com/datasets/ricgomes/global-fashion-retail-stores-dataset)
(Kaggle) — 6.4M transaction line items across 35 stores in 7 countries,
2023-01-01 to 2025-03-18, with product attributes, per-line discounts, and a
recurring promotional calendar (`discounts.csv`).

Raw per-SKU-per-day demand is extremely sparse/zero-inflated (17,940 products
over ~800 days), so demand is aggregated to the **(category, sub-category,
day)** grain — 26 product groups × ~800 days — which is the standard
granularity for retail demand planning and avoids fitting noise instead of
signal.

## Pipeline

```
data/raw/*.csv → src/data_prep.py → daily demand panel (continuous, no gaps)
                → src/features.py → calendar + lag + rolling features
                → src/train.py    → XGBoost + PyTorch MLP, time-aware split
```

Run everything with:

```bash
pip install -r requirements.txt
kaggle datasets download -d ricgomes/global-fashion-retail-stores-dataset -p data/raw --unzip
python run_pipeline.py
```

### Feature engineering (`src/features.py`)

- **Calendar**: day of week, day of month/year, week/month/quarter, weekend
  and month-start/end flags.
- **Lag features**: demand 1, 7, 14, and 28 days prior, per category group.
- **Rolling statistics**: 7/14/28-day rolling mean and std, shifted by one day
  so the window never includes the day being predicted.
- **Pricing & discounts**: average unit price, realized discount, and the
  *planned* discount rate from the promo calendar (known in advance, so it's
  a legitimate forward-looking feature rather than leakage).

### Time-aware evaluation (`src/split.py`)

No random shuffling — the last 60 days are held out as a test set, the 60
days before that as validation, and the model is trained only on the past.
This simulates how the model would actually be used and avoids the
overly-optimistic accuracy you'd get from a random split on autocorrelated
time series data.

### Models (`src/train.py`, `src/models.py`)

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| XGBoost (gradient-boosted trees) | 19.3 | 32.9 | 10.1% |
| PyTorch MLP (embeddings + dense layers) | 22.9 | 54.7 | 10.8% |

Both models share the same engineered feature set (categorical `category`/
`sub_category` + the numeric features above), so the comparison isolates
model family rather than feature quality. XGBoost edges out the MLP, which
is the expected result for structured/tabular tabular data — deep nets tend
to win when there's abundant data or raw sequential/image input to learn
representations from, neither of which applies here.

See `reports/actual_vs_predicted.png` (test-period forecast vs. actual daily
demand) and `reports/feature_importance.png` (XGBoost gain — discounts,
weekend effects, and promo-calendar days dominate, followed by short-horizon
lag features).

## Project layout

```
src/
  config.py     paths, split windows, lag/rolling window sizes
  data_prep.py  raw CSVs -> continuous daily (category, sub_category) panel
  features.py   calendar/lag/rolling feature engineering
  split.py      time-aware train/val/test split
  models.py     PyTorch DemandMLP definition
  train.py      trains + evaluates XGBoost and the MLP, saves artifacts
  evaluate.py   MAE/RMSE/MAPE + plotting helpers
data/           raw/ (gitignored) and processed/ (gitignored) parquet files
models/         saved model weights + metrics.json (gitignored)
reports/        prediction and feature-importance plots
run_pipeline.py single command to run the whole pipeline
```

## Possible extensions

- Per-store or per-SKU forecasts (would need a hierarchical/global model to
  handle the sparser series).
- A sequence model (LSTM/temporal conv net) over raw daily windows instead of
  hand-built lag features.
- Quantile regression for inventory safety-stock planning instead of point
  forecasts.
