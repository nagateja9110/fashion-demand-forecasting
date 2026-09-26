# Fashion Demand Forecasting System

A demand forecasting pipeline for a fashion retailer: predicts daily unit
demand per product category from historical sales, pricing, discounts, and
seasonality, to support short-term inventory planning.

**Live demo:** https://fashion-demand-forecasting.azurewebsites.net
(dashboard with actual-vs-predicted charts, feature importance, and an
interactive "forecast the next N days" tool backed by the trained XGBoost
model — deployed on Azure App Service, free tier, so the first request after
idle may take a few seconds to wake up).

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

## Live API + dashboard (`app/`, `static/`)

A FastAPI service (`app/main.py`) serves the trained XGBoost model plus a
static dashboard (`static/`) with the same result plots and an interactive
recursive forecasting tool: pick a category, a horizon, and optionally
override price/discount, and it walks forward day-by-day feeding each
prediction back in as pseudo-history for the next day's lag/rolling features
(`src/forecast.py`) — the same technique a production forecasting service
uses when future actuals aren't known yet.

Run it locally:

```bash
pip install -r app/requirements.txt
python run_pipeline.py   # produces models/forecast_context.json + reports/dashboard_data.json
uvicorn app.main:app --reload --port 8000
```

Deployed on **Azure App Service** (Linux, Python 3.11, free F1 tier):

```bash
az group create --name fashion-demand-forecasting-rg --location centralindia
az appservice plan create --name fashion-demand-plan --resource-group fashion-demand-forecasting-rg --sku F1 --is-linux
az webapp create --name fashion-demand-forecasting --resource-group fashion-demand-forecasting-rg --plan fashion-demand-plan --runtime "PYTHON:3.11"
az webapp config appsettings set --name fashion-demand-forecasting --resource-group fashion-demand-forecasting-rg --settings SCM_DO_BUILD_DURING_DEPLOYMENT=true
az webapp config set --name fashion-demand-forecasting --resource-group fashion-demand-forecasting-rg \
  --startup-file "gunicorn --bind=0.0.0.0:8000 --timeout 600 -k uvicorn.workers.UvicornWorker app.main:app"
az webapp deploy --name fashion-demand-forecasting --resource-group fashion-demand-forecasting-rg --src-path deploy.zip --type zip
```

The deployed package only needs `app/`, `static/`, a few `src/` modules
(`config.py`, `features.py`, `forecast.py`), and the small trained
artifacts (`models/xgboost_model.json`, `models/forecast_context.json`,
`reports/dashboard_data.json`) — training-only dependencies (PyTorch,
scikit-learn, matplotlib) are never installed on the server.

## Possible extensions

- Per-store or per-SKU forecasts (would need a hierarchical/global model to
  handle the sparser series).
- A sequence model (LSTM/temporal conv net) over raw daily windows instead of
  hand-built lag features.
- Quantile regression for inventory safety-stock planning instead of point
  forecasts.
