import json

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import config
from .evaluate import compute_metrics, plot_feature_importance, plot_predictions
from .models import DemandMLP
from .split import time_aware_split
from .utils import set_seed

NUMERIC_FEATURES = [
    "avg_unit_price",
    "avg_realized_discount",
    "planned_discount_pct",
    "is_promo_day",
    "day_of_week",
    "day_of_month",
    "day_of_year",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_14",
    "rolling_std_14",
    "rolling_mean_28",
    "rolling_std_28",
]
CATEGORICAL_FEATURES = ["category", "sub_category"]


def train_xgboost(train, val, test):
    features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    dtrain = xgb.DMatrix(train[features], label=train[config.TARGET_COL], enable_categorical=True)
    dval = xgb.DMatrix(val[features], label=val[config.TARGET_COL], enable_categorical=True)
    dtest = xgb.DMatrix(test[features], label=test[config.TARGET_COL], enable_categorical=True)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": config.RANDOM_SEED,
    }
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    test_pred = booster.predict(dtest)
    return booster, test_pred


def _encode_categoricals(df: pd.DataFrame, category_codes: dict) -> tuple[np.ndarray, np.ndarray]:
    category_idx = df["category"].map(category_codes[0]).to_numpy()
    subcategory_idx = df["sub_category"].map(category_codes[1]).to_numpy()
    return category_idx, subcategory_idx


def train_torch_mlp(train, val, test):
    set_seed(config.RANDOM_SEED)

    category_codes = (
        {c: i for i, c in enumerate(train["category"].cat.categories)},
        {c: i for i, c in enumerate(train["sub_category"].cat.categories)},
    )

    scaler = StandardScaler()
    train_numeric = scaler.fit_transform(train[NUMERIC_FEATURES])
    val_numeric = scaler.transform(val[NUMERIC_FEATURES])
    test_numeric = scaler.transform(test[NUMERIC_FEATURES])

    # Neural nets train far more stably when the regression target is
    # standardized too -- otherwise large-magnitude demand values dominate
    # the loss and gradients early on. Predictions are un-scaled before
    # evaluation so metrics stay in real units.
    target_scaler = StandardScaler()
    target_scaler.fit(train[[config.TARGET_COL]])

    def to_tensors(df, numeric):
        cat_idx, subcat_idx = _encode_categoricals(df, category_codes)
        y_scaled = target_scaler.transform(df[[config.TARGET_COL]]).ravel()
        return (
            torch.tensor(cat_idx, dtype=torch.long),
            torch.tensor(subcat_idx, dtype=torch.long),
            torch.tensor(numeric, dtype=torch.float32),
            torch.tensor(y_scaled, dtype=torch.float32),
        )

    train_cat, train_subcat, train_num, train_y = to_tensors(train, train_numeric)
    val_cat, val_subcat, val_num, val_y = to_tensors(val, val_numeric)
    test_cat, test_subcat, test_num, test_y = to_tensors(test, test_numeric)

    train_loader = DataLoader(
        TensorDataset(train_cat, train_subcat, train_num, train_y), batch_size=64, shuffle=True
    )

    model = DemandMLP(
        n_categories=len(category_codes[0]),
        n_subcategories=len(category_codes[1]),
        n_numeric=len(NUMERIC_FEATURES),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.L1Loss()  # MAE loss matches the reporting metric.

    best_val_loss = float("inf")
    best_state = None
    patience, patience_counter = 15, 0
    max_epochs = 200

    for epoch in range(max_epochs):
        model.train()
        for cat_b, subcat_b, num_b, y_b in train_loader:
            optimizer.zero_grad()
            pred = model(cat_b, subcat_b, num_b)
            loss = loss_fn(pred, y_b)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_cat, val_subcat, val_num)
            val_loss = loss_fn(val_pred, val_y).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred_scaled = model(test_cat, test_subcat, test_num).numpy()
    test_pred = target_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).ravel()

    return model, scaler, target_scaler, category_codes, test_pred


def main() -> None:
    set_seed(config.RANDOM_SEED)
    df = pd.read_parquet(config.FEATURES_PARQUET)
    train, val, test = time_aware_split(df)
    print(f"train={len(train)} val={len(val)} test={len(test)} rows")

    xgb_model, xgb_test_pred = train_xgboost(train, val, test)
    torch_model, scaler, target_scaler, category_codes, torch_test_pred = train_torch_mlp(train, val, test)

    test = test.copy()
    test["pred_xgboost"] = xgb_test_pred
    test["pred_mlp"] = torch_test_pred

    metrics = {
        "XGBoost": compute_metrics(test[config.TARGET_COL], test["pred_xgboost"]),
        "PyTorch MLP": compute_metrics(test[config.TARGET_COL], test["pred_mlp"]),
    }
    print(json.dumps(metrics, indent=2))

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    xgb_model.save_model(config.MODELS_DIR / "xgboost_model.json")
    torch.save(torch_model.state_dict(), config.MODELS_DIR / "mlp_model.pt")
    joblib.dump(scaler, config.MODELS_DIR / "mlp_scaler.pkl")
    joblib.dump(target_scaler, config.MODELS_DIR / "mlp_target_scaler.pkl")
    joblib.dump(category_codes, config.MODELS_DIR / "category_codes.pkl")

    with open(config.MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_predictions(
        test,
        y_true_col=config.TARGET_COL,
        pred_cols={"XGBoost": "pred_xgboost", "PyTorch MLP": "pred_mlp"},
        save_path=config.REPORTS_DIR / "actual_vs_predicted.png",
    )
    plot_feature_importance(
        xgb_model.get_score(importance_type="gain"),
        save_path=config.REPORTS_DIR / "feature_importance.png",
    )
    print(f"Saved models to {config.MODELS_DIR}, plots to {config.REPORTS_DIR}")


if __name__ == "__main__":
    main()
