import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .utils import mean_absolute_percentage_error


def compute_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
    }


def plot_predictions(test_df, y_true_col: str, pred_cols: dict, save_path) -> None:
    """Plot actual vs. predicted total daily demand (summed across all
    product categories) over the held-out test window."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    daily_actual = test_df.groupby("date")[y_true_col].sum()

    plt.figure(figsize=(11, 5))
    plt.plot(daily_actual.index, daily_actual.values, label="Actual", color="black", linewidth=2)
    for name, pred_col in pred_cols.items():
        daily_pred = test_df.groupby("date")[pred_col].sum()
        plt.plot(daily_pred.index, daily_pred.values, label=name, linestyle="--")

    plt.title("Total Daily Demand: Actual vs. Predicted (Test Period)")
    plt.xlabel("Date")
    plt.ylabel("Units Sold (all categories)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_feature_importance(importance: dict, save_path, top_n: int = 15) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    names, values = zip(*reversed(top))

    plt.figure(figsize=(8, 6))
    plt.barh(names, values, color="#4C72B0")
    plt.title("XGBoost Feature Importance (gain)")
    plt.xlabel("Gain")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
