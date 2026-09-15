import numpy as np
import pandas as pd

def compute_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    y_true = np.array(actual, dtype=float)
    y_pred = np.array(predicted, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))

def compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    y_true = np.array(actual, dtype=float)
    y_pred = np.array(predicted, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))

def compute_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    y_true = np.array(actual, dtype=float)
    y_pred = np.array(predicted, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & (y_true != 0)
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def interpret_mape(val: float) -> str:
    if val < 10:
        return "Accurate Forecast (MAPE < 10%)"
    elif val < 20:
        return "Good Accuracy (10% <= MAPE < 20%)"
    elif val < 50:
        return "Reasonable Accuracy (20% <= MAPE < 50%)"
    else:
        return "Low Accuracy (MAPE >= 50%)"

def evaluate_model(res, series: pd.Series) -> dict:
    y_true = series.dropna().values
    y_fit = res.fittedvalues.dropna().values

    n = min(len(y_true), len(y_fit))
    y_true = y_true[-n:]
    y_fit = y_fit[-n:]

    mae_val = compute_mae(y_true, y_fit)
    rmse_val = compute_rmse(y_true, y_fit)
    mape_val = compute_mape(y_true, y_fit)

    return {
        "mae": round(mae_val, 4),
        "rmse": round(rmse_val, 4),
        "mape": round(mape_val, 4),
        "interpretation": interpret_mape(mape_val),
        "actual": y_true.tolist(),
        "fitted": y_fit.tolist(),
        "n_obs": n,
    }


def backtest_forecast(series: pd.Series, order: tuple, train_frac: float = 0.75) -> dict:
    """Out-of-sample accuracy check (the honest measure of forecast quality).

    In-sample metrics flatter the model, because the same observations were used
    to estimate the coefficients. Here the series is split chronologically: the
    model is re-fitted on the first ``train_frac`` of periods only, then asked to
    forecast the held-out tail it has never seen. MAE / RMSE / MAPE are computed
    against those unseen actuals.

    Returns None when the series is too short to split meaningfully.
    """
    from statsmodels.tsa.arima.model import ARIMA

    s = series.dropna()
    if len(s) < 10:
        return None

    split_idx = int(len(s) * train_frac)
    train, test = s.iloc[:split_idx], s.iloc[split_idx:]
    if len(test) < 2:
        return None

    fitted = ARIMA(train, order=order).fit()
    predicted = fitted.get_forecast(steps=len(test)).predicted_mean

    # Demand cannot be negative, so clip before scoring.
    y_pred = np.maximum(np.asarray(predicted, dtype=float), 0.0)
    y_true = test.values.astype(float)

    mae_val = compute_mae(y_true, y_pred)
    rmse_val = compute_rmse(y_true, y_pred)
    mape_val = compute_mape(y_true, y_pred)

    return {
        "mae": round(mae_val, 2),
        "rmse": round(rmse_val, 2),
        "mape": round(mape_val, 2),
        "interpretation": interpret_mape(mape_val),
        "train_size": len(train),
        "test_size": len(test),
        "actual": y_true.round(2).tolist(),
        "predicted": y_pred.round(2).tolist(),
        "periods": [d.strftime("%Y-%m") for d in test.index],
    }
