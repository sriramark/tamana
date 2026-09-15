import pandas as pd
from statsmodels.tsa.stattools import adfuller

def adf_test(series: pd.Series, alpha: float = 0.05) -> dict:
    res = adfuller(series.dropna(), autolag="AIC")

    stat = float(res[0])
    p_val = float(res[1])
    lags = int(res[2])
    n_obs = int(res[3])
    crit = {k: round(v, 4) for k, v in res[4].items()}

    stationary = p_val < alpha

    if stationary:
        concl = f"p-value ({p_val:.4f}) < {alpha} -> Series is stationary."
        interp = "Series has constant mean and variance. No differencing required."
    else:
        concl = f"p-value ({p_val:.4f}) >= {alpha} -> Series is non-stationary."
        interp = "Series exhibits a trend. Differencing (d >= 1) is required."

    return {
        "test_statistic": round(stat, 4),
        "p_value": round(p_val, 6),
        "num_lags": lags,
        "num_obs": n_obs,
        "critical_values": crit,
        "is_stationary": stationary,
        "conclusion": concl,
        "interpretation": interp,
        "significance": alpha,
    }

def rolling_stats(series: pd.Series, window: int = 12) -> dict:
    r_mean = series.rolling(window=window, min_periods=1).mean()
    r_std = series.rolling(window=window, min_periods=1).std()

    return {
        "rolling_mean": r_mean,
        "rolling_std": r_std,
        "window": window,
    }

def difference_series(series: pd.Series, order: int = 1) -> pd.Series:
    diff_s = series.copy()
    for _ in range(order):
        diff_s = diff_s.diff()
    return diff_s

def suggest_d(series: pd.Series, max_d: int = 2) -> dict:
    steps = []
    curr = series.copy()

    for d in range(max_d + 1):
        res_adf = adf_test(curr.dropna())
        steps.append({"d": d, "adf": res_adf})
        if res_adf["is_stationary"]:
            return {"suggested_d": d, "steps": steps}
        curr = curr.diff()

    return {"suggested_d": max_d, "steps": steps}
