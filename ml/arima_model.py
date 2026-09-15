import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf

warnings.filterwarnings("ignore")

def compute_acf_pacf(series: pd.Series, nlags: int = 20) -> dict:
    s_clean = series.dropna()
    n = len(s_clean)
    lags = min(nlags, n // 2 - 1)

    acf_vals = acf(s_clean, nlags=lags, fft=True)
    pacf_vals = pacf(s_clean, nlags=lags, method="ywm")
    ci = 1.96 / np.sqrt(n)

    return {
        "acf_values": acf_vals.tolist(),
        "pacf_values": pacf_vals.tolist(),
        "conf_int": round(ci, 4),
        "nlags": lags,
    }

def suggest_order(series: pd.Series, d: int = 1) -> dict:
    s_clean = series.dropna()
    n = len(s_clean)
    max_lag = min(20, n // 2 - 1)
    ci = 1.96 / np.sqrt(n)

    acf_vals = acf(s_clean, nlags=max_lag, fft=True)
    pacf_vals = pacf(s_clean, nlags=max_lag, method="ywm")

    sig_acf = [i for i, v in enumerate(acf_vals[1:], 1) if abs(v) > ci]
    sig_pacf = [i for i, v in enumerate(pacf_vals[1:], 1) if abs(v) > ci]

    q_val = max(sig_acf[:3]) if sig_acf else 1
    p_val = max(sig_pacf[:3]) if sig_pacf else 1

    p_val = min(p_val, 3)
    q_val = min(q_val, 3)

    exp_text = f"ACF lag cutoff: q = {q_val}. PACF lag cutoff: p = {p_val}. Order d = {d}."

    return {
        "p": p_val,
        "d": d,
        "q": q_val,
        "conf_int": round(ci, 4),
        "significant_acf": sig_acf,
        "significant_pacf": sig_pacf,
        "explanation": exp_text,
    }

def fit_arima(series: pd.Series, order: tuple) -> object:
    model = ARIMA(series.dropna(), order=order)
    res = model.fit()
    return res

def get_model_summary(res) -> dict:
    params = res.params.to_dict()
    residuals = res.resid.dropna()
    p_disp = {k: round(float(v), 6) for k, v in params.items()}

    return {
        "aic": round(float(res.aic), 4),
        "bic": round(float(res.bic), 4),
        "hqic": round(float(res.hqic), 4),
        "params": p_disp,
        "residuals": residuals,
        "model_order": res.model.order,
        "nobs": int(res.nobs),
        "sigma2": round(float(res.params.get("sigma2", 0)), 4),
        "summary_text": str(res.summary()),
    }

def forecast_arima(res, steps: int = 30) -> dict:
    fc_obj = res.get_forecast(steps=steps)
    mean_fc = fc_obj.predicted_mean
    conf_int = fc_obj.conf_int(alpha=0.05)

    return {
        "forecast": mean_fc.tolist(),
        "lower_ci": conf_int.iloc[:, 0].tolist(),
        "upper_ci": conf_int.iloc[:, 1].tolist(),
        "steps": steps,
        "forecast_index": list(range(1, steps + 1)),
    }
