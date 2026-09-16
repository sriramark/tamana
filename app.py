"""
Smart Inventory Management and Demand Prediction Using ARIMA
============================================================

Flask web application implementing the project pipeline:

    Dataset -> Preprocessing -> ARIMA Model -> Demand Forecast ->
    Inventory Management -> Low Stock Alert -> Reorder Recommendation ->
    Accuracy Evaluation (MAE / RMSE / MAPE) -> Dashboard

Run with:
    .venv\\Scripts\\python.exe app.py      then open http://127.0.0.1:5000

Route map
---------
/                 project overview
/dataset          raw data, schema, missing values, preview
/analysis         demand over time + descriptive statistics + ADF test
/differencing     making the series stationary (choosing d)
/acf-pacf         reading ACF/PACF to choose p and q
/arima            fit the model, coefficients, AIC/BIC, residual diagnostics
/forecast         6-period ahead demand forecast with 95% confidence interval
/evaluation       forecast accuracy: MAE, RMSE, MAPE
/inventory        safety stock, reorder point, low-stock alert, EOQ, recommendation
/case-study       festive demand-spike scenario
/mathematics      formula reference
/math-derivation  step-by-step derivations
/about            methodology and tech stack
/conclusion       findings
"""

import os
import secrets
import traceback

import numpy as np
import pandas as pd
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash
)

from ml.preprocess import (
    download_kaggle_dataset, load_blinkit_merged_df,
    get_dataset_metadata, build_time_series, get_statistics, get_preview
)
from ml.stationarity import adf_test, difference_series, suggest_d
from ml.arima_model import (
    fit_arima, get_model_summary, forecast_arima, suggest_order, compute_acf_pacf
)
from ml.evaluation import evaluate_model, backtest_forecast
from ml.inventory import (
    compute_eoq, compute_safety_stock, compute_reorder_point, generate_recommendation
)
from ml.visualization import (
    plot_time_series, plot_differenced, plot_acf_pacf, plot_forecast,
    plot_residuals, plot_evaluation, plot_inventory_chart
)

app = Flask(__name__)

# Sessions only hold the selected category and ARIMA order, but the key still
# must not be a published constant in production: anyone could forge a cookie.
# Render provides SECRET_KEY; locally we fall back to a per-process random key,
# which simply means sessions reset when the dev server restarts.
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, "dataset")

os.makedirs(dataset_dir, exist_ok=True)

# How many periods ahead every page forecasts. Kept in one place so the
# forecast page, the evaluation page and the inventory page always agree.
FORECAST_STEPS = 6

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
# Parsing + merging the three CSVs takes ~1s, so the merged frame is cached in
# module scope and reused across requests. _load_error keeps the real exception
# text so the UI can show *why* loading failed instead of a blank page.
_data_cache = None
_load_error = None


def get_master_df():
    """Return the merged Blinkit dataframe, or None when it cannot be loaded."""
    global _data_cache, _load_error

    if _data_cache is None:
        try:
            path = download_kaggle_dataset(dataset_dir)
            _data_cache = load_blinkit_merged_df(path)
            _load_error = None
        except Exception as err:
            _load_error = str(err)
            app.logger.error("Dataset load failed: %s", traceback.format_exc())
            _data_cache = None
    return _data_cache


def require_data():
    """Fetch the dataset or redirect to /dataset with an explanatory message.

    Returns (dataframe, None) on success, or (None, redirect_response)."""
    df = get_master_df()
    if df is None or df.empty:
        flash(_load_error or "Dataset unavailable.", "danger")
        return None, redirect(url_for("dataset"))
    return df, None


def current_series(df):
    """Monthly demand series for whichever category the user selected."""
    cat = session.get("selected_category", "All Categories")
    ts_df = build_time_series(df, category=cat, freq="ME")
    return ts_df, cat


def current_order(series):
    """The (p, d, q) the user has chosen, defaulting to a data-driven suggestion.

    d comes from repeated ADF testing (suggest_d); p and q come from reading the
    ACF/PACF of the differenced series (suggest_order).
    """
    if all(k in session for k in ("arima_p", "arima_d", "arima_q")):
        return (session["arima_p"], session["arima_d"], session["arima_q"])

    d = suggest_d(series)["suggested_d"]
    working = difference_series(series, d) if d > 0 else series
    sugg = suggest_order(working, d=d)
    return (sugg["p"], d, sugg["q"])


# Fitted models are cached so that navigating between /arima, /forecast,
# /evaluation and /inventory does not refit the same model four times.
_model_cache = {}


def get_fit(series, order):
    """Fit ARIMA(order) on *series*, reusing a cached fit when possible."""
    key = (session.get("selected_category", "All Categories"),
           order, len(series), float(series.sum()))
    if key not in _model_cache:
        _model_cache[key] = fit_arima(series, order)
        if len(_model_cache) > 24:          # keep the cache small
            _model_cache.pop(next(iter(_model_cache)))
    return _model_cache[key]


@app.context_processor
def inject_context():
    """Make the active selection available to every template.

    The category and ARIMA order are chosen on one page but affect all the
    others, so each pipeline page shows which selection it is working with."""
    return {
        "current_category": session.get("selected_category", "All Categories"),
        "current_order_text": "({}, {}, {})".format(
            session.get("arima_p", "-"), session.get("arima_d", "-"), session.get("arima_q", "-")
        ) if "arima_p" in session else None,
    }


def future_index(series, steps):
    """Dates for the next *steps* periods after the end of *series*."""
    last = series.index[-1]
    step = series.index[1] - series.index[0] if len(series) > 1 else pd.Timedelta(days=30)
    if step > pd.Timedelta(days=20):
        return pd.date_range(start=last + pd.offsets.MonthEnd(1), periods=steps, freq="ME")
    return pd.date_range(start=last + step, periods=steps, freq="W")


# ---------------------------------------------------------------------------
# Overview pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Project overview.

    The figures and the chart on this page are computed from the real dataset
    and the real fitted model, so the landing page is an honest summary of the
    system rather than decorative placeholder numbers."""
    summary = None
    df = get_master_df()

    if df is not None and not df.empty:
        try:
            meta = get_dataset_metadata(df)
            ts_df = build_time_series(df, freq="ME")
            series = ts_df["Demand"]

            d = suggest_d(series)["suggested_d"]
            working = difference_series(series, d).dropna() if d > 0 else series
            sugg = suggest_order(working, d=d)
            order = (sugg["p"], d, sugg["q"])

            res = get_fit(series, order)
            fc = forecast_arima(res, steps=FORECAST_STEPS)
            accuracy = backtest_forecast(series, order)

            summary = {
                "transactions": meta["total_rows"],
                "categories": meta["num_categories"],
                "products": meta["num_products"],
                "periods": len(series),
                "date_from": meta["min_date"],
                "date_to": meta["max_date"],
                "order": f"({order[0]}, {order[1]}, {order[2]})",
                "accuracy": accuracy,
                "plot_hero": plot_forecast(
                    series, fc["forecast"], fc["lower_ci"], fc["upper_ci"]
                ),
            }
        except Exception as err:
            app.logger.warning("Home summary unavailable: %s", err)

    return render_template("index.html", active="home", summary=summary)


@app.route("/about")
def about():
    return render_template("about.html", active="about")


# ---------------------------------------------------------------------------
# Stage 1 - Dataset
# ---------------------------------------------------------------------------
@app.route("/dataset")
def dataset():
    df = get_master_df()
    if df is None or df.empty:
        return render_template(
            "dataset.html", active="dataset", metadata={}, preview=[],
            stats=None, error=_load_error or "Dataset unavailable",
        )

    meta = get_dataset_metadata(df)
    preview = get_preview(df, 15)
    ts = build_time_series(df, freq="ME")
    stats = get_statistics(ts)

    return render_template(
        "dataset.html", active="dataset",
        metadata=meta, preview=preview, stats=stats, error=None,
    )


# ---------------------------------------------------------------------------
# Stage 2 - Demand analysis and stationarity
# ---------------------------------------------------------------------------
@app.route("/analysis", methods=["GET", "POST"])
def analysis():
    df, bail = require_data()
    if bail:
        return bail

    meta = get_dataset_metadata(df)
    categories = ["All Categories"] + meta["categories"]

    selected_cat = request.form.get("category", session.get("selected_category", "All Categories"))
    if selected_cat not in categories:
        selected_cat = "All Categories"

    try:
        ts_df = build_time_series(df, category=selected_cat, freq="ME")
    except Exception as err:
        flash(f"Error filtering dataset: {err}", "warning")
        selected_cat = "All Categories"
        ts_df = build_time_series(df, freq="ME")

    # Changing category invalidates the chosen ARIMA order and cached fits.
    if selected_cat != session.get("selected_category"):
        for k in ("arima_p", "arima_d", "arima_q"):
            session.pop(k, None)
        _model_cache.clear()
    session["selected_category"] = selected_cat

    stats = get_statistics(ts_df)
    plot_ts = plot_time_series(ts_df)
    res_adf = adf_test(ts_df["Demand"])

    ts_table = [
        {"date": dt.strftime("%Y-%m"), "demand": int(row["Demand"])}
        for dt, row in ts_df.iterrows()
    ]

    return render_template(
        "analysis.html", active="analysis",
        categories=categories, selected_category=selected_cat,
        stats=stats, plot_ts=plot_ts, adf=res_adf, ts_table=ts_table,
    )


@app.route("/differencing")
def differencing():
    """Stage 2b: how many differences (d) are needed to reach stationarity."""
    df, bail = require_data()
    if bail:
        return bail

    ts_df, _ = current_series(df)
    series = ts_df["Demand"]

    diff_info = suggest_d(series)          # runs ADF at d = 0, 1, 2 ...
    d = diff_info["suggested_d"]

    adf_before = adf_test(series)
    working = difference_series(series, d) if d > 0 else series
    adf_after = adf_test(working)

    plot_diff = plot_differenced(series, difference_series(series, max(d, 1)))

    return render_template(
        "differencing.html", active="differencing",
        d=d, diff_info=diff_info,
        adf_before=adf_before, adf_after=adf_after, plot_diff=plot_diff,
    )


@app.route("/acf-pacf")
def acf_pacf():
    """Stage 2c: read ACF/PACF of the differenced series to pick p and q."""
    df, bail = require_data()
    if bail:
        return bail

    ts_df, _ = current_series(df)
    series = ts_df["Demand"]

    d = suggest_d(series)["suggested_d"]
    working = difference_series(series, d).dropna() if d > 0 else series

    order_info = suggest_order(working, d=d)
    corr_info = compute_acf_pacf(working)
    plot_ap = plot_acf_pacf(working)

    return render_template(
        "acf_pacf.html", active="acf_pacf",
        order_info=order_info, corr=corr_info, plot_ap=plot_ap,
    )


# ---------------------------------------------------------------------------
# Stage 3 - ARIMA model
# ---------------------------------------------------------------------------
@app.route("/arima", methods=["GET", "POST"])
def arima():
    """Fit ARIMA(p,d,q) and show coefficients, information criteria and
    residual diagnostics."""
    df, bail = require_data()
    if bail:
        return bail

    ts_df, _ = current_series(df)
    series = ts_df["Demand"]

    p, d, q = current_order(series)
    if request.method == "POST":
        try:
            p = max(0, min(5, int(request.form.get("p", p))))
            d = max(0, min(2, int(request.form.get("d", d))))
            q = max(0, min(5, int(request.form.get("q", q))))
        except (TypeError, ValueError):
            flash("ARIMA order must be whole numbers.", "warning")
    session["arima_p"], session["arima_d"], session["arima_q"] = p, d, q

    model_info = None
    plot_res = None
    error = None
    try:
        res = get_fit(series, (p, d, q))
        model_info = get_model_summary(res)
        plot_res = plot_residuals(model_info["residuals"])
    except Exception as err:
        error = f"Could not fit ARIMA({p},{d},{q}): {err}"

    return render_template(
        "arima.html", active="arima",
        p=p, d=d, q=q, model_info=model_info, plot_res=plot_res, error=error,
    )


# ---------------------------------------------------------------------------
# Stage 4 - Demand forecast
# ---------------------------------------------------------------------------
@app.route("/forecast", methods=["GET", "POST"])
def forecast():
    df, bail = require_data()
    if bail:
        return bail

    ts_df, cat = current_series(df)
    series = ts_df["Demand"]

    p, d, q = current_order(series)
    if request.method == "POST":
        try:
            p = max(0, min(5, int(request.form.get("p", p))))
            d = max(0, min(2, int(request.form.get("d", d))))
            q = max(0, min(5, int(request.form.get("q", q))))
        except (TypeError, ValueError):
            flash("ARIMA order must be whole numbers.", "warning")
    session["arima_p"], session["arima_d"], session["arima_q"] = p, d, q

    order = (p, d, q)
    err_msg = None
    fc_rows = []
    plot_fc = None
    model_info = None
    eval_metrics = None

    try:
        res_model = get_fit(series, order)
        model_info = get_model_summary(res_model)
        fc_data = forecast_arima(res_model, steps=FORECAST_STEPS)

        # Demand cannot be negative, so the forecast and its interval are
        # clipped at zero before being shown or used for stock decisions.
        for idx, (dt, val, lo, hi) in enumerate(
            zip(future_index(series, FORECAST_STEPS),
                fc_data["forecast"], fc_data["lower_ci"], fc_data["upper_ci"]), 1
        ):
            fc_rows.append({
                "period": idx,
                "month": dt.strftime("%B %Y"),
                "forecast": round(max(0.0, val), 2),
                "lower_ci": round(max(0.0, lo), 2),
                "upper_ci": round(max(0.0, hi), 2),
            })

        plot_fc = plot_forecast(series, fc_data["forecast"], fc_data["lower_ci"], fc_data["upper_ci"])
        eval_metrics = backtest_forecast(series, order)

    except Exception as err:
        err_msg = f"ARIMA fitting error: {err}"

    return render_template(
        "forecast.html", active="forecast",
        p=p, d=d, q=q, category=cat,
        fc_rows=fc_rows, plot_fc=plot_fc, model_info=model_info,
        eval_metrics=eval_metrics, error=err_msg,
    )


# ---------------------------------------------------------------------------
# Stage 5 - Accuracy evaluation
# ---------------------------------------------------------------------------
@app.route("/evaluation")
def evaluation():
    """MAE / RMSE / MAPE, both in-sample (fitted vs actual) and out-of-sample
    (train on the first 75% of periods, forecast the remaining 25%)."""
    df, bail = require_data()
    if bail:
        return bail

    ts_df, _ = current_series(df)
    series = ts_df["Demand"]
    p, d, q = current_order(series)

    try:
        res = get_fit(series, (p, d, q))
        eval_data = evaluate_model(res, series)
        holdout = backtest_forecast(series, (p, d, q))
        plot_eval = plot_evaluation(eval_data["actual"], eval_data["fitted"])
        error = None
    except Exception as err:
        eval_data, holdout, plot_eval = None, None, None
        error = f"Could not evaluate ARIMA({p},{d},{q}): {err}"

    return render_template(
        "evaluation.html", active="evaluation",
        p=p, d=d, q=q, eval_data=eval_data, holdout=holdout,
        plot_eval=plot_eval, error=error,
    )


# ---------------------------------------------------------------------------
# Stage 6 - Inventory management driven by the forecast
# ---------------------------------------------------------------------------
@app.route("/inventory", methods=["GET", "POST"])
def inventory():
    """Turn the ARIMA forecast into a stocking decision.

    This is the step that closes the project pipeline: the *forecast* (not the
    historical average) drives demand during lead time, the reorder point and
    the reorder quantity.
    """
    df, bail = require_data()
    if bail:
        return bail

    ts_df, cat = current_series(df)
    series = ts_df["Demand"]
    p, d, q = current_order(series)

    # --- Forecast-driven demand -------------------------------------------
    forecast_failed = None
    try:
        res = get_fit(series, (p, d, q))
        fc = forecast_arima(res, steps=FORECAST_STEPS)
        fc_values = [max(0.0, v) for v in fc["forecast"]]
        # Next period's forecast is what the stock must actually cover.
        monthly_forecast = float(fc_values[0])
        annual_demand = float(np.mean(fc_values)) * 12.0
        demand_source = f"ARIMA({p},{d},{q}) forecast"
    except Exception as err:
        # Fall back to the historical mean so the page still works.
        forecast_failed = str(err)
        fc_values = []
        monthly_forecast = float(series.mean())
        annual_demand = monthly_forecast * 12.0
        demand_source = "historical average (forecast unavailable)"

    # Forecast uncertainty is the right measure of demand variability, but with
    # a short series the historical std is the more stable estimate, so use the
    # larger of the two - the conservative choice for safety stock.
    hist_std = float(series.std(ddof=1)) if len(series) > 1 else monthly_forecast * 0.2
    fc_std = float(np.std(fc_values, ddof=1)) if len(fc_values) > 1 else 0.0
    monthly_std = max(hist_std, fc_std)

    daily_avg = monthly_forecast / 30.0

    # --- User-supplied operating parameters -------------------------------
    def num(field, default):
        try:
            return float(request.form.get(field, default))
        except (TypeError, ValueError):
            return float(default)

    lead_days = max(0.5, num("lead_time", 7.0))
    service_lvl = min(99.9, max(50.0, num("service_level", 95.0)))
    curr_inv = max(0.0, num("current_inventory", round(monthly_forecast * 0.8)))
    ordering_cost = max(1.0, num("ordering_cost", 500.0))       # cost per purchase order
    holding_cost = max(1.0, num("holding_cost", 50.0))          # cost to hold 1 unit per year

    lead_months = lead_days / 30.0

    # --- Inventory mathematics --------------------------------------------
    # Safety Stock = z * sigma * sqrt(L)
    ss_info = compute_safety_stock(service_lvl / 100.0, monthly_std, lead_months)
    ss_val = ss_info["safety_stock"]

    # Reorder Point = (average demand during lead time) + safety stock
    rop_info = compute_reorder_point(daily_avg, lead_days, ss_val)
    rop_val = rop_info["rop"]

    # EOQ = sqrt(2DS / H)
    eoq_info = compute_eoq(annual_demand, ordering_cost, holding_cost)

    # Low-stock alert + reorder / investment recommendation
    rec = generate_recommendation(curr_inv, rop_val, eoq_info["eoq"], monthly_forecast, lead_months)
    rec["investment"] = round(rec["suggested_order"] * holding_cost / 12.0, 2)

    plot_inv = plot_inventory_chart(curr_inv, rop_val, ss_val, eoq_info["eoq"])

    inv_info = {
        "category": cat,
        "order": f"({p},{d},{q})",
        "demand_source": demand_source,
        "forecast_failed": forecast_failed,
        "forecast_next": round(monthly_forecast, 2),
        "forecast_values": [round(v, 1) for v in fc_values],
        "monthly_avg": round(float(series.mean()), 2),
        "daily_avg": round(daily_avg, 2),
        "monthly_std": round(monthly_std, 2),
        "annual_demand": round(annual_demand, 2),
        "lead_time_days": lead_days,
        "service_level": service_lvl,
        "current_inventory": curr_inv,
        "ordering_cost": ordering_cost,
        "holding_cost": holding_cost,
        "demand_during_lt": rop_info["demand_during_lt"],
        "safety_stock": ss_val,
        "rop": rop_val,
        "z_score": ss_info["z_score"],
        "sigma_lt": ss_info["sigma_lt"],
        "eoq": eoq_info["eoq"],
        "total_cost": eoq_info["total_cost"],
        # Alert + recommendation
        "status_text": rec["status"],
        "status_badge": rec["urgency"],
        "recommendation_msg": rec["message"],
        "action": rec["action"],
        "suggested_order": rec["suggested_order"],
        "investment": rec["investment"],
        "periods_to_rop": rec["periods_to_rop"],
        "plot_inv": plot_inv,
    }

    return render_template("inventory.html", active="inventory", inv=inv_info)


# ---------------------------------------------------------------------------
# Supporting / write-up pages
# ---------------------------------------------------------------------------
@app.route("/case-study")
def case_study():
    """Worked scenario: a festive demand spike that exceeded planned stock."""
    expected, actual = 4800, 6150
    unfulfilled = actual - expected
    scenario = {
        "product": "Amul Toned Milk 1 L",
        "expected_demand": expected,
        "actual_demand": actual,
        "unfulfilled_demand": unfulfilled,
        "lead_time": 2,
        "stockout_duration": 2,
        "emergency_cost": 18000,
        "fulfillment_pct": round(expected / actual * 100, 1),
        "shortage_pct": round(unfulfilled / actual * 100, 1),
    }
    return render_template("case_study.html", active="case_study", scenario=scenario)


@app.route("/mathematics")
def mathematics():
    return render_template("mathematics.html", active="mathematics")


@app.route("/math-derivation")
def math_derivation():
    return render_template("math_derivation.html", active="math_derivation")


@app.route("/conclusion")
def conclusion():
    return render_template("conclusion.html", active="conclusion")


if __name__ == "__main__":
    # Local development only. In production (Render) gunicorn imports the `app`
    # object directly and this block never runs, so debug stays off there.
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))
