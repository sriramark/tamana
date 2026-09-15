import io
import base64
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import scipy.stats as stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Chart palette. These mirror the CSS design tokens in static/css/style.css so
# the generated PNGs sit inside the page instead of clashing with it.
#
# ACCENT LOCK: cobalt is the only accent. A chart still needs to separate the
# observed series from the projected one, so exactly two hues carry meaning:
#   cobalt  = measured / historical / fitted
#   amber   = projected / forecast / uncertain
# Everything else is neutral. Green appears only for the safety-stock band,
# where it encodes real inventory state, matching --state-safe in the CSS.
# ---------------------------------------------------------------------------
bg_color   = "#ffffff"   # --bg-primary
card_color = "#ffffff"   # --bg-card
grid_color = "#e4e8ee"   # --border-color
txt_color  = "#14171c"   # --text-primary
mut_color  = "#737b87"   # --text-secondary

c_accent  = "#2563eb"    # --accent, the locked accent
c_fill    = "#2563eb"    # --accent, for fills and bars
c_project = "#c2410c"    # rust, the projected series (line only, not text)
c_safe    = "#047857"    # --state-safe, safety-stock band only

# Backwards-compatible aliases used further down this module.
c_blue   = c_accent
c_orange = c_project

def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=120)
    buf.seek(0)
    enc = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return enc

def _apply_style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_facecolor(card_color)
    ax.set_title(title, color=txt_color, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=txt_color, fontsize=10)
    ax.set_ylabel(ylabel, color=txt_color, fontsize=10)
    ax.tick_params(colors=txt_color, labelsize=8)
    ax.spines["bottom"].set_color(grid_color)
    ax.spines["left"].set_color(grid_color)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=grid_color, linewidth=0.5, alpha=0.7)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(txt_color)

def plot_time_series(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(12, 4.5), facecolor=bg_color)
    ax.plot(df.index, df["Demand"], color=c_blue, linewidth=1.8, label="Demand")
    ax.fill_between(df.index, df["Demand"], alpha=0.07, color=c_fill)
    _apply_style(ax, "Demand Time Series", "Date", "Demand (Units)")
    ax.legend(facecolor=card_color, edgecolor=grid_color, labelcolor=txt_color)
    fig.tight_layout()
    return _fig_to_base64(fig)

def plot_rolling_stats(df: pd.DataFrame, window: int = 12) -> str:
    r_mean = df["Demand"].rolling(window=window, min_periods=1).mean()
    r_std = df["Demand"].rolling(window=window, min_periods=1).std()

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), facecolor=bg_color)

    axes[0].plot(df.index, df["Demand"], color=c_blue, linewidth=1.5, label="Original", alpha=0.8)
    axes[0].plot(df.index, r_mean, color=c_orange, linewidth=2, label=f"Rolling Mean (w={window})")
    _apply_style(axes[0], f"Rolling Mean (Window = {window})", "Date", "Demand")
    axes[0].legend(facecolor=card_color, edgecolor=grid_color, labelcolor=txt_color)

    axes[1].plot(df.index, r_std, color=c_safe, linewidth=2, label=f"Rolling Std (w={window})")
    axes[1].fill_between(df.index, r_std, alpha=0.15, color=c_safe)
    _apply_style(axes[1], f"Rolling Standard Deviation (Window = {window})", "Date", "Std Dev")
    axes[1].legend(facecolor=card_color, edgecolor=grid_color, labelcolor=txt_color)

    fig.tight_layout(pad=2.0)
    return _fig_to_base64(fig)

def plot_differenced(original: pd.Series, diff_series: pd.Series) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), facecolor=bg_color)

    axes[0].plot(original.index, original.values, color=c_blue, linewidth=1.8)
    axes[0].fill_between(original.index, original.values, alpha=0.12, color=c_blue)
    _apply_style(axes[0], "Original Series", "Date", "Demand")

    diff_clean = diff_series.dropna()
    axes[1].plot(diff_clean.index, diff_clean.values, color=c_orange, linewidth=1.8)
    axes[1].fill_between(diff_clean.index, diff_clean.values, alpha=0.12, color=c_orange)
    axes[1].axhline(0, color=txt_color, linewidth=0.8, linestyle="--", alpha=0.5)
    _apply_style(axes[1], "First-Differenced Series", "Date", "Delta Demand")

    fig.tight_layout(pad=2.0)
    return _fig_to_base64(fig)

def plot_acf_pacf(series: pd.Series, nlags: int = 20) -> str:
    s_clean = series.dropna()
    n = len(s_clean)
    lags = min(nlags, n // 2 - 1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), facecolor=bg_color)

    plot_acf(s_clean, lags=lags, ax=axes[0], color=c_blue, title="", zero=False, alpha=0.05)
    _apply_style(axes[0], "Autocorrelation Function (ACF)", "Lag", "Correlation")
    for patch in axes[0].patches:
        patch.set_facecolor(c_blue)
        patch.set_alpha(0.7)

    plot_pacf(s_clean, lags=lags, ax=axes[1], color=c_orange, method="ywm", title="", zero=False, alpha=0.05)
    _apply_style(axes[1], "Partial Autocorrelation Function (PACF)", "Lag", "Partial Corr.")
    for patch in axes[1].patches:
        patch.set_facecolor(c_orange)
        patch.set_alpha(0.7)

    for ax in axes:
        for line in ax.lines:
            if line.get_linestyle() == "--":
                line.set_color(c_project)
                line.set_linewidth(1.2)
        ax.set_facecolor(card_color)

    fig.tight_layout(pad=2.0)
    return _fig_to_base64(fig)

def plot_forecast(
    series: pd.Series,
    forecast: list,
    lower_ci: list,
    upper_ci: list
) -> str:
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=bg_color)

    ax.plot(range(len(series)), series.values, color=c_blue, linewidth=1.8, label="Historical Demand", zorder=3)

    offset = len(series)
    x_fc = range(offset, offset + len(forecast))
    ax.plot(x_fc, forecast, color=c_orange, linewidth=2, label="Forecast", zorder=3)
    # The interval band uses the accent hue, not the amber of the forecast line:
    # keeping the band and the line in different hues separates them cleanly.
    ax.fill_between(x_fc, lower_ci, upper_ci, color=c_fill, alpha=0.22, label="95% CI")
    ax.axvline(x=offset - 1, color=txt_color, linewidth=1, linestyle="--", alpha=0.5)

    _apply_style(ax, "ARIMA Demand Forecast", "Period", "Demand (Units)")
    ax.legend(facecolor=card_color, edgecolor=grid_color, labelcolor=txt_color)

    fig.tight_layout()
    return _fig_to_base64(fig)

def plot_residuals(residuals: pd.Series) -> str:
    res = residuals.dropna()

    fig = plt.figure(figsize=(14, 9), facecolor=bg_color)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(range(len(res)), res.values, color=c_blue, linewidth=1.2, alpha=0.8)
    ax1.axhline(0, color=c_orange, linewidth=1.5, linestyle="--")
    _apply_style(ax1, "Residuals over Time", "Period", "Residual")

    ax2 = fig.add_subplot(gs[1, 0])
    n_vals, bins, patches = ax2.hist(res, bins=20, color=c_blue, edgecolor=bg_color, alpha=0.8)
    mu, sigma = res.mean(), res.std()
    x_norm = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
    y_norm = (n_vals.max() / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_norm-mu)/sigma)**2)
    ax2.plot(x_norm, y_norm, color=c_orange, linewidth=2, label="Normal Fit")
    _apply_style(ax2, "Residual Histogram", "Residual", "Frequency")
    ax2.legend(facecolor=card_color, edgecolor=grid_color, labelcolor=txt_color, fontsize=8)

    ax3 = fig.add_subplot(gs[1, 1])
    (osm, osr), (slope, intercept, r) = stats.probplot(res, dist="norm")
    ax3.scatter(osm, osr, color=c_blue, s=15, alpha=0.7, label="Sample Quantiles")
    x_line = np.array([osm[0], osm[-1]])
    ax3.plot(x_line, slope * x_line + intercept, color=c_orange, linewidth=2, label="Theoretical Line")
    _apply_style(ax3, "Normal Q-Q Plot", "Theoretical Quantiles", "Sample Quantiles")
    ax3.legend(facecolor=card_color, edgecolor=grid_color, labelcolor=txt_color, fontsize=8)

    return _fig_to_base64(fig)

def plot_evaluation(actual: list, fitted: list) -> str:
    fig, ax = plt.subplots(figsize=(12, 4.5), facecolor=bg_color)
    n = min(len(actual), len(fitted))
    x = range(n)
    ax.plot(x, actual[:n], color=c_blue, linewidth=1.8, label="Actual Demand", zorder=3)
    ax.plot(x, fitted[:n], color=c_orange, linewidth=1.8, label="Fitted (ARIMA)", zorder=3, linestyle="--")
    _apply_style(ax, "Actual vs Fitted", "Period", "Demand (Units)")
    ax.legend(facecolor=card_color, edgecolor=grid_color, labelcolor=txt_color)
    fig.tight_layout()
    return _fig_to_base64(fig)

def plot_inventory_chart(
    current: float,
    rop_val: float,
    safety_stock: float,
    eoq_val: float,
) -> str:
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=bg_color)

    cycles = 3
    cycle_len = 30
    t_total = cycles * cycle_len
    t_arr = np.linspace(0, t_total, t_total * 10)

    inv_levels = []
    q_val = current + eoq_val
    for ti in t_arr:
        pos = ti % cycle_len
        level = q_val - (q_val - safety_stock) * (pos / cycle_len)
        inv_levels.append(level)

    ax.plot(t_arr, inv_levels, color=c_blue, linewidth=2, label="Inventory Level")
    ax.axhline(rop_val, color=c_orange, linewidth=1.8, linestyle="--", label=f"Reorder Point ({rop_val:.0f})")
    ax.axhline(safety_stock, color=c_safe, linewidth=1.5, linestyle=":", label=f"Safety Stock ({safety_stock:.0f})")
    ax.axhline(current, color=txt_color, linewidth=1.5, linestyle="-.", label=f"Current Inventory ({current:.0f})")

    ax.fill_between(t_arr, 0, safety_stock, color=c_safe, alpha=0.07, label="Safety Buffer Zone")
    _apply_style(ax, "Inventory Sawtooth Model", "Time (Periods)", "Inventory Level (Units)")
    ax.legend(facecolor=card_color, edgecolor=grid_color, labelcolor=txt_color, fontsize=8)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    return _fig_to_base64(fig)
