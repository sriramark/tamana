"""
Data acquisition and preprocessing for the Blinkit sales dataset.

Pipeline
--------
download_kaggle_dataset() -> locate/obtain the raw CSV files
load_blinkit_merged_df()  -> merge order_items + orders + products into one table
build_time_series()       -> aggregate transactions into a regular monthly demand series
get_statistics()          -> descriptive statistics used by the dashboard
"""

import os
import pandas as pd

# The three CSV files this project needs out of the Kaggle dataset.
REQUIRED_FILES = ("blinkit_orders.csv", "blinkit_order_items.csv", "blinkit_products.csv")

# Bundled copy that ships with the project: <project root>/dataset
PROJECT_DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")

# kagglehub's download cache for the current user (resolved at runtime, never hardcoded
# to one machine's user profile).
KAGGLE_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "kagglehub", "datasets",
    "akxiit", "blinkit-sales-dataset", "versions", "1",
)


def _has_required_files(folder: str) -> bool:
    """True only if *folder* contains all three CSVs we merge."""
    if not folder or not os.path.isdir(folder):
        return False
    return all(os.path.exists(os.path.join(folder, f)) for f in REQUIRED_FILES)


def download_kaggle_dataset(fallback_folder: str = PROJECT_DATASET_DIR) -> str:
    """Return a folder holding the Blinkit CSVs.

    Resolution order (cheapest first):
      1. the copy bundled with this project   -> works offline, no Kaggle account
      2. the local kagglehub download cache   -> already downloaded previously
      3. a fresh download via kagglehub       -> needs internet
    """
    if _has_required_files(fallback_folder):
        return fallback_folder

    if _has_required_files(KAGGLE_CACHE_DIR):
        return KAGGLE_CACHE_DIR

    try:
        import kagglehub
        path = kagglehub.dataset_download("akxiit/blinkit-sales-dataset")
        if _has_required_files(path):
            return path
        raise FileNotFoundError(f"Downloaded dataset at {path} is missing {REQUIRED_FILES}")
    except Exception as err:
        raise RuntimeError(
            "Could not locate the Blinkit dataset. Place "
            + ", ".join(REQUIRED_FILES)
            + f" in '{fallback_folder}', or install kagglehub to download it. Cause: {err}"
        ) from err


def load_blinkit_merged_df(target_dir: str = None) -> pd.DataFrame:
    if not target_dir:
        target_dir = download_kaggle_dataset()

    orders_path = os.path.join(target_dir, "blinkit_orders.csv")
    items_path = os.path.join(target_dir, "blinkit_order_items.csv")
    products_path = os.path.join(target_dir, "blinkit_products.csv")

    if not (os.path.exists(orders_path) and os.path.exists(items_path) and os.path.exists(products_path)):
        raise FileNotFoundError(f"Missing CSV files in {target_dir}")

    orders = pd.read_csv(orders_path)
    items = pd.read_csv(items_path)
    products = pd.read_csv(products_path)

    orders.columns = [c.strip().lower() for c in orders.columns]
    items.columns = [c.strip().lower() for c in items.columns]
    products.columns = [c.strip().lower() for c in products.columns]

    df_merged = items.merge(orders, on="order_id", how="inner").merge(products, on="product_id", how="inner")
    df_merged["order_date"] = pd.to_datetime(df_merged["order_date"], errors="coerce")
    df_merged["quantity"] = pd.to_numeric(df_merged["quantity"], errors="coerce").fillna(1)
    df_merged["unit_price"] = pd.to_numeric(df_merged["unit_price"], errors="coerce").fillna(0)
    df_merged["total_sales"] = df_merged["quantity"] * df_merged["unit_price"]

    df_merged = df_merged.dropna(subset=["order_date"]).sort_values("order_date").reset_index(drop=True)
    return df_merged

def get_dataset_metadata(df: pd.DataFrame) -> dict:
    total_rows = len(df)
    total_cols = len(df.columns)
    cols_list = list(df.columns)

    missing_dict = df.isna().sum().to_dict()
    total_missing = sum(missing_dict.values())

    min_dt = df["order_date"].min().strftime("%Y-%m-%d")
    max_dt = df["order_date"].max().strftime("%Y-%m-%d")

    categories = sorted(df["category"].dropna().unique().tolist())
    products = sorted(df["product_name"].dropna().unique().tolist())

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "columns": cols_list,
        "missing_dict": missing_dict,
        "total_missing": total_missing,
        "min_date": min_dt,
        "max_date": max_dt,
        "categories": categories,
        "num_categories": len(categories),
        "num_products": len(products),
    }

def build_time_series(
    df: pd.DataFrame,
    category: str = None,
    product_name: str = None,
    freq: str = "ME",
    trim_partial: bool = True,
) -> pd.DataFrame:
    """Aggregate raw transactions into a regular demand time series.

    ARIMA assumes observations are equally spaced in time, so this function
    guarantees a *gap-free* index:

    * ``resample(freq).sum()`` creates one row per period, even for periods with
      no orders (those become 0 rather than being skipped).
    * Periods with zero demand are **kept**. Dropping them would silently shift
      later observations earlier in time and corrupt the lag structure that the
      AR and MA terms are estimated from.
    * The first and last periods are dropped when the raw data does not cover
      them completely (``trim_partial``). The Blinkit data starts on 16-Mar-2023
      and ends on 04-Nov-2024, so those two months are partial buckets that would
      otherwise look like a demand collapse to the model.

    Parameters
    ----------
    freq : "ME" = month end, "W" = weekly. Falls back to weekly automatically
           when a monthly series would be too short to fit ARIMA on.
    """
    sub_df = df.copy()

    if category and category != "All Categories":
        sub_df = sub_df[sub_df["category"] == category]

    if product_name and product_name != "All Products":
        sub_df = sub_df[sub_df["product_name"] == product_name]

    if sub_df.empty:
        raise ValueError("No records found for selection")

    sub_df = sub_df.set_index("order_date").sort_index()
    ts = _resample_demand(sub_df, freq, trim_partial)

    # A monthly series that is too short for ARIMA -> retry at weekly resolution,
    # which yields roughly 4x more observations from the same data.
    if len(ts) < 8 and freq == "ME":
        ts = _resample_demand(sub_df, "W", trim_partial)

    return ts


def _resample_demand(indexed_df: pd.DataFrame, freq: str, trim_partial: bool) -> pd.DataFrame:
    """Resample a datetime-indexed frame to *freq* and return a Demand column."""
    ts = indexed_df.resample(freq)["quantity"].sum().to_frame(name="Demand")
    ts["Demand"] = ts["Demand"].fillna(0).astype(float)

    if trim_partial and len(ts) > 2:
        first_obs = indexed_df.index.min()
        last_obs = indexed_df.index.max()

        # Each resampled label is the *end* of its period. The opening period is
        # partial when the data begins after that period already started.
        period_start = ts.index[0] - _period_length(ts.index)
        if first_obs > period_start + pd.Timedelta(days=1):
            ts = ts.iloc[1:]

        # The closing period is partial when the data stops before the label date.
        if len(ts) > 1 and last_obs < ts.index[-1] - pd.Timedelta(days=1):
            ts = ts.iloc[:-1]

    return ts


def _period_length(idx: pd.DatetimeIndex) -> pd.Timedelta:
    """Approximate length of one period in a resampled index."""
    if len(idx) >= 2:
        return idx[1] - idx[0]
    return pd.Timedelta(days=30)


def get_statistics(ts_df: pd.DataFrame) -> dict:
    series = ts_df["Demand"]
    n = len(series)

    mean_val = float(series.mean())
    median_val = float(series.median())
    std_val = float(series.std(ddof=1)) if n > 1 else 0.0
    var_val = float(series.var(ddof=1)) if n > 1 else 0.0
    min_val = float(series.min())
    max_val = float(series.max())
    range_val = max_val - min_val
    skew_val = float(series.skew()) if n > 2 else 0.0
    kurt_val = float(series.kurt()) if n > 3 else 0.0
    cv_val = (std_val / mean_val * 100) if mean_val != 0 else 0.0
    q1_val = float(series.quantile(0.25))
    q3_val = float(series.quantile(0.75))
    iqr_val = q3_val - q1_val

    return {
        "count": n,
        "mean": round(mean_val, 2),
        "median": round(median_val, 2),
        "std": round(std_val, 2),
        "variance": round(var_val, 2),
        "min": round(min_val, 2),
        "max": round(max_val, 2),
        "range": round(range_val, 2),
        "skewness": round(skew_val, 2),
        "kurtosis": round(kurt_val, 2),
        "cv": round(cv_val, 2),
        "q1": round(q1_val, 2),
        "q3": round(q3_val, 2),
        "iqr": round(iqr_val, 2),
    }

def get_preview(df: pd.DataFrame, n: int = 10) -> list[dict]:
    prev_df = df.head(n).copy()
    if "order_date" in prev_df.columns:
        prev_df["order_date"] = prev_df["order_date"].dt.strftime("%Y-%m-%d")
    return prev_df.to_dict(orient="records")
