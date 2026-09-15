# Smart Inventory Management and Demand Prediction Using ARIMA

A Flask dashboard that forecasts product demand from historical sales data using an
ARIMA model, and turns that forecast into concrete stocking decisions — safety stock,
reorder point, low-stock alerts and reorder/investment recommendations.

**BSc Mathematics project.**

---

## Pipeline

```
Dataset  →  Preprocessing  →  ARIMA Model  →  Demand Forecast  →  Inventory Management
   →  Low Stock Alert  →  Reorder / Investment Recommendation  →  Accuracy Evaluation  →  Dashboard
```

Each arrow is a page in the app, so the whole flow can be walked through during a review.

---

## Quick start

```bash
# 1. Create the virtual environment (Python 3.12 required — see note below)
py -3.12 -m venv .venv

# 2. Install dependencies
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. Run
.venv\Scripts\python.exe app.py
```

Then open <http://127.0.0.1:5000>.

> **Why Python 3.12?**
> `pandas >= 2.2` is needed for the `"ME"` (month-end) resample alias used in
> `ml/preprocess.py`, and `statsmodels` has no wheels for Python 3.13/3.14 yet.
> Python 3.10 is too old (its pandas predates `"ME"`).

---

## Dataset

The project uses the [Blinkit Sales Dataset](https://www.kaggle.com/datasets/akxiit/blinkit-sales-dataset)
from Kaggle. Three CSVs are merged into one transaction table:

| File | Role | Key |
|---|---|---|
| `blinkit_order_items.csv` | quantity + unit price per line item | `order_id`, `product_id` |
| `blinkit_orders.csv` | order date | `order_id` |
| `blinkit_products.csv` | product name, category | `product_id` |

These three files ship in `dataset/`, so **the app runs offline with no Kaggle account**.
`ml/preprocess.download_kaggle_dataset()` resolves them in this order:

1. `dataset/` in this project
2. the local `kagglehub` cache (`~/.cache/kagglehub/...`)
3. a fresh `kagglehub` download

After merging: **5,000 transactions**, 51 products, 11 categories,
**16 Mar 2023 → 04 Nov 2024**, aggregated to **19 complete monthly periods**.

### Using your own data

The **Upload Your CSV** page accepts a two-column file (date + demand); see
`dataset/sample_inventory.csv` for the expected shape. Column names are detected
case-insensitively (`Date`/`order_date`/`month`, `Demand`/`quantity`/`sales`/…).

---

## Project structure

```
app.py                  Flask routes — one per pipeline stage
ml/
  preprocess.py         dataset resolution, merging, time-series construction, statistics
  stationarity.py       ADF test, differencing, automatic choice of d
  arima_model.py        ACF/PACF, automatic choice of p and q, fitting, forecasting
  evaluation.py         MAE, RMSE, MAPE — in-sample and out-of-sample backtest
  inventory.py          EOQ, safety stock, reorder point, recommendation logic
  visualization.py      all matplotlib charts, returned as base64 PNGs
templates/              Jinja2 pages (MathJax for formulas)
static/css/style.css    dark dashboard theme
static/js/main.js       sidebar, chart download, form validation
dataset/                bundled CSVs + sample upload file
uploads/                user-uploaded CSVs (created at runtime)
```

---

## Page map

| Route | Stage | What it shows |
|---|---|---|
| `/` | — | Project overview |
| `/dataset` | Data | Schema, missing values, date range, preview |
| `/upload` | Data | Run the pipeline on your own CSV |
| `/analysis` | Preprocessing | Demand over time, descriptive statistics, ADF test |
| `/differencing` | Stationarity | Iterative ADF testing to choose **d** |
| `/acf-pacf` | Identification | ACF/PACF plots to choose **p** and **q** |
| `/arima` | Model | Coefficients, AIC/BIC/HQIC, residual diagnostics (Q-Q, histogram) |
| `/forecast` | Forecast | 6-period forecast with 95% confidence interval |
| `/evaluation` | Accuracy | MAE, RMSE, MAPE — in-sample and held-out |
| `/inventory` | Inventory | Safety stock, ROP, EOQ, low-stock alert, reorder recommendation |
| `/case-study` | Reference | Festive demand-spike scenario |
| `/mathematics`, `/math-derivation` | Reference | Formula reference and full derivations |
| `/about`, `/conclusion` | Reference | Methodology and findings |

---

## The mathematics

**ADF test** — tests $H_0$: the series has a unit root (non-stationary). The number of
differences $d$ is the smallest value for which the ADF p-value falls below 0.05.

**ARIMA(p, d, q)** on the differenced series $W_t = \Delta^d Y_t$:

$$W_t = c + \sum_{i=1}^{p}\phi_i W_{t-i} + \sum_{j=1}^{q}\theta_j \varepsilon_{t-j} + \varepsilon_t$$

**Accuracy**

$$\text{MAE}=\frac1n\sum|y_i-\hat y_i| \qquad
\text{RMSE}=\sqrt{\frac1n\sum(y_i-\hat y_i)^2} \qquad
\text{MAPE}=\frac{100}{n}\sum\left|\frac{y_i-\hat y_i}{y_i}\right|$$

**Inventory**

$$SS = Z_{\alpha}\,\sigma\sqrt{L} \qquad ROP = D_{\text{daily}}\cdot L + SS \qquad EOQ=\sqrt{\frac{2DS}{H}}$$

where $D$, the annual demand in EOQ, and $D_{\text{daily}}$, the demand rate in ROP,
both come from the **ARIMA forecast** — not from the historical average.

---

## Design decisions worth explaining in a review

**Partial months are trimmed.** The raw data starts on 16 Mar 2023 and ends on
4 Nov 2024, so those two monthly buckets are incomplete (227 and 58 units against a
~500 baseline). Left in, they look like a demand collapse and bias the model. See
`ml/preprocess.build_time_series`.

**Zero-demand periods are kept, not dropped.** Dropping an empty month would shift
every later observation one period earlier and corrupt the lag structure the AR and
MA terms are estimated from. ARIMA requires equally spaced observations.

**d is chosen by test, not assumed.** `suggest_d()` runs the ADF test repeatedly.
On this dataset it returns **d = 0** — the series is already stationary around ~500
units/month, so differencing it would only add noise.

**Accuracy is measured out-of-sample.** `backtest_forecast()` refits the model on the
first 75% of periods and scores it against the unseen remainder. In-sample metrics
are also shown on `/evaluation`, but they flatter the model because the same data
fitted the coefficients.

**Forecast uncertainty vs. historical variance.** Safety stock uses the larger of the
two — the conservative choice when the series is short.

---

## Not deployable to Cloudflare Workers

Workers run Python on Pyodide/WASM with a 64 MiB bundle limit, 128 MB memory and an
async `fetch` entrypoint. This app is WSGI (Flask, not ASGI) and its scientific stack
(pandas + statsmodels + matplotlib + scipy) exceeds those limits. Cloudflare
**Containers**, or any ordinary Python host, will run it unchanged.
