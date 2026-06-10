# Eataway AB — Weekly Demand Forecasting System

An end-to-end demand forecasting pipeline for **Eataway AB**, a Swedish prepared-meal
delivery business. The system predicts **weekly demand for every (store × product)
combination** and turns those predictions into two operational views:

- a **Kitchen view** — how much of each product to prepare per weekday, and
- a **Driver view** — what to load per route, store, and weekday.

It covers the full path from the company database to a live web app: pull raw orders →
clean & engineer features → train an ensemble model → generate operational views →
serve them through a Flask app (and export to Google Sheets).

---

## Highlights

- **Intermittent-demand aware.** Most (store, product) weeks have zero or very low
  sales, so a plain regressor systematically under- or over-shoots. The core model is a
  **calibrated hurdle model** (will it sell? → how much?) blended with a **Tweedie**
  model.
- **Leakage-free feature engineering.** All store/product/category aggregates and target
  encodings are computed using **training-period data only**, then mapped onto
  validation/test rows.
- **Swedish-calendar & weather features.** Public holidays (incl. high-impact ones like
  Midsommar, Jul, Påsk, Nyår), pre/post-holiday effects, and per-city weather.
- **Truncated-week handling.** Recent weeks have a data-entry lag, so they are detected,
  excluded from training, and used for out-of-sample validation instead.
- **Operational output, not just metrics.** Predictions are split into weekday-level
  kitchen and driver sheets with prediction intervals.
- **One-command weekly refresh.** `auto_sync.py` retrains locally and redeploys the app.

---

## System Architecture

```
Company MySQL DB
       │  (auto_sync.py pulls ~730 days of orders/returns)
       ▼
  1year.csv  ──►  feature.py  ──►  trainable_data.csv
                  (clean + feature engineering, leakage-free)
                       │
   weather_weekly.csv ─┤  (produced by eataway_weather.py)
                       ▼
              eataway_train_v7.py
        (Hurdle + Tweedie ensemble, time-split eval)
                       │
                       ▼
        output_v7/  +  predictions/
   (model files, kitchen_view, driver_view, metrics, plots)
                       │
                       ▼
        eataway_system/  (Flask web app on Railway)
        ── Kitchen view · Driver view · History query
        ── Google Sheets export
```

---

## Modeling Approach

The predictor in `eataway_train_v7.py` is a two-component ensemble (`V4Ensemble`):

**1. Calibrated Hurdle model**
- **Stage 1 — Classifier:** a LightGBM binary classifier predicts whether a
  (store, product) sells in a given week, followed by **isotonic calibration** so the
  probabilities are well-scaled.
- **Stage 2 — Regressor:** a LightGBM L1 regressor predicts the quantity **in log space**
  for positive cases, with a **lognormal back-transform correction** to counter the
  systematic under-estimation from Jensen's inequality.
- A **hard probability gate** (threshold tuned on validation) decides zero vs. positive.
- **Stratified bias correction** by predicted-value bin to fix regression-to-the-mean
  (especially under-estimation of high-demand items).

**2. Tweedie model**
- A LightGBM regressor with a `tweedie` objective, which handles the zero-inflated,
  right-skewed target directly.

**Ensemble & calibration**
- Ensemble weights are optimized on the validation set using a custom score
  (`MAE + penalty for under-estimation`, since stock-outs cost more than over-prep).
- A **global scale factor** is auto-derived (and clipped) to compensate for residual
  seasonal under-estimation; it is applied at the float level before rounding so
  low-probability stores aren't all rounded down to zero.

**Evaluation**
- **Time-based split:** last 6 weeks = test, previous 6 weeks = validation, rest = train.
- Reports MAE, bias, WMAPE, hit-rate (±1 / ±2), regression-to-mean ratios per demand
  band, per-week bias, false negatives, and feature importance.
- Out-of-sample validation on truncated weeks vs. known actual totals.

---

## Features

About 30 leakage-free features after pruning, grouped as:

- **Time / seasonality:** week, month, sin/cos cycles, `is_december`, `is_summer`,
  season one-hots.
- **Swedish holidays:** holiday-week flags, high-impact flag, pre/post-holiday weeks,
  holiday count, plus interactions (`combo_holiday_avg`, `holiday_lift_ratio`,
  `holiday_x_mean`, …).
- **Lag & rolling:** `lag_1w/2w`, rolling mean/std/median/max/q75 (4–12w), 4-week trend
  slope, year-over-year same week.
- **Returns:** rolling return rate, censored (sold-out) ratio.
- **Store / product / category aggregates:** weekly averages & dispersion, product share,
  route average, category × season — all computed on training data only.
- **Target encodings:** smoothed encodings for store, product, type.
- **Weather:** temperature, precipitation, wind, snow/rain days, anomalies, and a
  composite bad-weather score (merged by city × week).

---

## Repository Structure

```
.
├── auto_sync.py            # Orchestrator: DB pull → train → push/deploy → upload
├── feature.py              # Data cleaning + leakage-free feature engineering
├── eataway_weather.py      # Builds weather_weekly.csv (per-city weekly weather)
├── eataway_train_v7.py     # Hurdle + Tweedie ensemble: train, evaluate, generate views
├── eataway_system/         # Flask web app (prediction UI, history query, exports)
├── output_v7/              # Model artifacts, evaluation, kitchen/driver views, plots
├── predictions/            # Weekly prediction exports
├── 1year.csv.zip           # Raw historical orders/returns (sample)
├── weather_weekly.csv      # Weekly weather features
├── weather_forecast.csv    # Forward weather for upcoming weeks
├── eataway.xlsx            # Reference workbook
├── requirements.txt
├── Procfile                # gunicorn entry (deployment)
├── railway.toml            # Railway config
├── render.yaml             # Render config
└── run_sync.bat            # Windows scheduled-task entry for auto_sync.py
```

---

## Getting Started

### Requirements

Python 3.10+ and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key libraries: `lightgbm`, `scikit-learn`, `pandas`, `numpy`, `holidays`,
`flask` + `gunicorn` (web app), `pymysql` + `sqlalchemy` (database), `gspread` +
`google-auth` (Google Sheets export).

### 1. Prepare the data

Unzip the sample data (or supply your own `1year.csv` with columns
`datum, namn, ort, typ, sort, antal_ordrar, antal_returer`):

```bash
unzip 1year.csv.zip
```

### 2. Build weather features (optional but recommended)

```bash
python eataway_weather.py      # → weather_weekly.csv
```

### 3. Clean + engineer features

```bash
python feature.py              # → trainable_data.csv (+ cleaned_weekly.csv, features_ready.csv)
```

### 4. Train + evaluate + generate views

```bash
python eataway_train_v7.py
```

Outputs land in `output_v7/`:

- `model_cls.txt`, `model_reg.txt`, `model_tweedie.txt`, `calibrator.pkl` — model
- `evaluation_v7.csv`, `metrics_v7.csv`, `feature_importance_v7.csv` — diagnostics
- `kitchen_view_v7.csv`, `driver_view_v7.csv` — operational outputs
- `diagnostics_v7.png` — evaluation plots
- `config_v4.json` — thresholds, bias factors, ensemble weights

---

## Output Views

**Kitchen view** — aggregated by weekday × product, with quantity, prediction range, and
number of stores. Used to decide how much to cook each day.

**Driver view** — by route × store × weekday, with total items, range, product count, and
a per-store product breakdown. Used to load delivery vehicles.

Weekly volume is split across delivery days (Sun/Mon/Tue/Wed/Thu) by a fixed day-ratio,
and each prediction carries an approximate lower/upper interval.

---

## Automation & Deployment

`auto_sync.py` runs the weekly refresh end to end:

1. Pull the latest ~730 days of orders/returns from the company **MySQL** database.
2. Run `feature.py` and `eataway_train_v7.py` **locally** (training is RAM/CPU heavy).
3. Commit `output_v7/` and `predictions/` to GitHub → Railway redeploys automatically.
4. Wake the server, upload prediction results, and export the latest predictions to
   **Google Sheets**.

Schedule it weekly (e.g. Windows Task Scheduler via `run_sync.bat`).

The web app (`eataway_system/`) is a Flask app served by **gunicorn** (`Procfile`) and
configured for **Railway** (`railway.toml`) / **Render** (`render.yaml`). It queries the
database directly in production for live historical data.

### Configuration

`auto_sync.py` reads from a `.env` file:

```
DB_HOST=...
DB_PORT=3306
DB_USER=...
DB_PASSWORD=...
DB_NAME=...
SITE_URL=https://<your-app>.up.railway.app
APP_PASSWORD=...
```

> Keep `.env` and any Google service-account credentials out of version control.

---

## Notes & Limitations

- The dataset currently spans roughly one year, so the model cannot yet learn full
  seasonality from history; the global scale factor's lower bound is intentionally
  conservative and can be relaxed as more data accumulates.
- Recent weeks are treated as data-incomplete by design (entry lag of ~4 weeks).
- Predictions are tuned to slightly favor over-preparation, since missed sales (stock-outs)
  are costlier than modest over-prep for this business.

---

## License

No license specified yet. Add one (e.g. MIT) if you intend others to reuse the code.
