"""
forecast.py — Two-step forecasting pipeline.

Upgraded for AWS SageMaker ml.m5.2xlarge (8 vCPUs, 32GB RAM):
  - XGBoost + LightGBM ensemble (averaged predictions)
  - Optuna hyperparameter tuning (OPTUNA_TRIALS trials per model)
  - 5-fold time-series cross-validation during tuning
  - 1000+ estimators in final models
  - n_jobs=-1 throughout

Step A: Forecast daily call volume totals for August using XGBoost + LightGBM
        ensemble trained on 2-year daily data (Jan 2024 – Dec 2025).

Step B: Disaggregate daily totals to 30-min intervals using intraday shape
        patterns from Apr–Jun interval data.

Step C: Derive CCT and Abandoned Rate from interval-level historical averages,
        blended with service-level signal (90/10 weights, calibrated).

Output: forecast_v01.csv matching template_forecast_v00.csv exactly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import optuna
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Constants ─────────────────────────────────────────────────────────────────

EXCEL_PATH = Path("/Users/falmug/Downloads/DATATHON/Data/Data for Datathon (Revised).xlsx")
DATA_DIR = Path("/Users/falmug/Downloads/DATATHON/Data")
TEMPLATE_PATH = Path("/Users/falmug/Downloads/DATATHON/template_forecast_v00.csv")
OUTPUT_PATH = Path("/Users/falmug/Downloads/DATATHON/forecast_v01.csv")
PORTFOLIOS = ["A", "B", "C", "D"]

# Upward bias multiplier applied to every interval forecast.
# Scoring penalises understaffing more than overstaffing.
# Tune based on α/β ratio from scoring formula:
#   if α/β > 2 consider increasing to 1.05
#   if α/β close to 1 reduce to 1.01
UPWARD_BIAS = 1.02

# Optuna + CV settings
# 50 trials × 2 frameworks × 12 models = 1200 total Optuna fits
# Each fit: 5-fold CV on ~700 daily rows — fast on SageMaker
OPTUNA_TRIALS = 50
N_CV_FOLDS = 5

US_HOLIDAYS = pd.to_datetime([
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27",
    "2024-07-04", "2024-09-02", "2024-10-14", "2024-11-11",
    "2024-11-28", "2024-11-29", "2024-12-25",
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26",
    "2025-07-04", "2025-09-01", "2025-10-13", "2025-11-11",
    "2025-11-27", "2025-11-28", "2025-12-25",
    # August 2025 forecast target — no major holidays
]).normalize()


# ── Step A: Load & prepare daily data ────────────────────────────────────────

def load_daily_data(portfolio: str) -> pd.DataFrame:
    sheet = f"{portfolio} - Daily"
    df = pd.read_excel(EXCEL_PATH, sheet_name=sheet, header=0)
    df.columns = ["Date", "Call_Volume", "CCT", "Service_Level", "Abandon_Rate"]

    # Parse date like "01/01/24 Mon"
    df["Date"] = pd.to_datetime(df["Date"].astype(str).str[:8], format="%m/%d/%y")
    df = df.dropna(subset=["Date", "Call_Volume"])
    df["CCT"] = pd.to_numeric(df["CCT"], errors="coerce").ffill().fillna(0)
    df["Abandon_Rate"] = pd.to_numeric(df["Abandon_Rate"], errors="coerce").fillna(0).clip(0, 1)
    df["Service_Level"] = pd.to_numeric(df["Service_Level"], errors="coerce").fillna(0)
    df = df.sort_values("Date").reset_index(drop=True)
    df["portfolio"] = portfolio
    return df


def make_daily_features(dates: pd.Series) -> pd.DataFrame:
    """Build feature matrix from a series of dates."""
    feat = pd.DataFrame({"date": pd.to_datetime(dates)})
    feat["day_of_week"] = feat["date"].dt.dayofweek
    feat["month"] = feat["date"].dt.month
    feat["week_of_year"] = feat["date"].dt.isocalendar().week.astype(int)
    feat["day_of_month"] = feat["date"].dt.day
    feat["is_weekend"] = feat["day_of_week"].isin([5, 6]).astype(int)
    feat["is_holiday"] = feat["date"].isin(US_HOLIDAYS).astype(int)
    feat["dow_sin"] = np.sin(2 * np.pi * feat["day_of_week"] / 7)
    feat["dow_cos"] = np.cos(2 * np.pi * feat["day_of_week"] / 7)
    feat["month_sin"] = np.sin(2 * np.pi * feat["month"] / 12)
    feat["month_cos"] = np.cos(2 * np.pi * feat["month"] / 12)
    feat["week_sin"] = np.sin(2 * np.pi * feat["week_of_year"] / 52)
    feat["week_cos"] = np.cos(2 * np.pi * feat["week_of_year"] / 52)
    # Billing cycle features (retail credit card — spikes around statement/payment dates)
    feat["is_billing_date"] = feat["day_of_month"].isin([1, 2, 15, 16]).astype(int)
    feat["days_to_billing"] = feat["day_of_month"].apply(
        lambda d: min(abs(d - 1), abs(d - 15), abs(d - 31))
    )
    return feat.drop(columns=["date"])


# ── Optuna objectives ─────────────────────────────────────────────────────────

def _cv_score(model_fn, X, y):
    """TimeSeriesSplit CV MAE — respects temporal ordering, no future leakage."""
    tscv = TimeSeriesSplit(n_splits=N_CV_FOLDS)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        m = model_fn()
        m.fit(X[train_idx], y[train_idx])
        scores.append(mean_absolute_error(y[val_idx], m.predict(X[val_idx])))
    return float(np.mean(scores))


def xgb_objective(trial, X, y):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 500, 2000),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 8),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        random_state=42, n_jobs=-1,
    )
    return _cv_score(lambda: XGBRegressor(**params), X, y)


def lgb_objective(trial, X, y):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 500, 2000),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 8),
        num_leaves=trial.suggest_int("num_leaves", 20, 150),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        random_state=42, n_jobs=-1, verbose=-1,
    )
    return _cv_score(lambda: lgb.LGBMRegressor(**params), X, y)


# ── Step A: Tune and train per portfolio ──────────────────────────────────────

def tune_and_train(X: np.ndarray, y: np.ndarray, label: str):
    """
    Run Optuna for XGBoost and LightGBM separately, then train final models
    on the full dataset using the best hyperparameters found.
    Returns (xgb_model, lgb_model).
    """
    # XGBoost
    xgb_study = optuna.create_study(direction="minimize",
                                     sampler=optuna.samplers.TPESampler(seed=42))
    xgb_study.optimize(lambda t: xgb_objective(t, X, y),
                        n_trials=OPTUNA_TRIALS, n_jobs=1, show_progress_bar=False)
    best_xgb = xgb_study.best_params
    xgb_model = XGBRegressor(**best_xgb, random_state=42, n_jobs=-1)
    xgb_model.fit(X, y)
    print(f"    [{label}] XGB  CV-MAE={xgb_study.best_value:.2f}  "
          f"n_est={best_xgb['n_estimators']}  lr={best_xgb['learning_rate']:.4f}")

    # LightGBM
    lgb_study = optuna.create_study(direction="minimize",
                                     sampler=optuna.samplers.TPESampler(seed=42))
    lgb_study.optimize(lambda t: lgb_objective(t, X, y),
                        n_trials=OPTUNA_TRIALS, n_jobs=1, show_progress_bar=False)
    best_lgb = lgb_study.best_params
    lgb_model = lgb.LGBMRegressor(**best_lgb, random_state=42, n_jobs=-1, verbose=-1)
    lgb_model.fit(X, y)
    print(f"    [{label}] LGB  CV-MAE={lgb_study.best_value:.2f}  "
          f"n_est={best_lgb['n_estimators']}  lr={best_lgb['learning_rate']:.4f}")

    return xgb_model, lgb_model


def train_daily_model(portfolio: str) -> dict:
    """Train XGBoost+LightGBM ensemble for CV, CCT, and ABD for one portfolio."""
    df = load_daily_data(portfolio)
    X = make_daily_features(df["Date"]).values
    print(f"\n  Portfolio {portfolio}: {len(df)} rows  "
          f"({df['Date'].min().date()} – {df['Date'].max().date()})")

    models = {}
    for target, col in [("cv", "Call_Volume"), ("cct", "CCT"), ("abd", "Abandon_Rate")]:
        print(f"  Tuning {target.upper()} ...")
        y = df[col].values
        models[target] = tune_and_train(X, y, f"{portfolio}/{target}")
    return models


def predict_ensemble(xgb_model, lgb_model, X: np.ndarray) -> np.ndarray:
    """Average XGBoost and LightGBM predictions (equal weight ensemble)."""
    return (xgb_model.predict(X) + lgb_model.predict(X)) / 2.0


# ── Step B: Intraday shape from interval data ─────────────────────────────────

def compute_intraday_shapes() -> pd.DataFrame:
    """
    For each (portfolio, day_of_week, interval_index):
      shape_fraction = mean(call_volume) / sum(mean(call_volume) across day)
    """
    df = pd.read_csv(DATA_DIR / "all_portfolios_preprocessed.csv", parse_dates=["timestamp"])
    shape = (
        df.groupby(["portfolio", "day_of_week", "interval_index"])["Call_Volume"]
        .mean().reset_index().rename(columns={"Call_Volume": "mean_cv"})
    )
    day_totals = (shape.groupby(["portfolio", "day_of_week"])["mean_cv"]
                  .sum().reset_index().rename(columns={"mean_cv": "day_total"}))
    shape = shape.merge(day_totals, on=["portfolio", "day_of_week"])
    shape["shape_frac"] = np.where(
        shape["day_total"] > 0, shape["mean_cv"] / shape["day_total"], 1 / 48
    )
    return shape


# ── Step C: CCT, Abandoned Rate, and Service Level ────────────────────────────

def compute_intraday_cct_abd() -> pd.DataFrame:
    """
    For each (portfolio, day_of_week, interval_index):
      mean CCT, mean Abandoned Rate, and mean Service_Level.
    Only call-volume-positive intervals are used to exclude overnight zeros.
    """
    df = pd.read_csv(DATA_DIR / "all_portfolios_preprocessed.csv", parse_dates=["timestamp"])
    df_cv = df[df["Call_Volume"] > 0]

    cct_shape = (df_cv.groupby(["portfolio", "day_of_week", "interval_index"])["CCT"]
                 .mean().reset_index().rename(columns={"CCT": "mean_cct"}))
    abd_shape = (df_cv.groupby(["portfolio", "day_of_week", "interval_index"])["Abandoned_Rate"]
                 .mean().reset_index().rename(columns={"Abandoned_Rate": "mean_abd"}))
    svc_shape = (df_cv.groupby(["portfolio", "day_of_week", "interval_index"])["Service_Level"]
                 .mean().reset_index().rename(columns={"Service_Level": "mean_svc_level"}))

    shapes = (cct_shape
              .merge(abd_shape, on=["portfolio", "day_of_week", "interval_index"], how="outer")
              .merge(svc_shape, on=["portfolio", "day_of_week", "interval_index"], how="outer"))
    shapes["mean_cct"] = shapes["mean_cct"].fillna(0)
    shapes["mean_abd"] = shapes["mean_abd"].clip(0, 1).fillna(0)
    shapes["mean_svc_level"] = shapes["mean_svc_level"].clip(0, 1).fillna(0)
    return shapes


# ── August date grid ──────────────────────────────────────────────────────────

def build_august_dates() -> pd.DataFrame:
    """August 2025: 31 days × 48 intervals = 1488 rows."""
    dates = pd.date_range("2025-08-01", "2025-08-31", freq="D")
    intervals = pd.date_range("2025-08-01 00:00", "2025-08-01 23:30", freq="30min")
    rows = []
    for d in dates:
        for idx, ivt in enumerate(intervals):
            rows.append({
                "timestamp": pd.Timestamp(year=d.year, month=d.month, day=d.day,
                                          hour=ivt.hour, minute=ivt.minute),
                "Day": d.day,
                "day_of_week": d.dayofweek,
                "interval_index": idx,
                "Interval": f"{ivt.hour}:{ivt.minute:02d}",
            })
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Step A: Tune and train daily models ===")
    print(f"    {OPTUNA_TRIALS} Optuna trials × 2 frameworks × 12 models  "
          f"({N_CV_FOLDS}-fold TimeSeriesSplit CV)")
    models = {}
    for p in PORTFOLIOS:
        models[p] = train_daily_model(p)

    print("\n=== Step B & C: Compute intraday shapes ===")
    cv_shape = compute_intraday_shapes()
    cct_abd_shape = compute_intraday_cct_abd()
    intraday = cv_shape.merge(cct_abd_shape,
                               on=["portfolio", "day_of_week", "interval_index"], how="left")

    print("\n=== Building August 2025 forecast ===")
    aug = build_august_dates()
    august_dates = pd.date_range("2025-08-01", "2025-08-31", freq="D")
    daily_feats = make_daily_features(pd.Series(august_dates)).values

    daily_forecasts = {}
    for p in PORTFOLIOS:
        xgb_cv, lgb_cv = models[p]["cv"]
        xgb_cct, lgb_cct = models[p]["cct"]
        xgb_abd, lgb_abd = models[p]["abd"]
        daily_forecasts[p] = {
            "cv":  predict_ensemble(xgb_cv,  lgb_cv,  daily_feats),
            "cct": predict_ensemble(xgb_cct, lgb_cct, daily_feats),
            "abd": predict_ensemble(xgb_abd, lgb_abd, daily_feats),
        }

    template = pd.read_csv(TEMPLATE_PATH)
    result = template.copy()

    for p in PORTFOLIOS:
        cv_list, abd_list, abd_calls_list, cct_list = [], [], [], []

        for _, row in aug.iterrows():
            day_idx = row["Day"] - 1
            dow = row["day_of_week"]
            iidx = row["interval_index"]

            daily_cv = max(daily_forecasts[p]["cv"][day_idx], 0)

            match = intraday[
                (intraday["portfolio"] == p) &
                (intraday["day_of_week"] == dow) &
                (intraday["interval_index"] == iidx)
            ]
            if len(match) > 0:
                frac = float(match["shape_frac"].iloc[0])
                mean_cct = float(match["mean_cct"].iloc[0])
                mean_abd = float(match["mean_abd"].iloc[0])
                mean_svc_level = float(match["mean_svc_level"].iloc[0])
            else:
                frac = 1 / 48
                mean_cct = 0.0
                mean_abd = 0.0
                mean_svc_level = 0.0

            interval_cv = daily_cv * frac * UPWARD_BIAS

            # 90/10 blend: historical ABD + service-level-implied signal.
            # (1-SVC) runs ~4x higher than actual ABD so 10% weight is calibrated.
            # Validated: hist ABD ~2-3%, (1-SVC) ~6-9% per portfolio.
            if mean_svc_level > 0:
                svc_implied_abd = np.clip(1.0 - mean_svc_level, 0, 1)
                interval_abd = np.clip(0.9 * mean_abd + 0.1 * svc_implied_abd, 0, 1)
            else:
                interval_abd = np.clip(mean_abd, 0, 1)

            interval_abd_calls = round(interval_cv * interval_abd)
            interval_cct = max(mean_cct, 0)

            cv_list.append(round(interval_cv, 2))
            abd_list.append(round(interval_abd, 6))
            abd_calls_list.append(int(interval_abd_calls))
            cct_list.append(round(interval_cct, 2))

        result[f"Calls_Offered_{p}"] = cv_list
        result[f"Abandoned_Rate_{p}"] = abd_list
        result[f"Abandoned_Calls_{p}"] = abd_calls_list
        result[f"CCT_{p}"] = cct_list

    # Clip negatives
    for p in PORTFOLIOS:
        result[f"Calls_Offered_{p}"] = result[f"Calls_Offered_{p}"].clip(lower=0)
        result[f"Abandoned_Calls_{p}"] = result[f"Abandoned_Calls_{p}"].clip(lower=0)
        result[f"Abandoned_Rate_{p}"] = result[f"Abandoned_Rate_{p}"].clip(0, 1)
        result[f"CCT_{p}"] = result[f"CCT_{p}"].clip(lower=0)

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved forecast to {OUTPUT_PATH}  shape={result.shape}")

    print("\n=== Sanity checks ===")
    for p in PORTFOLIOS:
        total_cv = result[f"Calls_Offered_{p}"].sum()
        avg_abd = result[f"Abandoned_Rate_{p}"].mean()
        avg_cct = result[f"CCT_{p}"].mean()
        print(f"  Portfolio {p}: total_CV={total_cv:,.0f}  avg_ABD={avg_abd:.3f}  avg_CCT={avg_cct:.1f}")


if __name__ == "__main__":
    main()
