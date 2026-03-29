"""
forecast_direct.py — Direct interval-level XGBoost+LightGBM ensemble.

Trains one model per target (CV, CCT, ABD) directly on 30-min interval data
across all 4 portfolios combined. Uses actual August 2025 daily totals from
the Excel file as an anchor feature at inference.

Usage:
    python3 forecast_direct.py

Output:
    forecast_v20.csv  — 1488 rows x 19 cols matching template_forecast_v00.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

import xgboost as xgb
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
EXCEL_PATH    = BASE_DIR / "Data" / "Data for Datathon (Revised).xlsx"
DATA_PATH     = BASE_DIR / "Data" / "all_portfolios_preprocessed.csv"
TEMPLATE_PATH = BASE_DIR / "template_forecast_v00.csv"
OUTPUT_PATH   = BASE_DIR / "forecast_v20.csv"

# ── Constants ─────────────────────────────────────────────────────────────────
PORTFOLIOS    = ["A", "B", "C", "D"]
PORT_CODE     = {"A": 0, "B": 1, "C": 2, "D": 3}
OPTUNA_TRIALS = 30
N_CV_FOLDS    = 5
SEED          = 42

FEATURE_COLS = [
    "hour", "minute", "day_of_week", "day_of_month", "month", "interval_index",
    "is_weekend", "is_holiday", "is_billing_date", "days_to_billing",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "interval_sin", "interval_cos",
    "is_morning", "is_afternoon", "is_evening", "is_night",
    "Call_Volume_lag_1", "Call_Volume_lag_48", "Call_Volume_lag_336",
    "CCT_lag_1", "CCT_lag_48", "CCT_lag_336",
    "Abandoned_Rate_lag_1", "Abandoned_Rate_lag_48", "Abandoned_Rate_lag_336",
    "Call_Volume_rollmean_4", "Call_Volume_rollmean_48",
    "portfolio_code", "actual_daily_cv",
]

US_HOLIDAYS_2025 = set(pd.to_datetime([
    "2025-01-01",  # New Year's
    "2025-01-20",  # MLK Day
    "2025-02-17",  # Presidents Day
    "2025-05-26",  # Memorial Day
    "2025-07-04",  # Independence Day
    "2025-09-01",  # Labor Day
    "2025-10-13",  # Columbus Day
    "2025-11-11",  # Veterans Day
    "2025-11-27",  # Thanksgiving
    "2025-11-28",  # Black Friday
    "2025-12-25",  # Christmas
]).date)


# ── Step 1: Load training data ─────────────────────────────────────────────────
def load_training_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df["portfolio_code"] = df["portfolio"].map(PORT_CODE)
    df["date"] = df["timestamp"].dt.date
    df["actual_daily_cv"] = df.groupby(["portfolio", "date"])["Call_Volume"].transform("sum")
    # Drop rows with NaN in any feature column (early rows where lag_336 not yet available)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    print(f"  Training rows after dropna on features: {len(df):,}")
    return df


# ── Step 2: Historical profiles for lag inference proxy ───────────────────────
def build_hist_profiles(df: pd.DataFrame) -> dict:
    """
    Per-(portfolio, dow, interval_index):
      - Mean of every lag/rollmean column in FEATURE_COLS
      - Mean daily CV for CV-lag scaling at inference
    """
    lag_cols = [c for c in FEATURE_COLS if "lag" in c or "rollmean" in c]

    profiles = {}
    for p in PORTFOLIOS:
        sub = df[df["portfolio"] == p].copy()
        grp = sub.groupby(["day_of_week", "interval_index"])
        lag_means   = grp[lag_cols].mean()            # MultiIndex (dow, idx) → lag values
        daily_cv_mean = sub.groupby("day_of_week")["actual_daily_cv"].mean()  # dow → mean daily CV
        profiles[p] = {
            "lag_means":      lag_means,
            "daily_cv_mean":  daily_cv_mean,
            "lag_cols":       lag_cols,
        }
    return profiles


# ── Step 3: Load August 2025 actuals from Excel ───────────────────────────────
def load_august_actuals() -> dict:
    """
    Returns dict: portfolio -> DataFrame with columns:
        date, day, day_of_week, Call_Volume, CCT, Abandoned_Rate
    """
    result = {}
    for p in PORTFOLIOS:
        sheet = f"{p} - Daily"
        df_d = pd.read_excel(EXCEL_PATH, sheet_name=sheet, header=0)
        df_d.columns = [str(c).strip() for c in df_d.columns]

        # Date column is always first
        date_col = df_d.columns[0]
        df_d = df_d.rename(columns={date_col: "date"})
        df_d["date"] = pd.to_datetime(df_d["date"], errors="coerce")
        df_d = df_d.dropna(subset=["date"])

        # Flexible column renaming
        col_map = {}
        for c in df_d.columns:
            cl = c.lower().replace(" ", "_").replace("-", "_")
            if "call" in cl and "vol" in cl:
                col_map[c] = "Call_Volume"
            elif cl in ("cct", "average_handle_time", "avg_handle_time", "handle_time"):
                col_map[c] = "CCT"
            elif "abandon" in cl and "rate" in cl:
                col_map[c] = "Abandoned_Rate"
            elif "service" in cl and "level" in cl:
                col_map[c] = "Service_Level"
        df_d = df_d.rename(columns=col_map)

        # Clean numerics
        for col in ["Call_Volume", "CCT", "Abandoned_Rate"]:
            if col in df_d.columns:
                if df_d[col].dtype == object:
                    df_d[col] = (df_d[col].astype(str)
                                 .str.replace(",", "", regex=False)
                                 .str.replace("%", "", regex=False)
                                 .str.strip())
                df_d[col] = pd.to_numeric(df_d[col], errors="coerce")

        # Normalise Abandoned_Rate to [0,1]
        if "Abandoned_Rate" in df_d.columns:
            if df_d["Abandoned_Rate"].dropna().max() > 1.5:
                df_d["Abandoned_Rate"] /= 100.0

        # Filter to August 2025
        aug = df_d[(df_d["date"].dt.year == 2025) & (df_d["date"].dt.month == 8)].copy()
        aug["day"]        = aug["date"].dt.day
        aug["day_of_week"] = aug["date"].dt.dayofweek
        result[p] = aug.sort_values("day").reset_index(drop=True)

    return result


def _extrap_linreg(df_dow: pd.DataFrame, col: str) -> float:
    vals  = pd.to_numeric(df_dow[col], errors="coerce").values
    x     = np.arange(len(vals), dtype=float)
    valid = ~np.isnan(vals)
    if valid.sum() < 2:
        return float(vals[valid].mean()) if valid.sum() == 1 else np.nan
    sl, ic, *_ = stats.linregress(x[valid], vals[valid])
    return max(ic + sl * len(vals), 0.0)


def fill_portfolio_d_missing(aug_d: pd.DataFrame) -> pd.DataFrame:
    """Fill Aug 27-31 for Portfolio D using DoW averages + Friday linreg."""
    aug_d = aug_d.copy()

    def dow_avg(days_in_month: list, col: str) -> float:
        return pd.to_numeric(
            aug_d.loc[aug_d["day"].isin(days_in_month), col], errors="coerce"
        ).mean()

    fridays = aug_d[aug_d["day_of_week"] == 4].sort_values("day")
    aug29_cv = _extrap_linreg(fridays, "Call_Volume")

    fill_specs = [
        (27, 2, dow_avg([6, 13, 20],     "Call_Volume")),   # Wednesday
        (28, 3, dow_avg([7, 14, 21],     "Call_Volume")),   # Thursday
        (29, 4, aug29_cv),                                   # Friday linreg
        (30, 5, dow_avg([2, 9, 16, 23],  "Call_Volume")),   # Saturday
        (31, 6, dow_avg([3, 10, 17, 24], "Call_Volume")),   # Sunday
    ]

    for day_num, dow, cv_val in fill_specs:
        # Fill if row is absent OR if Call_Volume is NaN for that day
        if aug_d[(aug_d["day"] == day_num) & aug_d["Call_Volume"].notna()].empty:
            # CCT / ABD: use mean from same DoW in known rows
            cct_fill = (aug_d[aug_d["day_of_week"] == dow]["CCT"].mean()
                        if "CCT" in aug_d.columns else np.nan)
            abd_fill = (aug_d[aug_d["day_of_week"] == dow]["Abandoned_Rate"].mean()
                        if "Abandoned_Rate" in aug_d.columns else np.nan)
            new_row = {
                "date":         pd.Timestamp(f"2025-08-{day_num:02d}"),
                "day":          day_num,
                "day_of_week":  dow,
                "Call_Volume":  cv_val,
                "CCT":          cct_fill,
                "Abandoned_Rate": abd_fill,
            }
            aug_d = pd.concat([aug_d, pd.DataFrame([new_row])], ignore_index=True)

    return aug_d.sort_values("day").reset_index(drop=True)


# ── Step 4: Build inference feature matrix ────────────────────────────────────
def build_inference_features(aug_daily: dict, hist_profiles: dict) -> pd.DataFrame:
    """
    Build 4 × 1488 = 5952 inference rows.
    CV lag/rollmean features are scaled by sf = actual_daily_cv / hist_mean_daily_cv.
    CCT/ABD lag features use historical means directly.
    """
    cv_lag_cols    = [c for c in FEATURE_COLS if ("lag" in c or "rollmean" in c) and "Call_Volume" in c]
    other_lag_cols = [c for c in FEATURE_COLS if ("lag" in c or "rollmean" in c) and "Call_Volume" not in c]

    aug_dates = pd.date_range("2025-08-01", "2025-08-31", freq="D")
    all_rows  = []

    for p in PORTFOLIOS:
        aug  = aug_daily[p].copy()
        prof = hist_profiles[p]
        lag_means     = prof["lag_means"]
        daily_cv_mean = prof["daily_cv_mean"]

        # One-row-per-day lookup (day → Call_Volume)
        daily_lkp = (aug.dropna(subset=["Call_Volume"])
                       .groupby("day").first())

        for dt in aug_dates:
            day_num = dt.day
            dow     = dt.dayofweek

            # Actual daily CV for this portfolio+day
            if day_num in daily_lkp.index:
                actual_cv_day = float(daily_lkp.loc[day_num, "Call_Volume"])
            else:
                actual_cv_day = float(daily_cv_mean.get(dow, daily_cv_mean.mean()))

            hist_mean_cv  = float(daily_cv_mean.get(dow, daily_cv_mean.mean()))
            sf = (actual_cv_day / hist_mean_cv) if hist_mean_cv > 0 else 1.0

            for iv_idx in range(48):
                hour   = iv_idx // 2
                minute = (iv_idx % 2) * 30

                row = {
                    "portfolio":      p,
                    "portfolio_code": PORT_CODE[p],
                    "date":           dt.date(),
                    "timestamp":      dt + pd.Timedelta(hours=hour, minutes=minute),
                    "interval_index": iv_idx,
                    # Time features
                    "hour":           hour,
                    "minute":         minute,
                    "day_of_week":    dow,
                    "day_of_month":   day_num,
                    "month":          8,
                    "is_weekend":     int(dow >= 5),
                    "is_holiday":     int(dt.date() in US_HOLIDAYS_2025),
                    "is_billing_date": int(day_num in [1, 2, 15, 16]),
                    "days_to_billing": min(abs(day_num - 1), abs(day_num - 15), abs(day_num - 31)),
                    # Cyclical
                    "hour_sin":      np.sin(2 * np.pi * hour   / 24),
                    "hour_cos":      np.cos(2 * np.pi * hour   / 24),
                    "dow_sin":       np.sin(2 * np.pi * dow    / 7),
                    "dow_cos":       np.cos(2 * np.pi * dow    / 7),
                    "interval_sin":  np.sin(2 * np.pi * iv_idx / 48),
                    "interval_cos":  np.cos(2 * np.pi * iv_idx / 48),
                    # ToD buckets
                    "is_morning":   int(6  <= hour <= 11),
                    "is_afternoon": int(12 <= hour <= 17),
                    "is_evening":   int(18 <= hour <= 23),
                    "is_night":     int(0  <= hour <= 5),
                    # Anchor
                    "actual_daily_cv": actual_cv_day,
                }

                # Lag/rollmean proxies from historical means
                key = (dow, iv_idx)
                if key in lag_means.index:
                    lag_row = lag_means.loc[key]
                    for col in cv_lag_cols:
                        row[col] = (float(lag_row[col]) * sf
                                    if col in lag_row.index and not np.isnan(lag_row[col])
                                    else 0.0)
                    for col in other_lag_cols:
                        row[col] = (float(lag_row[col])
                                    if col in lag_row.index and not np.isnan(lag_row[col])
                                    else 0.0)
                else:
                    for col in cv_lag_cols + other_lag_cols:
                        row[col] = 0.0

                all_rows.append(row)

    inf_df = pd.DataFrame(all_rows)
    # Safety: fill any remaining NaN in feature cols
    for col in FEATURE_COLS:
        if col in inf_df.columns:
            inf_df[col] = inf_df[col].fillna(0.0)

    return inf_df


# ── Step 5: Optuna tuning + training ─────────────────────────────────────────
def tune_and_train(X: np.ndarray, y: np.ndarray, label: str):
    """Tune XGBoost and LightGBM with Optuna; refit on full data with best params."""
    tscv = TimeSeriesSplit(n_splits=N_CV_FOLDS)

    def objective_xgb(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 300, 1000),
            "max_depth":       trial.suggest_int("max_depth", 4, 8),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha":       trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "objective":       "reg:absoluteerror",
            "tree_method":     "hist",
            "n_jobs":          -1,
            "random_state":    SEED,
        }
        maes = []
        for tr_idx, val_idx in tscv.split(X):
            m = xgb.XGBRegressor(**params)
            m.fit(X[tr_idx], y[tr_idx], verbose=False)
            maes.append(mean_absolute_error(y[val_idx], m.predict(X[val_idx])))
        return np.mean(maes)

    def objective_lgb(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 300, 1000),
            "max_depth":        trial.suggest_int("max_depth", 4, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "objective":        "mae",
            "n_jobs":           -1,
            "random_state":     SEED,
            "verbose":          -1,
        }
        maes = []
        for tr_idx, val_idx in tscv.split(X):
            m = lgb.LGBMRegressor(**params)
            m.fit(X[tr_idx], y[tr_idx])
            maes.append(mean_absolute_error(y[val_idx], m.predict(X[val_idx])))
        return np.mean(maes)

    print(f"  Tuning XGBoost [{label}] ({OPTUNA_TRIALS} trials)...")
    study_xgb = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
    study_xgb.optimize(objective_xgb, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

    print(f"  Tuning LightGBM [{label}] ({OPTUNA_TRIALS} trials)...")
    study_lgb = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
    study_lgb.optimize(objective_lgb, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

    print(f"  {label} — XGB best MAE: {study_xgb.best_value:.4f} | "
          f"LGB best MAE: {study_lgb.best_value:.4f}")

    # Refit on full training set
    best_xgb = {**study_xgb.best_params, "objective": "reg:absoluteerror",
                "tree_method": "hist", "n_jobs": -1, "random_state": SEED}
    best_lgb = {**study_lgb.best_params, "objective": "mae",
                "n_jobs": -1, "random_state": SEED, "verbose": -1}

    xgb_model = xgb.XGBRegressor(**best_xgb)
    xgb_model.fit(X, y, verbose=False)

    lgb_model = lgb.LGBMRegressor(**best_lgb)
    lgb_model.fit(X, y)

    return xgb_model, lgb_model


def predict_ensemble(xm, lm, X: np.ndarray) -> np.ndarray:
    return (xm.predict(X) + lm.predict(X)) / 2.0


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("forecast_direct.py — Direct interval-level ensemble")
    print("=" * 60)

    # ── 1. Training data ──────────────────────────────────────────────────────
    print("\n[1/6] Loading training data...")
    df = load_training_data()
    hist_profiles = build_hist_profiles(df)

    X_all = df[FEATURE_COLS].values.astype(np.float32)
    y_cv  = df["Call_Volume"].values.astype(np.float32)
    y_cct = df["CCT"].values.astype(np.float32)
    y_abd = df["Abandoned_Rate"].values.astype(np.float32)

    # CCT and ABD: train only where Call_Volume > 0
    mask_pos  = y_cv > 0
    X_nonzero = X_all[mask_pos]
    print(f"  Total rows: {len(X_all):,}  |  CV>0 rows (CCT/ABD training): {mask_pos.sum():,}")

    # ── 2. Train CV model ─────────────────────────────────────────────────────
    print("\n[2/6] Training Call_Volume model...")
    xgb_cv, lgb_cv = tune_and_train(X_all, y_cv, "Call_Volume")

    # ── 3. Train CCT model ────────────────────────────────────────────────────
    print("\n[3/6] Training CCT model...")
    xgb_cct, lgb_cct = tune_and_train(X_nonzero, y_cct[mask_pos], "CCT")

    # ── 4. Train ABD model ────────────────────────────────────────────────────
    print("\n[4/6] Training Abandoned_Rate model...")
    xgb_abd, lgb_abd = tune_and_train(X_nonzero, y_abd[mask_pos], "Abandoned_Rate")

    # ── 5. Build August inference features ───────────────────────────────────
    print("\n[5/6] Building August 2025 inference features...")
    aug_daily = load_august_actuals()
    aug_daily["D"] = fill_portfolio_d_missing(aug_daily["D"])

    for p in PORTFOLIOS:
        total_cv = pd.to_numeric(aug_daily[p]["Call_Volume"], errors="coerce").sum()
        print(f"  Portfolio {p}: {len(aug_daily[p])} days  total_CV = {total_cv:,.0f}")

    inf_df = build_inference_features(aug_daily, hist_profiles)
    X_inf  = inf_df[FEATURE_COLS].values.astype(np.float32)
    print(f"  Inference matrix: {X_inf.shape}")

    # ── 6. Predict & post-process ─────────────────────────────────────────────
    print("\n[6/6] Predicting and writing forecast_v20.csv...")

    pred_cv_raw  = predict_ensemble(xgb_cv,  lgb_cv,  X_inf)
    pred_cct_raw = predict_ensemble(xgb_cct, lgb_cct, X_inf)
    pred_abd_raw = predict_ensemble(xgb_abd, lgb_abd, X_inf)

    pred_cv  = np.nan_to_num(np.maximum(pred_cv_raw,  0.0))
    pred_cct = np.nan_to_num(np.where(pred_cv < 0.5, 0.0, np.maximum(pred_cct_raw, 0.0)))
    pred_abd = np.nan_to_num(np.where(pred_cv < 0.5, 0.0, np.clip(pred_abd_raw, 0.0, 1.0)))

    inf_df["pred_cv"]              = pred_cv
    inf_df["pred_cct"]             = pred_cct
    inf_df["pred_abd"]             = pred_abd
    inf_df["pred_abandoned_calls"] = np.round(pred_cv * pred_abd).astype(int)

    # Build output matching template column order
    template = pd.read_csv(TEMPLATE_PATH)
    # Detect identifier columns (non-forecast columns: Month/Day/Interval or Date/Time)
    id_cols = [c for c in template.columns
               if c.lower() in ("date", "time", "month", "day", "interval")]
    out = template.copy()
    for col in out.columns:
        if col not in id_cols:
            out[col] = np.nan

    for p in PORTFOLIOS:
        sub = (inf_df[inf_df["portfolio"] == p]
               .sort_values(["date", "interval_index"])
               .reset_index(drop=True))
        assert len(sub) == 1488, f"Portfolio {p}: got {len(sub)} rows, expected 1488"

        out[f"Calls_Offered_{p}"]   = sub["pred_cv"].values
        out[f"Abandoned_Calls_{p}"] = sub["pred_abandoned_calls"].values
        out[f"Abandoned_Rate_{p}"]  = sub["pred_abd"].values
        out[f"CCT_{p}"]             = sub["pred_cct"].values

    out.to_csv(OUTPUT_PATH, index=False)

    # ── Validation ────────────────────────────────────────────────────────────
    print("\n── Validation ─────────────────────────────────────────────────")
    print(f"  Rows : {len(out)}  (expected 1488)")
    print(f"  NaN  : {out.isnull().sum().sum()}")
    for p in PORTFOLIOS:
        cv_tot  = out[f"Calls_Offered_{p}"].sum()
        neg_cv  = (out[f"Calls_Offered_{p}"] < 0).sum()
        neg_cct = (out[f"CCT_{p}"] < 0).sum()
        bad_abd = ((out[f"Abandoned_Rate_{p}"] < 0) | (out[f"Abandoned_Rate_{p}"] > 1)).sum()
        print(f"  {p}: total_CV={cv_tot:,.0f}  neg_CV={neg_cv}  neg_CCT={neg_cct}  bad_ABD={bad_abd}")

    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
