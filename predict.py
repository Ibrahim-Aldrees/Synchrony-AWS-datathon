import subprocess
import sys

# Install dependencies not present in the base container
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'holidays==0.45'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'lightgbm==4.3.0'])

import argparse
import os
import glob
import tarfile
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
import holidays


# ── Mirrors preprocess_time from train.py exactly ──────────────────────────
def preprocess_time(df):
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df = df.copy()
    df['month_idx'] = df['Month'].map(month_map)

    # Normalize time strings that lack seconds (e.g. '9:30' -> '9:30:00')
    df['Interval_normalized'] = df['Interval'].apply(
        lambda x: x if str(x).count(':') == 2 else str(x) + ':00'
    )
    df['time_obj'] = pd.to_datetime(df['Interval_normalized'], format='%H:%M:%S')
    df['hour'] = df['time_obj'].dt.hour
    df['minute'] = df['time_obj'].dt.minute

    df['interval_sin'] = np.sin(2 * np.pi * (df['hour'] + df['minute'] / 60) / 24)
    df['interval_cos'] = np.cos(2 * np.pi * (df['hour'] + df['minute'] / 60) / 24)

    us_holidays = holidays.US()
    df['is_holiday'] = df.apply(
        lambda x: 1 if datetime(2024, int(x['month_idx']), int(x['Day'])) in us_holidays else 0,
        axis=1
    )
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir',  type=str, default='/opt/ml/processing/input')
    parser.add_argument('--model-dir',  type=str, default='/opt/ml/processing/model')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/processing/output')
    args = parser.parse_args()

    # ── 1. Extract model artifacts ──────────────────────────────────────────
    # ScriptProcessor does not auto-extract .tar.gz unlike Training jobs
    print("Files in model dir (before extraction):", glob.glob(args.model_dir + '/**/*', recursive=True))

    for tar_path in glob.glob(args.model_dir + '/*.tar.gz'):
        print(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=args.model_dir)
        print("Extraction complete.")

    print("Files in model dir (after extraction):", glob.glob(args.model_dir + '/**/*', recursive=True))

    # ── 2. Load per-portfolio models ────────────────────────────────────────
    # Expected filenames: model_{target}_{portfolio}.joblib
    # e.g. model_calls_a.joblib, model_cct_b.joblib, model_abandoned_rate_c.joblib
    # Structure: { (col_prefix, portfolio) : model }
    portfolios = ['A', 'B', 'C', 'D']

    target_file_map = {
        'calls':          'Calls_Offered',
        'cct':            'CCT',
        'abandoned_rate': 'Abandoned_Rate',
    }

    loaded_models = {}
    for file_prefix, col_prefix in target_file_map.items():
        for portfolio in portfolios:
            filename = f"model_{file_prefix}_{portfolio.lower()}.joblib"
            model_path = os.path.join(args.model_dir, filename)
            if os.path.exists(model_path):
                loaded_models[(col_prefix, portfolio)] = joblib.load(model_path)
                print(f"Loaded: {filename}")
            else:
                print(f"WARNING: {filename} not found — {col_prefix}_{portfolio} will be NaN")

    # ── 3. Load template ────────────────────────────────────────────────────
    template_path = os.path.join(args.input_dir, 'template_forecast_v00.csv')
    df = pd.read_csv(template_path)
    print(f"\nTemplate loaded: {df.shape[0]} rows, columns: {list(df.columns)}")

    # ── 4. Load and merge staffing ──────────────────────────────────────────
    # Fix: correct filename is staffing_filled.csv
    # Fix: date format is YYYY-MM-DD, id_vars is 'date'
    staffing = pd.read_csv(os.path.join(args.input_dir, "staffing_filled.csv"))
    staffing = staffing.melt(id_vars='date', var_name='Portfolio', value_name='Staffing')
    staffing['date_parsed'] = pd.to_datetime(staffing['date'], format='%Y-%m-%d')
    staffing['Month'] = staffing['date_parsed'].dt.strftime('%B')
    staffing['Day']   = staffing['date_parsed'].dt.day
    staffing = staffing[['Month', 'Day', 'Portfolio', 'Staffing']]

    # Expand template to one row per date+interval+portfolio so staffing
    # can be joined, then used as a feature per portfolio model
    df_expanded = pd.DataFrame()
    for p in portfolios:
        temp = df[['Month', 'Day', 'Interval']].copy()
        temp['Portfolio'] = p
        df_expanded = pd.concat([df_expanded, temp], ignore_index=True)

    df_expanded = df_expanded.merge(staffing, on=['Month', 'Day', 'Portfolio'], how='left')
    print(f"Staffing NaN count after merge: {df_expanded['Staffing'].isna().sum()}")

    # ── 5. Preprocess features ──────────────────────────────────────────────
    df_expanded = preprocess_time(df_expanded)
    base_features = ['month_idx', 'Day', 'interval_sin', 'interval_cos', 'is_holiday', 'Staffing']

    # ── 6. Run inference for each portfolio ────────────────────────────────
    for portfolio in portfolios:
        print(f"\nRunning inference for Portfolio {portfolio}...")

        # Filter to this portfolio's rows — preserves same row order as template
        X = df_expanded[df_expanded['Portfolio'] == portfolio][base_features].copy()

        for (col_prefix, p), model in loaded_models.items():
            if p != portfolio:
                continue
            col_name = f"{col_prefix}_{portfolio}"
            predictions = np.clip(model.predict(X), 0, None)

            # predictions is a numpy array — assign positionally to df
            df[col_name] = predictions
            print(f"  Filled '{col_name}': min={predictions.min():.2f}, "
                  f"max={predictions.max():.2f}, mean={predictions.mean():.2f}")

        # Derive Abandoned_Calls = Abandoned_Rate × Calls_Offered
        # (no separate model needed — formula-derived per the grading spec)
        offered_col = f"Calls_Offered_{portfolio}"
        rate_col    = f"Abandoned_Rate_{portfolio}"
        abandon_col = f"Abandoned_Calls_{portfolio}"

        if offered_col in df.columns and rate_col in df.columns:
            if not df[offered_col].isna().all() and not df[rate_col].isna().all():
                df[abandon_col] = np.round(df[offered_col] * df[rate_col], 0).astype('Int64')
                print(f"  Derived '{abandon_col}' = Calls_Offered × Abandoned_Rate")

    # ── 7. Save output — same columns and order as the template ────────────
    output_columns = [
        'Month', 'Day', 'Interval',
        'Calls_Offered_A', 'Abandoned_Calls_A', 'Abandoned_Rate_A', 'CCT_A',
        'Calls_Offered_B', 'Abandoned_Calls_B', 'Abandoned_Rate_B', 'CCT_B',
        'Calls_Offered_C', 'Abandoned_Calls_C', 'Abandoned_Rate_C', 'CCT_C',
        'Calls_Offered_D', 'Abandoned_Calls_D', 'Abandoned_Rate_D', 'CCT_D',
    ]

    # Warn about any columns that are still entirely NaN (missing models)
    output_df = df[output_columns]
    for col in output_columns[3:]:  # skip Month, Day, Interval
        if output_df[col].isna().all():
            print(f"WARNING: Column '{col}' is entirely NaN — corresponding model was not found.")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'forecast_output.csv')
    output_df.to_csv(output_path, index=False)
    print(f"\nOutput saved to {output_path} — shape: {output_df.shape}")
