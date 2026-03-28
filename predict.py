import subprocess
import sys

# Install dependencies not present in the base container
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'holidays==0.45'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'lightgbm==4.3.0'])

import argparse
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import holidays
import glob
import tarfile



# ── Mirrors preprocess_time from train.py exactly ──────────────────────────
def preprocess_time(df):
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df = df.copy()
    df['month_idx'] = df['Month'].map(month_map)

    # Same normalization fix as train.py — handles 'H:MM' with no seconds
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

    # ── 1. Load template ────────────────────────────────────────────────────
    template_path = os.path.join(args.input_dir, 'template_forecast_v00.csv')
    df = pd.read_csv(template_path)
    print(f"Template loaded: {df.shape[0]} rows, columns: {list(df.columns)}")

    # ── 2. Preprocess shared features ──────────────────────────────────────
    df = preprocess_time(df)
    base_features = ['month_idx', 'Day', 'interval_sin', 'interval_cos', 'is_holiday']

    # ── 3. Load available models ────────────────────────────────────────────
    # Map: model filename → which output column prefix it fills
    # Train separate models and save with these names to enable CCT / abandon predictions
    model_registry = {
        'model_calls.joblib':         'Calls_Offered',   # trained in train.py
        'model_cct.joblib':     'CCT',             # optional — train separately
        'model_abandoned_rate.joblib': 'Abandoned_Rate',  # optional — train separately
    }

    loaded_models = {}
    print("Files in model dir:", glob.glob(args.model_dir + '/**/*', recursive=True))

    for tar_path in glob.glob(args.model_dir + '/*.tar.gz'):
        print(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=args.model_dir)
        print("Extraction complete.")

    print("Files after extraction:", glob.glob(args.model_dir + '/**/*', recursive=True))

    for filename, target_prefix in model_registry.items():
        model_path = os.path.join(args.model_dir, filename)
        if os.path.exists(model_path):
            loaded_models[target_prefix] = joblib.load(model_path)
            print(f"Loaded model for '{target_prefix}' from {filename}")
        else:
            print(f"WARNING: No model found for '{target_prefix}' ({filename} missing) — columns will remain NaN")

    # ── 4. Predict for each portfolio (A, B, C, D) ─────────────────────────
    portfolios = ['A', 'B', 'C', 'D']

    for portfolio in portfolios:
        print(f"Running inference for Portfolio {portfolio}...")

        # Build feature matrix with this portfolio set for every row
        X = df[base_features].copy()
        X['Portfolio'] = pd.Categorical([portfolio] * len(df), categories=portfolios)

        for target_prefix, model in loaded_models.items():
            col_name = f"{target_prefix}_{portfolio}"
            predictions = model.predict(X)

            # Predictions should be non-negative counts / rates
            predictions = np.clip(predictions, 0, None)

            df[col_name] = np.round(predictions, 4)
            print(f"  Filled '{col_name}': min={predictions.min():.2f}, max={predictions.max():.2f}")

        # Derive Abandoned_Calls from rate × offered (only if both are available)
        offered_col  = f"Calls_Offered_{portfolio}"
        rate_col     = f"Abandoned_Rate_{portfolio}"
        abandon_col  = f"Abandoned_Calls_{portfolio}"

        if offered_col in df.columns and rate_col in df.columns:
            if not df[offered_col].isna().all() and not df[rate_col].isna().all():
                df[abandon_col] = np.round(df[offered_col] * df[rate_col], 0).astype('Int64')
                print(f"  Derived '{abandon_col}' from Calls_Offered × Abandoned_Rate")

    # ── 5. Save output — same columns and shape as the template ────────────
    original_columns = [
        'Month', 'Day', 'Interval',
        'Calls_Offered_A', 'Abandoned_Calls_A', 'Abandoned_Rate_A', 'CCT_A',
        'Calls_Offered_B', 'Abandoned_Calls_B', 'Abandoned_Rate_B', 'CCT_B',
        'Calls_Offered_C', 'Abandoned_Calls_C', 'Abandoned_Rate_C', 'CCT_C',
        'Calls_Offered_D', 'Abandoned_Calls_D', 'Abandoned_Rate_D', 'CCT_D',
    ]

    output_df = df[original_columns]  # enforces exact column order and set
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'forecast_output.csv')
    output_df.to_csv(output_path, index=False)
    print(f"\nOutput saved to {output_path} — shape: {output_df.shape}")