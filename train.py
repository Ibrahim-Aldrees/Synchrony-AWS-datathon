import argparse
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from datetime import datetime
import holidays

def preprocess_time(df):
    month_map = {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6,
                 'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12}
    df['month_idx'] = df['Month'].map(month_map)
    df['Interval_normalized'] = df['Interval'].apply(
        lambda x: x if str(x).count(':') == 2 else str(x) + ':00'
    )
    df['time_obj'] = pd.to_datetime(df['Interval_normalized'], format='%H:%M:%S')
    df['hour'] = df['time_obj'].dt.hour
    df['minute'] = df['time_obj'].dt.minute
    df['interval_sin'] = np.sin(2 * np.pi * (df['hour'] + df['minute']/60) / 24)
    df['interval_cos'] = np.cos(2 * np.pi * (df['hour'] + df['minute']/60) / 24)
    us_holidays = holidays.US()
    df['is_holiday'] = df.apply(
        lambda x: 1 if datetime(2024, x['month_idx'], int(x['Day'])) in us_holidays else 0, axis=1
    )
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--target', type=str, default='Call Volume',
                        help='Target column: "Call Volume", "CCT", or "Abandoned Rate"')
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.train, "combined_data.csv"))

    # ── Load and merge staffing ─────────────────────────────────────────────
    # staffing_filled.csv is wide format: columns = date, A, B, C, D
    staffing = pd.read_csv(os.path.join(args.train, "staffing_filled.csv"))
    staffing = staffing.melt(id_vars='date', var_name='Portfolio', value_name='Staffing')

    # Fix: date format is YYYY-MM-DD
    staffing['date_parsed'] = pd.to_datetime(staffing['date'], format='%Y-%m-%d')
    staffing['Month'] = staffing['date_parsed'].dt.strftime('%B')  # e.g. 'August'
    staffing['Day']   = staffing['date_parsed'].dt.day
    staffing = staffing[['Month', 'Day', 'Portfolio', 'Staffing']]

    df = df.merge(staffing, on=['Month', 'Day', 'Portfolio'], how='left')
    print(f"Staffing NaN count after merge: {df['Staffing'].isna().sum()}")
    # ───────────────────────────────────────────────────────────────────────

    df = df.dropna(subset=[args.target, 'Portfolio', 'Month', 'Day', 'Interval'])
    df['Day'] = pd.to_numeric(df['Day'], errors='coerce')
    df = df.dropna(subset=['Day'])
    df = preprocess_time(df)

    # Features no longer include Portfolio — each model IS a portfolio
    features = ['month_idx', 'Day', 'interval_sin', 'interval_cos', 'is_holiday', 'Staffing']

    # Map target name to a safe filename prefix
    target_file_map = {
        'Call Volume':    'calls',
        'CCT':            'cct',
        'Abandoned Rate': 'abandoned_rate',
    }
    prefix = target_file_map.get(args.target, 'model')

    params = {
        'objective': 'tweedie',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt'
    }

    for portfolio in ['A', 'B', 'C', 'D']:
        df_p = df[df['Portfolio'] == portfolio]
        if df_p.empty:
            print(f"WARNING: No data for Portfolio {portfolio}, skipping.")
            continue

        X = df_p[features]
        y = df_p[args.target].astype(float).values
        print(f"Portfolio {portfolio} — training on {len(df_p)} rows, "
              f"target mean={y.mean():.2f}, max={y.max():.2f}")

        train_data = lgb.Dataset(X, label=y)
        model = lgb.train(params, train_data, num_boost_round=100)

        filename = f"model_{prefix}_{portfolio.lower()}.joblib"
        joblib.dump(model, os.path.join(args.model_dir, filename))
        print(f"Saved: {filename}")
