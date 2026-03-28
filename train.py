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

    # Fix: normalize time strings that lack seconds (e.g. '9:30' -> '9:30:00')
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
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.train, "combined_data.csv"))
    df = df.dropna(subset=['Call Volume', 'Portfolio', 'Month', 'Day', 'Interval'])
    df['Day'] = pd.to_numeric(df['Day'], errors='coerce')
    df = df.dropna(subset=['Day'])

    df = preprocess_time(df)
    df['Portfolio'] = df['Portfolio'].astype('category')

    features = ['month_idx', 'Day', 'interval_sin', 'interval_cos', 'is_holiday', 'Portfolio']
    X = df[features]
    df['Abandoned Rate'] = df['Abandoned Rate'].astype(float)
    y = df['Abandoned Rate'].values
    

    print(f"Training shape: {X.shape}")

    train_data = lgb.Dataset(X, label=y, categorical_feature=['Portfolio'])
    params = {
        'objective': 'tweedie',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt'
    }
    model = lgb.train(params, train_data, num_boost_round=100)
    joblib.dump(model, os.path.join(args.model_dir, "model_abandoned_rate.joblib"))
    print("Model saved successfully.")