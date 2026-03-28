import pandas as pd
import numpy as np

def preprocess_interval_data(csv_path, output_path="interval_data_preprocessed.csv"):
    df = pd.read_csv(csv_path)

    critical_cols = ["portfolio", "date", "time", "call_volume", "cct", "abandon_rate"]
    df = df.dropna(subset=critical_cols).copy()

    for col in ["call_volume", "cct", "abandon_rate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["call_volume", "cct", "abandon_rate"]).copy()
    df = df[(df["call_volume"] >= 0) & (df["cct"] >= 0) & (df["abandon_rate"] >= 0)].copy()
    df["abandon_rate"] = df["abandon_rate"].clip(0, 1)
    df = df.drop_duplicates().copy()

    df["timestamp"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        errors="coerce"
    )
    df = df.dropna(subset=["timestamp"]).copy()

    df["timestamp"] = df["timestamp"].dt.tz_localize(
        "America/New_York",
        ambiguous="NaT",
        nonexistent="shift_forward"
    )
    df = df.dropna(subset=["timestamp"]).copy()

    df = df.sort_values(["portfolio", "timestamp"]).reset_index(drop=True)

    portfolios = df["portfolio"].dropna().unique()
    start_time = df["timestamp"].min().floor("30min")
    end_time = df["timestamp"].max().ceil("30min")

    full_time_index = pd.date_range(
        start=start_time,
        end=end_time,
        freq="30min",
        tz=df["timestamp"].dt.tz
    )

    full_index = pd.MultiIndex.from_product(
        [portfolios, full_time_index],
        names=["portfolio", "timestamp"]
    )

    df = df.set_index(["portfolio", "timestamp"]).reindex(full_index).reset_index()

    def fill_by_portfolio(group):
        group = group.sort_values("timestamp").set_index("timestamp")
        group["call_volume"] = group["call_volume"].interpolate(method="time").ffill().bfill()
        group["cct"] = group["cct"].interpolate(method="time").ffill().bfill()
        group["abandon_rate"] = group["abandon_rate"].interpolate(method="time").ffill().bfill()
        return group.reset_index()

    df = df.groupby("portfolio", group_keys=False).apply(fill_by_portfolio).reset_index(drop=True)

    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["day_of_month"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["interval_index"] = df["hour"] * 2 + (df["minute"] // 30)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["interval_sin"] = np.sin(2 * np.pi * df["interval_index"] / 48)
    df["interval_cos"] = np.cos(2 * np.pi * df["interval_index"] / 48)

    lag_steps = {"lag_1": 1, "lag_2": 2, "lag_48": 48, "lag_336": 336}
    target_cols = ["call_volume", "cct", "abandon_rate"]

    for feature_name, step in lag_steps.items():
        for col in target_cols:
            df[f"{col}_{feature_name}"] = df.groupby("portfolio")[col].shift(step)

    for col in target_cols:
        df[f"{col}_rollmean_4"] = (
            df.groupby("portfolio")[col]
              .transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
        )
        df[f"{col}_rollmean_48"] = (
            df.groupby("portfolio")[col]
              .transform(lambda s: s.shift(1).rolling(48, min_periods=1).mean())
        )
        df[f"{col}_rollstd_48"] = (
            df.groupby("portfolio")[col]
              .transform(lambda s: s.shift(1).rolling(48, min_periods=1).std())
        )

    df["portfolio_code"] = df["portfolio"].astype("category").cat.codes

    df["is_morning"] = df["hour"].between(6, 11).astype(int)
    df["is_afternoon"] = df["hour"].between(12, 17).astype(int)
    df["is_evening"] = df["hour"].between(18, 23).astype(int)
    df["is_night"] = df["hour"].between(0, 5).astype(int)

    df["date_only"] = df["timestamp"].dt.date
    us_holidays = pd.to_datetime([
        "2024-01-01", "2024-07-04", "2024-11-28", "2024-12-25"
    ]).date
    df["is_holiday"] = df["date_only"].isin(us_holidays).astype(int)

    required_feature_cols = [
        "call_volume_lag_1", "call_volume_lag_48", "call_volume_lag_336",
        "cct_lag_1", "abandon_rate_lag_1"
    ]
    df = df.dropna(subset=required_feature_cols).copy()

    df["call_volume"] = df["call_volume"].clip(lower=0)
    df["cct"] = df["cct"].clip(lower=0)
    df["abandon_rate"] = df["abandon_rate"].clip(0, 1)

    df.to_csv(output_path, index=False)
    return df

# Example usage
processed_df = preprocess_interval_data("interval_data.csv")
print(processed_df.head())