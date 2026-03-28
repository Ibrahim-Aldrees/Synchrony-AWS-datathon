"""
preprocess.py — Clean preprocessing for all 4 portfolios.
Reads from Excel (daily data) and raw interval CSVs.
Outputs: Data/all_portfolios_preprocessed.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

EXCEL_PATH = Path("/Users/falmug/Downloads/DATATHON/Data/Data for Datathon (Revised).xlsx")
DATA_DIR = Path("/Users/falmug/Downloads/DATATHON/Data")
PORTFOLIOS = ["A", "B", "C", "D"]
YEAR = 2024  # All interval data is Apr–Jun 2024

MONTH_MAP = {"April": 4, "May": 5, "June": 6}

US_HOLIDAYS_2024 = pd.to_datetime([
    "2024-01-01",  # New Year's
    "2024-01-15",  # MLK Day
    "2024-02-19",  # Presidents Day
    "2024-05-27",  # Memorial Day
    "2024-07-04",  # Independence Day
    "2024-09-02",  # Labor Day
    "2024-10-14",  # Columbus Day
    "2024-11-11",  # Veterans Day
    "2024-11-28",  # Thanksgiving
    "2024-11-29",  # Black Friday
    "2024-12-25",  # Christmas
]).date


def load_interval_from_excel(portfolio: str) -> pd.DataFrame:
    """Load interval data from Excel sheet for a given portfolio."""
    sheet = f"{portfolio} - Interval"
    df = pd.read_excel(EXCEL_PATH, sheet_name=sheet, header=0)
    df.columns = ["Month", "Day", "Interval", "Service_Level", "Call_Volume",
                  "Abandoned_Calls", "Abandoned_Rate", "CCT"]
    df["portfolio"] = portfolio
    return df


def build_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Month/Day/Interval to a proper datetime timestamp."""
    df = df.copy()
    df["month_num"] = df["Month"].map(MONTH_MAP)

    # Interval can be datetime.time (from Excel) or string "H:MM"
    def parse_interval(v):
        if pd.isna(v) if not hasattr(v, "hour") else False:
            return np.nan, np.nan
        if hasattr(v, "hour"):
            return v.hour, v.minute
        parts = str(v).split(":")
        return int(parts[0]), int(parts[1])

    df[["iv_hour", "iv_min"]] = df["Interval"].apply(
        lambda v: pd.Series(parse_interval(v))
    )

    df["timestamp"] = pd.to_datetime({
        "year": YEAR,
        "month": df["month_num"],
        "day": df["Day"],
        "hour": df["iv_hour"],
        "minute": df["iv_min"],
    })
    return df


def reindex_to_grid(df: pd.DataFrame, portfolio: str) -> pd.DataFrame:
    """Reindex to full 30-min grid, handle overnight zeros, forward-fill CCT within day only."""
    # Drop rows with NaT timestamps
    df = df.dropna(subset=["timestamp"])
    # Aggregate duplicate timestamps (same interval recorded twice) by taking mean
    numeric_cols = ["Call_Volume", "Abandoned_Calls", "Abandoned_Rate", "CCT", "Service_Level"]
    agg_dict = {c: "mean" for c in numeric_cols if c in df.columns}
    df = df.groupby("timestamp").agg(agg_dict).reset_index()
    df = df.set_index("timestamp").sort_index()

    start = df.index.min().normalize()
    end = df.index.max().normalize() + pd.Timedelta(hours=23, minutes=30)
    full_index = pd.date_range(start=start, end=end, freq="30min")

    df = df.reindex(full_index)
    df["portfolio"] = portfolio

    # Fill call volume with 0 where truly missing (overnight)
    df["Call_Volume"] = df["Call_Volume"].fillna(0)
    df["Abandoned_Calls"] = df["Abandoned_Calls"].fillna(0)

    # Forward-fill CCT and Service_Level within each day only (reset at midnight)
    df["date"] = df.index.date
    df["CCT"] = df.groupby("date")["CCT"].transform(lambda s: s.ffill())
    df["Service_Level"] = df.groupby("date")["Service_Level"].transform(lambda s: s.ffill())

    # Any remaining NaN (no prior value in that day) — fill with 0
    df["CCT"] = df["CCT"].fillna(0)
    df["Service_Level"] = df["Service_Level"].fillna(0)

    # CCT and Service_Level are meaningless at zero-volume intervals — force to 0
    df["CCT"] = np.where(df["Call_Volume"] == 0, 0.0, df["CCT"])
    df["Service_Level"] = np.where(df["Call_Volume"] == 0, 0.0, df["Service_Level"])

    # Recompute Abandoned Rate
    df["Abandoned_Rate"] = np.where(
        df["Call_Volume"] > 0,
        df["Abandoned_Calls"] / df["Call_Volume"],
        0.0
    )

    df = df.drop(columns=["date"])
    df.index.name = "timestamp"
    df = df.reset_index()
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time, cyclical, and holiday features."""
    ts = df["timestamp"]

    df["hour"] = ts.dt.hour
    df["minute"] = ts.dt.minute
    df["day_of_week"] = ts.dt.dayofweek       # 0=Mon, 6=Sun
    df["day_of_month"] = ts.dt.day
    df["month"] = ts.dt.month
    df["interval_index"] = df["hour"] * 2 + (df["minute"] // 30)  # 0–47
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["interval_sin"] = np.sin(2 * np.pi * df["interval_index"] / 48)
    df["interval_cos"] = np.cos(2 * np.pi * df["interval_index"] / 48)

    # Time-of-day buckets
    df["is_morning"] = df["hour"].between(6, 11).astype(int)
    df["is_afternoon"] = df["hour"].between(12, 17).astype(int)
    df["is_evening"] = df["hour"].between(18, 23).astype(int)
    df["is_night"] = df["hour"].between(0, 5).astype(int)

    df["is_holiday"] = ts.dt.date.isin(US_HOLIDAYS_2024).astype(int)

    # Billing cycle features (retail credit card — spikes around statement/payment dates)
    df["is_billing_date"] = df["day_of_month"].isin([1, 2, 15, 16]).astype(int)
    df["days_to_billing"] = df["day_of_month"].apply(
        lambda d: min(abs(d - 1), abs(d - 15), abs(d - 31))
    )

    # Service_Level is already a raw column in the DataFrame from reindex_to_grid.
    # It is kept as-is here — zero at zero-volume intervals, forward-filled within day.
    # For August forecasting, historical mean per (portfolio, day_of_week, interval_index)
    # is used as a proxy since actual August service levels are unknown at forecast time.

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling features grouped by portfolio — no cross-portfolio leakage."""
    df = df.sort_values(["portfolio", "timestamp"]).reset_index(drop=True)

    # lag_672 = 2 weeks back, lag_2016 = 6 weeks back
    # interval data spans 91 days so lag_2016 has ~49 days of valid rows — acceptable
    for lag in [1, 2, 48, 336, 672, 2016]:
        for col in ["Call_Volume", "CCT", "Abandoned_Rate"]:
            df[f"{col}_lag_{lag}"] = df.groupby("portfolio")[col].shift(lag)

    for col in ["Call_Volume", "CCT", "Abandoned_Rate"]:
        df[f"{col}_rollmean_4"] = df.groupby("portfolio")[col].transform(
            lambda s: s.shift(1).rolling(4, min_periods=1).mean()
        )
        df[f"{col}_rollmean_48"] = df.groupby("portfolio")[col].transform(
            lambda s: s.shift(1).rolling(48, min_periods=1).mean()
        )

    return df


def main():
    all_dfs = []
    for p in PORTFOLIOS:
        print(f"Processing portfolio {p}...")
        df = load_interval_from_excel(p)
        df = build_timestamp(df)
        df = reindex_to_grid(df, p)
        df = add_features(df)
        all_dfs.append(df)
        print(f"  {len(df)} rows")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = add_lag_features(combined)

    out_path = DATA_DIR / "all_portfolios_preprocessed.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved {len(combined)} rows to {out_path}")
    print(combined.dtypes)
    print(combined.head(3))


if __name__ == "__main__":
    main()
