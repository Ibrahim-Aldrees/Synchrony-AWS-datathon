import pandas as pd
import numpy as np

def preprocess_staffing(input_path: str, output_path: str = "staffing_filled.csv") -> pd.DataFrame:
    # Load file
    df = pd.read_csv(input_path)

    # Rename date column
    df = df.rename(columns={"Unnamed: 0": "date"})

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop rows with invalid dates
    df = df.dropna(subset=["date"]).copy()

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    # Staffing columns
    staffing_cols = ["A", "B", "C", "D"]

    # Convert staffing columns to numeric
    for col in staffing_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build a full daily calendar in case any dates are missing
    full_dates = pd.date_range(start=df["date"].min(), end=df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full_dates).reset_index().rename(columns={"index": "date"})

    # Add calendar features for filling logic
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # Fill each staffing column
    for col in staffing_cols:
        # 1) Fill from same day-of-week median
        dow_median = df.groupby("day_of_week")[col].transform("median")
        df[col] = df[col].fillna(dow_median)

        # 2) Time interpolation for local trends
        df[col] = df[col].interpolate(method="linear", limit_direction="both")

        # 3) Final safety fill
        df[col] = df[col].ffill().bfill()

        # 4) Staffing should not be negative
        df[col] = df[col].clip(lower=0)

        # 5) Round to whole staff counts
        df[col] = df[col].round().astype(int)

    # Drop helper columns
    df = df.drop(columns=["day_of_week", "month"])

    # Save
    df.to_csv(output_path, index=False)

    return df


# Example usage
cleaned_df = preprocess_staffing("Data/Staffing.csv", "staffing_filled.csv")
print(cleaned_df.head())
print(cleaned_df.isna().sum())
