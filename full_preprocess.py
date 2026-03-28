import pandas as pd
import numpy as np

df = pd.read_csv("Data/D-interval.csv")

# =========================
# 1. Clean column types
# =========================
numeric_cols = [
    "Service Level",
    "Call Volume",
    "Abandoned Calls",
    "Abandoned Rate",
    "CCT"
]


df['CCT'] = df['CCT'].str.replace(',', '').astype(float)
df['Service Level'] = df['Service Level'].str.replace('%', '').astype(float)
df['Abandoned Rate'] = df['Abandoned Rate'].str.replace('%', '').astype(float)

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")



df["Month"] = df["Month"].astype(str).str.strip()
df["Day"] = pd.to_numeric(df["Day"], errors="coerce")
df["Interval"] = df["Interval"].astype(str).str.strip()

# Drop only rows missing the pieces needed to form time
df = df.dropna(subset=["Month", "Day", "Interval"]).copy()


# Remove bad Interval strings if they exist
df = df[(df["Interval"] != "") & (df["Interval"].str.lower() != "nan")].copy()
#no
# =========================
# 2. Build timestamp
# =========================
YEAR = 2025

df["timestamp"] = pd.to_datetime(
    df["Month"] + " " + df["Day"].astype(int).astype(str) + " " + str(YEAR) + " " + df["Interval"],
    errors="coerce"
)




df = df.dropna(subset=["timestamp"]).copy()

print("timessss")

if df.empty:
    raise ValueError("No valid timestamps were created.")

# =========================
# 3. Sort chronologically
# =========================
df = df.sort_values("timestamp").reset_index(drop=True)

# =========================
# 4. Enforce 30-minute continuity
# =========================
full_range = pd.date_range(
    start=df["timestamp"].min(),
    end=df["timestamp"].max(),
    freq="30min"
)

df = (
    df.set_index("timestamp")
      .reindex(full_range)
      .reset_index()
      .rename(columns={"index": "timestamp"})
)

# =========================
# 5. Fill values after reindexing
# =========================

# Keep time columns aligned with timestamp
df["Month"] = df["timestamp"].dt.month_name()
df["Day"] = df["timestamp"].dt.day
df["Interval"] = df["timestamp"].dt.strftime("%-H:%M")

# Counts default to 0 for missing intervals
df["Call Volume"] = df["Call Volume"].fillna(0)
df["Abandoned Calls"] = df["Abandoned Calls"].fillna(0)

# Recompute abandon rate from counts
df["Abandoned Rate"] = np.where(
    df["Call Volume"] > 0,
    df["Abandoned Calls"] / df["Call Volume"],
    0
)

# CCT: forward fill, then fallback to 0 if needed at the very beginning
df["CCT"] = df["CCT"].ffill().fillna(0)

# Service Level is optional for baseline; keep but don't rely on it
df["Service Level"] = df["Service Level"].ffill()

# =========================
# 6. Basic time features
# =========================
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month_num"] = df["timestamp"].dt.month
df["interval_index"] = df["timestamp"].dt.hour * 2 + (df["timestamp"].dt.minute // 30)

# =========================
# 7. Essential lag features
# =========================
for lag in [1, 2, 48, 96, 336]:
    df[f"cv_lag_{lag}"] = df["Call Volume"].shift(lag)
    df[f"abd_lag_{lag}"] = df["Abandoned Rate"].shift(lag)
    df[f"cct_lag_{lag}"] = df["CCT"].shift(lag)

# =========================
# 8. Drop early rows without lag history
# =========================
df = df.dropna().reset_index(drop=True)

print(df.head())
print(df.shape)


# Rolling Features 
df["cv_roll_mean_3"] = df["Call Volume"].shift(1).rolling(3).mean()
df["cv_roll_mean_48"] = df["Call Volume"].shift(1).rolling(48).mean()

df["abd_roll_mean_3"] = df["Abandoned Rate"].shift(1).rolling(3).mean()
df["cct_roll_mean_3"] = df["CCT"].shift(1).rolling(3).mean()

# clip features 
df["Abandoned Rate"] = df["Abandoned Rate"] / 100
df["Abandoned Rate"] = df["Abandoned Rate"].clip(0, 1)

# check if it's weekend
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)


# Add staffing data
df["date"] = df["timestamp"].dt.date
df["date"] = pd.to_datetime(df["date"])

# Produce abandoned ration because they asked us

df.to_csv("Data/D-preprocessed.csv", index=False)