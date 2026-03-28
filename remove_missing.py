import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import statsmodels.formula.api as sm


# Load the raw interval data
df = pd.read_csv("Data/Staffing.csv")

# Optional: inspect the first few rows
print(df.head())
print(df.columns)

df = df.dropna()

# Make sure numeric columns are actual integers and not strings
numeric_cols = ["call_volume", "cct", "abandon_rate"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows that became NaN after conversion
df = df.dropna(subset=numeric_cols).copy()

df.to_csv("Staffing_data_preprocessed.csv", index=False)