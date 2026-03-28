import pandas as pd

def add_portfolio_staffing(call_df, staffing_df, portfolio_col_name):
    call_df = call_df.copy()
    staffing_df = staffing_df.copy()

    # Ensure date columns exist and are datetime
    call_df["date"] = pd.to_datetime(call_df["timestamp"]).dt.normalize()
    staffing_df["date"] = pd.to_datetime(staffing_df["date"]).dt.normalize()

    # Keep only the relevant staffing column
    staffing_subset = staffing_df[["date", portfolio_col_name]].copy()
    staffing_subset = staffing_subset.rename(columns={portfolio_col_name: "staffing"})

    # Merge daily staffing onto interval data
    merged = call_df.merge(staffing_subset, on="date", how="left")

    return merged

df_A = pd.read_csv("Data/A-preprocessed.csv")
df_B = pd.read_csv("Data/B-preprocessed.csv")
df_C = pd.read_csv("Data/C-preprocessed.csv")
df_D = pd.read_csv("Data/D-preprocessed.csv")

staffing_df = pd.read_csv("Data/staffing_filled.csv")

df_A = add_portfolio_staffing(df_A, staffing_df, "A")
df_B = add_portfolio_staffing(df_B, staffing_df, "B")
df_C = add_portfolio_staffing(df_C, staffing_df, "C")
df_D = add_portfolio_staffing(df_D, staffing_df, "D")

df_A.to_csv("Data/A-preprocessed-staffing.csv", index=False)
df_B.to_csv("Data/B-preprocessed-staffing.csv", index=False)
df_C.to_csv("Data/C-preprocessed-staffing.csv", index=False)
df_D.to_csv("Data/D-preprocessed-staffing.csv", index=False)