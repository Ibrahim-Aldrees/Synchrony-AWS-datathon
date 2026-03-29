import pandas as pd
import numpy as np

df_A = pd.read_csv("Data/A-preprocessed.csv")
df_B = pd.read_csv("Data/B-preprocessed.csv")
df_C = pd.read_csv("Data/C-preprocessed.csv")
df_D = pd.read_csv("Data/D-preprocessed.csv")


for lag in [1, 2, 48, 96, 336]:
    df_A = df_A.drop(columns=[f"cv_lag_{lag}"])
    df_A = df_A.drop(columns=[f"abd_lag_{lag}"])
    df_A = df_A.drop(columns=[f"cct_lag_{lag}"])
    df_B = df_B.drop(columns=[f"cv_lag_{lag}"])
    df_B = df_B.drop(columns=[f"abd_lag_{lag}"])
    df_B = df_B.drop(columns=[f"cct_lag_{lag}"])
    df_C = df_C.drop(columns=[f"cv_lag_{lag}"])
    df_C = df_C.drop(columns=[f"abd_lag_{lag}"])
    df_C = df_C.drop(columns=[f"cct_lag_{lag}"])
    df_D = df_D.drop(columns=[f"cv_lag_{lag}"])
    df_D = df_D.drop(columns=[f"abd_lag_{lag}"])
    df_D = df_D.drop(columns=[f"cct_lag_{lag}"])



df_A = df_A.drop(columns=['cv_roll_mean_3'])
df_A = df_A.drop(columns=["cv_roll_mean_48"])
df_A = df_A.drop(columns=["abd_roll_mean_3"])
df_A = df_A.drop(columns=["cct_roll_mean_3"])

df_B = df_B.drop(columns=['cv_roll_mean_3'])
df_B = df_B.drop(columns=["cv_roll_mean_48"])
df_B = df_B.drop(columns=["abd_roll_mean_3"])
df_B = df_B.drop(columns=["cct_roll_mean_3"])

df_C = df_C.drop(columns=['cv_roll_mean_3'])
df_C = df_C.drop(columns=["cv_roll_mean_48"])
df_C = df_C.drop(columns=["abd_roll_mean_3"])
df_C = df_C.drop(columns=["cct_roll_mean_3"])

df_D = df_D.drop(columns=['cv_roll_mean_3'])
df_D = df_D.drop(columns=["cv_roll_mean_48"])
df_D = df_D.drop(columns=["abd_roll_mean_3"])
df_D = df_D.drop(columns=["cct_roll_mean_3"])


df_A = df_A.drop(columns=['date'])
df_B = df_B.drop(columns=['date'])
df_C = df_C.drop(columns=['date'])
df_D = df_D.drop(columns=['date'])


df_A.to_csv("Data/A-preprocessed-reduced.csv", index=False)
df_B.to_csv("Data/B-preprocessed-reduced.csv", index=False)
df_C.to_csv("Data/C-preprocessed-reduced.csv", index=False)
df_D.to_csv("Data/D-preprocessed-reduced.csv", index=False)