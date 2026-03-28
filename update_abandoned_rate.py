import pandas as pd

# Read CSV (use python engine if your file had parsing issues before)
df = pd.read_csv('combined_data.csv', engine='python')

# Convert 'Abandoned Rate' from percent string to proportion
df['Abandoned Rate'] = (
    df['Abandoned Rate']
    .astype(str)                 # ensure string
    .str.replace('%', '', regex=False)  # remove %
    .str.strip()                 # remove whitespace
)

# Convert to numeric and scale to [0,1]
df['Abandoned Rate'] = pd.to_numeric(df['Abandoned Rate'], errors='coerce') / 100

# Clip values to [0, 1]
df['Abandoned Rate'] = df['Abandoned Rate'].clip(0, 1)

# Save back to CSV
df.to_csv('combined_data.csv', index=False)