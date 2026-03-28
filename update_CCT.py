import pandas as pd

df = pd.read_csv('combined_data.csv', on_bad_lines='skip')

df['CCT'] = df['CCT'].astype(str).str.replace(',', '', regex=False)

df.to_csv('combined_data.csv', index=False)