import pandas as pd
df = pd.read_csv('combined_data.csv')

# Does each interval appear once (total) or 4 times (once per portfolio)?
print(df.groupby(['Month', 'Day', 'Interval']).size().value_counts())

# What does Call Volume look like per portfolio?
print(df.groupby('Portfolio')['Call Volume'].describe())