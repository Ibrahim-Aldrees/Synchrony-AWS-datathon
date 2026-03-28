import pandas as pd
import sagemaker

# 1. Combine Files
files = {
    'A': 'A_interval_data_preprocessed.csv',
    'B': 'B_interval_data_preprocessed.csv',
    'C': 'C_interval_data_preprocessed.csv',
    'D': 'D_interval_data_preprocessed.csv'
}

all_dfs = []
for portfolio_name, file_path in files.items():
    temp_df = pd.read_csv(file_path)
    temp_df['Portfolio'] = portfolio_name
    all_dfs.append(temp_df)

combined_df = pd.concat(all_dfs).dropna(subset=['Interval', 'Call Volume'])
combined_df.to_csv('combined_data.csv', index=False)

# 2. Upload to S3
session = sagemaker.Session()
s3_train_path = session.upload_data('combined_data.csv', bucket=session.default_bucket(), key_prefix='call-center/train')