import sagemaker
from sagemaker.sklearn.estimator import SKLearn

session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = session.default_bucket()

# Upload both files to the same S3 prefix
session.upload_data(
    path='combined_data.csv',
    bucket=bucket,
    key_prefix='call-center/train'
)

session.upload_data(
    path='staffing_filled.csv',
    bucket=bucket,
    key_prefix='call-center/train'
)

# Fix: point to the folder prefix, not the specific file
# This tells SageMaker to download everything under call-center/train/
# into /opt/ml/input/data/train/ — so both CSVs will be present
s3_train_prefix = f's3://{bucket}/call-center/train'

for target in ['Call Volume', 'CCT', 'Abandoned Rate']:
    sklearn_estimator = SKLearn(
        entry_point='train.py',
        source_dir='./',
        role=role,
        instance_count=1,
        instance_type='ml.m5.large',
        framework_version='1.2-1',
        py_version='py3',
        hyperparameters={'target': target}
    )
    sklearn_estimator.fit({'train': s3_train_prefix})
    print(f"Done: {target}")