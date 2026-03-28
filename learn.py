import sagemaker
from sagemaker.sklearn.estimator import SKLearn

session = sagemaker.Session()
role = sagemaker.get_execution_role()

s3_train_path = session.upload_data(
    path='combined_data.csv',
    bucket=session.default_bucket(),
    key_prefix='call-center/train'
)

# Fix: correct filename is staffing_filled.csv
session.upload_data(
    path='staffing_filled.csv',
    bucket=session.default_bucket(),
    key_prefix='call-center/train'  # same prefix as combined_data.csv
)

for target in ['Call Volume', 'CCT', 'Abandoned Rate']:
    sklearn_estimator = SKLearn(
        entry_point='train.py',
        source_dir='./',
        role=role,
        instance_count=1,
        instance_type='ml.m5.large',
        framework_version='1.2-1',
        py_version='py3',
        hyperparameters={'target': target}  # passed as --target to train.py
    )
    sklearn_estimator.fit({'train': s3_train_path})
    print(f"Done: {target}")
