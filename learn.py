import sagemaker
from sagemaker.sklearn.estimator import SKLearn

session = sagemaker.Session()
role = sagemaker.get_execution_role()

s3_train_path = session.upload_data(
    path='combined_data.csv',
    bucket=session.default_bucket(),
    key_prefix='call-center/train'
)

sklearn_estimator = SKLearn(
    entry_point='train.py',
    source_dir='./',
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    framework_version='1.2-1',
    py_version='py3'
)

sklearn_estimator.fit({'train': s3_train_path})