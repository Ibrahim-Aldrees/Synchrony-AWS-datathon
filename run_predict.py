import sagemaker
import boto3
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

session = sagemaker.Session()
role    = sagemaker.get_execution_role()
bucket  = session.default_bucket()
region  = session.boto_region_name

# ── 1. Upload input files to S3 ────────────────────────────────────────────
# Both files go to the same S3 prefix so they land together in
# /opt/ml/processing/input inside the container — only one ProcessingInput needed
session.upload_data(
    path='template_forecast_v00.csv',
    bucket=bucket,
    key_prefix='call-center/predict/input'
)

# Fix: correct filename is staffing_filled.csv
session.upload_data(
    path='staffing_filled.csv',
    bucket=bucket,
    key_prefix='call-center/predict/input'
)

s3_input_prefix = f's3://{bucket}/call-center/predict/input'
print(f"Input files uploaded to: {s3_input_prefix}")

# ── 2. Point to combined model artifact in S3 ──────────────────────────────
# This should be the models_combined.tar.gz you built containing all 12 .joblib files
TRAINING_JOB_MODEL_URI = "s3://sagemaker-us-west-2-590183996636/call-center/models/combined/model.tar.gz"

# ── 3. Define the Processing Job ───────────────────────────────────────────
processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve(
        framework='sklearn',
        region=region,
        version='1.2-1',
        py_version='py3',
        instance_type='ml.m5.large',
    ),
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    command=['python3'],
)

s3_output_path = f's3://{bucket}/call-center/predict/output'

processor.run(
    code='predict.py',
    inputs=[
        # Fix: one ProcessingInput covers both template and staffing since
        # they share the same S3 prefix — two inputs to the same destination
        # would conflict and one would silently overwrite the other
        ProcessingInput(
            source=s3_input_prefix,
            destination='/opt/ml/processing/input'
        ),
        ProcessingInput(
            source=TRAINING_JOB_MODEL_URI,
            destination='/opt/ml/processing/model'
        ),
    ],
    outputs=[
        ProcessingOutput(
            source='/opt/ml/processing/output',
            destination=s3_output_path
        )
    ],
    wait=True,
    logs=True,
)

# ── 4. Download result locally ─────────────────────────────────────────────
s3 = boto3.client('s3')
s3.download_file(
    bucket,
    'call-center/predict/output/forecast_output.csv',
    'forecast_output.csv'
)
print("Done! forecast_output.csv downloaded locally.")
