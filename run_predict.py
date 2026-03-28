import sagemaker
import boto3
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

session   = sagemaker.Session()
role      = sagemaker.get_execution_role()
bucket    = session.default_bucket()
region    = session.boto_region_name

# ── 1. Upload the template CSV to S3 ───────────────────────────────────────
s3_template_path = session.upload_data(
    path='template_forecast_v00.csv',
    bucket=bucket,
    key_prefix='call-center/predict/input'
)
print(f"Template uploaded to: {s3_template_path}")

# ── 2. Point to your trained model artifacts in S3 ─────────────────────────
# Replace with the actual S3 URI from your training job output.
# Find it in Studio under: Experiments → your training job → Artifacts → Output
# It will look like: s3://<bucket>/call-center/models/<job-name>/output/model.tar.gz
TRAINING_JOB_MODEL_URI = "s3://sagemaker-us-west-2-590183996636/sagemaker-scikit-learn-2026-03-28-04-04-40-734/output/model.tar.gz"

# ── 3. Define the Processing Job ───────────────────────────────────────────
# Uses the same sklearn container as training — no custom image needed
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

# S3 destination for the output CSV
s3_output_path = f's3://{bucket}/call-center/predict/output'

processor.run(
    code='predict.py',
    inputs=[
        ProcessingInput(
            source=s3_template_path.rsplit('/', 1)[0],  # folder containing the CSV
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