import joblib
import pandas as pd
from datetime import datetime
import boto3

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

def prepare_prediction_data(raw_data):
    """Convert incoming data to proper DataFrame format"""
    df = pd.DataFrame(raw_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def upload_to_partitioned_s3(local_path, s3_base_path, bucket='handsonllms-raghu'):
    """Upload files to S3 with date partitioning"""
    s3 = boto3.client('s3')
    current_date = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%H%M%S")
    
    # Extract filename
    filename = local_path.split("/")[-1]
    
    # Create S3 path
    s3_path = f"{s3_base_path}/date={current_date}/{timestamp}_{filename}"
    print(f"Uploading to s3://{bucket}/{s3_path}")
    
    try:
        s3.upload_file(local_path, bucket, s3_path)
        print(f"Uploaded to s3://{bucket}/{s3_path}")
    except Exception as e:
        print(f"Error uploading to S3: {str(e)}")

