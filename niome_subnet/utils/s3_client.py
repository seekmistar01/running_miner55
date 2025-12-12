import boto3
import json
import os
from botocore.exceptions import ClientError
from niome_subnet.utils.constants import (
    AWS_ACCESS_KEY_ID, 
    AWS_SECRET_ACCESS_KEY,
    BUCKET_NAME
)


s3_client = boto3.client(
    's3',
    aws_access_key_id = AWS_ACCESS_KEY_ID,
    aws_secret_access_key = AWS_SECRET_ACCESS_KEY
)

def get_task_from_s3(bucket_name: str, task_key: str) -> dict:
    """
    Get a task json file from S3 and parse it into a Python dictionary
    
    Args:
        bucket_name: Name of your S3 bucket (e.g., 'niome-bucket')
        json_file_key: Path to the JSON file in S3 (e.g., 'tasks/tasks.json')
    
    Returns:
        Python dictionary with the JSON content, or None if failed
    """

    try:
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=task_key)
        
        json_content = response['Body'].read().decode('utf-8')
        
        data = json.loads(json_content)
        
        return data
    
    except ClientError as e:
        print(f"Error fetching {task_key} from bucket {bucket_name}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {task_key}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
    
def upload_results_to_s3(bucket_name: str, local_vcf_path : str):
    """
    Upload a local VCF file to S3 bucket
    
    Args:
        bucket_name: Name of your S3 bucket (e.g., 'niome-bucket')
        local_vcf_path: Local path to the VCF file to upload (e.g., '/path/to/results.vcf')
    
    Returns:
        True if upload succeeded, False otherwise
    """
    try:
        if not os.path.exists(local_vcf_path):
            print(f"Local file {local_vcf_path} does not exist.")
            return False

        file_size = os.path.getsize(local_vcf_path)
        file_name = os.path.basename(local_vcf_path)
        s3_key = f"results/{file_name}"
        
        s3_client.upload_file(local_vcf_path, bucket_name, s3_key)
        
        print(f"Successfully uploaded {local_vcf_path} to s3://{bucket_name}/{s3_key}")
        return True
    
    except ClientError as e:
        print(f"Error uploading {local_vcf_path} to bucket {bucket_name}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False