import boto3
import os
from dotenv import load_dotenv
from .s3_transfer_file import upload_file, download_file, write_file
from .url_generator import s3_url_generator


env_file = 's3_env.txt'
load_dotenv(dotenv_path=env_file)
client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('ACCESS_KEY'),
    aws_secret_access_key=os.getenv('SECRET_KEY'),
    endpoint_url=os.getenv('ENDPOINT_URL')
)

