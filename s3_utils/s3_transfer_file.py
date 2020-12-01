import boto3
from botocore.exceptions import ClientError


def upload_file(client, bucket, file_path, file_on_s3, log=False):
    try:
        client.upload_file(file_path, bucket, file_on_s3)
    except ClientError as ex:
        if log:
            print(ex)
        return False
    if log:
        print('Uploaded file {} to s3 ...'.format(file_on_s3))
    return True


def download_file(client, bucket, file_path, file_on_s3, log=False):
    try:
        client.download_file(bucket, file_on_s3, file_path)
    except ClientError as ex:
        if log:
           print(ex)
        return False
    if log:
        print('Downloaded file {} to path {} ...'.format(file_on_s3, file_path)) 

    return True
