import boto3


def s3_url_generator(client, bucket, file_on_s3, expiration=3600):
    params ={
                'Bucket': bucket,
                'Key': file_on_s3
            }
    url = client.generate_presigned_url('get_object', Params=params, 
            ExpiresIn=expiration)
    return url
