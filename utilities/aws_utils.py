import boto3

def get_client():
    cred = boto3.Session().get_credentials()

    ACCESS_KEY = cred.access_key
    SECRET_KEY = cred.secret_key
    SESSION_TOKEN = cred.token

    s3client = boto3.client('s3', 
                        aws_access_key_id = ACCESS_KEY, 
                        aws_secret_access_key = SECRET_KEY, 
                        aws_session_token = SESSION_TOKEN
                       )
    return s3client