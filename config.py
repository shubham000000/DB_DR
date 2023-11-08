import os
from dotenv import load_dotenv

class Config:
    load_dotenv()
    SECRET_KEY = os.environ.get('SECRET_KEY')
    AWS_DEFAULT_REGION=os.environ.get('DEFAULT_REGION')
    AWS_COGNITO_DOMAIN = os.environ.get("COGNITO_DOMAIN")
    AWS_COGNITO_USER_POOL_ID=os.environ.get("COGNITO_USER_POOL_ID")
    AWS_COGNITO_USER_POOL_CLIENT_ID = os.environ.get("COGNITO_CLIENT_ID")
    AWS_COGNITO_USER_POOL_CLIENT_SECRET = os.environ.get("COGNITO_CLIENT_SECRET")
    AWS_COGNITO_REDIRECT_URL = os.environ.get("LOCAL_COGNITO_APP_URI")
    DB_USER_NAME=os.environ.get("DB_USER_NAME")
    DB_PWD=os.environ.get("DB_PWD")
    DB_HOST=os.environ.get("DB_HOST")
    DB_NAME=os.environ.get("DB_NAME")
    DB_CURSORCLASS=os.environ.get("DB_CURSORCLASS")