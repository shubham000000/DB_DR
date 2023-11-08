import boto3
from botocore.exceptions import ClientError

class CognitoIdentityProviderWrapper:
    """Encapsulates Amazon Cognito actions"""
    def __init__(self, app):                                      
        self.__cognito_idp_client = boto3.client('cognito-idp', region_name=app.config['AWS_DEFAULT_REGION'])
        self.__user_pool_id = app.config['AWS_COGNITO_USER_POOL_ID']
        self.__client_id = app.config['AWS_COGNITO_USER_POOL_CLIENT_ID']
        self.__client_secret = app.config['AWS_COGNITO_USER_POOL_CLIENT_SECRET']

    # Sign Up
    def sign_up_user(self, user_name, password, user_email, name, birthdate):
        """
        Signs up a new user with Amazon Cognito. This action prompts Amazon Cognito
        to send an email to the specified email address. The email contains a code that
        can be used to confirm the user.

        When the user already exists, the user status is checked to determine whether
        the user has been confirmed.

        :param user_name: The user name that identifies the new user.
        :param password: The password for the new user.
        :param user_email: The email address for the new user.
        :return: True when the user is already confirmed with Amazon Cognito.
                 Otherwise, false.
        """
        try:
            kwargs = {
                'ClientId': self.__client_id, 'Username': user_name, 'Password': password,
                'UserAttributes': [{'Name': 'email', 'Value': user_email},
                                   {'Name': 'name', 'Value': name},
                                   {'Name': 'birthdate', 'Value': birthdate},]}
            if self.client_secret is not None:
                kwargs['SecretHash'] = self._secret_hash(user_name)
            response = self.__cognito_idp_client.sign_up(**kwargs)
            confirmed = response['UserConfirmed']
        except ClientError as err:
            if err.response['Error']['Code'] == 'UsernameExistsException':
                response = self.__cognito_idp_client.admin_get_user(
                    UserPoolId=self.__user_pool_id, Username=user_name)
                print("User %s exists and is %s.", user_name, response['UserStatus'])
                confirmed = response['UserStatus'] == 'CONFIRMED'
            else:
                print(
                    "Couldn't sign up %s. Here's why: %s: %s", user_name,
                    err.response['Error']['Code'], err.response['Error']['Message'])
                raise
        return confirmed

    def confirm_user_sign_up(self, user_name, confirmation_code):
        """
        Confirms a previously created user. A user must be confirmed before they
        can sign in to Amazon Cognito.

        :param user_name: The name of the user to confirm.
        :param confirmation_code: The confirmation code sent to the user's registered
                                  email address.
        :return: True when the confirmation succeeds.
        """
        try:
            kwargs = {
                'ClientId': self.__client_id, 'Username': user_name,
                'ConfirmationCode': confirmation_code}
            if self.__client_secret is not None:
                kwargs['SecretHash'] = self._secret_hash(user_name)
            self.__cognito_idp_client.confirm_sign_up(**kwargs)
        except ClientError as err:
            print(
                "Couldn't confirm sign up for %s. Here's why: %s: %s", user_name,
                err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return True
        
    def resend_confirmation(self, user_name):
        """
        Prompts Amazon Cognito to resend an email with a new confirmation code.

        :param user_name: The name of the user who will receive the email.
        :return: Delivery information about where the email is sent.
        """
        try:
            kwargs = {
                'ClientId': self.__client_id, 'Username': user_name}
            if self.__client_secret is not None:
                kwargs['SecretHash'] = self._secret_hash(user_name)
            response = self.__cognito_idp_client.resend_confirmation_code(**kwargs)
            delivery = response['CodeDeliveryDetails']
        except ClientError as err:
            print(
                "Couldn't resend confirmation to %s. Here's why: %s: %s", user_name,
                err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return delivery
        
    # Sign In
    def start_sign_in(self, user_name, password):
        """
        Starts the sign-in process for a user by using administrator credentials.
        This method of signing in is appropriate for code running on a secure server.

        :param user_name: The name of the user to sign in.
        :param password: The user's password.
        :return: The result of the sign-in attempt. When sign-in is successful, this
                 returns an access token that can be used to get AWS credentials. Otherwise,
                 Amazon Cognito returns a challenge to set up an MFA application,
                 or a challenge to enter an MFA code from a registered MFA application.
        """
        try:
            kwargs = {
                'UserPoolId': self.__user_pool_id,
                'ClientId': self.__client_id,
                'AuthFlow': 'ADMIN_USER_PASSWORD_AUTH',
                'AuthParameters': {'USERNAME': user_name, 'PASSWORD': password}}
            if self.__client_secret is not None:
                kwargs['AuthParameters']['SECRET_HASH'] = self._secret_hash(user_name)
            response = self.__cognito_idp_client.admin_initiate_auth(**kwargs)
            # challenge_name = response.get('ChallengeName', None)
            # if challenge_name == 'MFA_SETUP':
            #     if 'SOFTWARE_TOKEN_MFA' in response['ChallengeParameters']['MFAS_CAN_SETUP']:
            #         response.update(self.get_mfa_secret(response['Session']))
            #     else:
            #         raise RuntimeError(
            #             "The user pool requires MFA setup, but the user pool is not "
            #             "configured for TOTP MFA. This example requires TOTP MFA.")
        except ClientError as err:
            print(
                "Couldn't start sign in for %s. Here's why: %s: %s", user_name,
                err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            response.pop('ResponseMetadata', None)
            return response
    # User
    def get_user(self, access_token):
        """
        Gets the user attributes and metadata for a user.

        :return: user attributes and metadata
        """
        try:
            response = self.__cognito_idp_client.get_user(
                AccessToken=access_token
            )
            return response["UserAttributes"]
        except ClientError as err:
            print(
            "Couldn't get this user. Here's why: %s: %s",
            err.response['Error']['Code'], err.response['Error']['Message'])

            raise
    # Admin    
    def admin_delete_user(self, user_name):
        """
        Delete a user as an administrator. Work on any user.

        :return: None.
        """
        try:
            kwargs = {
                'UserPoolId': self.__user_pool_id, 'Username': user_name}
            _ = self.__cognito_idp_client.admin_delete_user(**kwargs)
        except ClientError as err:
            print(
                "Couldn't delete user %s from %s. Here's why: %s: %s", user_name, self.__user_pool_id,
                err.response['Error']['Code'], err.response['Error']['Message'])
            raise

    def admin_list_users(self):
        """
        Returns a list of the users in the current user pool.

        :return: The list of users.
        """
        try:
            response = self.__cognito_idp_client.list_users(UserPoolId=self.__user_pool_id)
            users = response['Users']
        except ClientError as err:
            print(
                "Couldn't list users for %s. Here's why: %s: %s", self.__user_pool_id,
                err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return users      

  