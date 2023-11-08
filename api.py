from botocore.exceptions import ClientError
import json
# flask
from flask import Blueprint, request, jsonify

from flask_swagger_ui import get_swaggerui_blueprint

#shubham 

import time
import datetime
from datetime import date,timedelta
from datetime import datetime, timedelta, timezone

import time
from config import Config
import cognito

from forms import RegistrationForm, LoginForm
import flask_awscognito
import random
from flask import Flask, render_template, url_for, session, request, redirect, jsonify
import requests

import json

import numpy as np
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


api_blueprint = Blueprint('api_v1', __name__)

# swagger configs
SWAGGER_URL = '/docs'  # URL for exposing Swagger UI (without trailing '/')
API_URL = '/static/swagger.json'  # Our API url (can of course be a local resource)

# # Call factory function to create our blueprint
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,  # Swagger UI static files will be mapped to '{SWAGGER_URL}/dist/'
    API_URL,
    config={  # Swagger UI config overrides
        'app_name': "Test application"
    },
)

api_blueprint.register_blueprint(swaggerui_blueprint)

model = pickle.load(open("DR_finalized_model_01102023.pkl", 'rb'))

# Health check
@api_blueprint.route('/health', methods=['GET'])
def health():
    response = {'message': 'Retimark Diabetes Management API Server is running'}
    return jsonify(response), 200

# Create a user
@api_blueprint.route('/user', methods=['POST'])
def create_user():
    from app import cip, db
    try:
        # Get the access token from the header
        access_token = request.headers.get('access_token') or "Dummy-String"
        # Verify the access token 
        # _ = cip.get_user(access_token)
        # Get the user data from the request body
        user_data = request.get_json()
        # Store the user data
        insert_query = """
        INSERT INTO Users (
            UserName, FullName, Email, Birthdate, Gender, risk_score_goal
        ) VALUES (
            %s, %s, %s, %s, %s, %s
        )
        """        
        try:
            with db.cursor() as cursor:
                cursor.execute(insert_query, (
                    user_data['username'], user_data['fullname'], f'\'{user_data["email"]}\'',
                    user_data['birthdate'], user_data['gender'],user_data['risk_score_goal']
                ))
                # Commit the change to database
                db.commit()
        except Exception as e:
            response = {'error': f"Failed to add user to database: {e}"}
            db.rollback()
            return jsonify(response), 500
        
        response = {'message': 'User created successfully'}
        return jsonify(response), 200
    except ClientError:
        response = {'error': 'Unauthorised'}
        return jsonify(response), 401
    except Exception as e:
        response = {'error': str(e)}
        return jsonify(response), 500

# Get all users
@api_blueprint.route('/users', methods=['GET'])
def get_users():
    from app import db
    try:
        cursor = db.cursor()
        cursor.execute(
            "SELECT * FROM Users"
        )
        users = cursor.fetchall()
        return jsonify(list(users)), 200

    except Exception as e:
        response = {'error': str(e)}
        return jsonify(response), 500
    finally:
        cursor.close()

# Get user by given email
@api_blueprint.route('/user/<string:email>', methods=['GET'])
def get_user(email):
    from app import cip, db
    try:
        # Get the access token from the header
        access_token = request.headers.get('access_token') or "Dummy-String"
        #Verify the access token 
        # _ = cip.get_user(access_token)
        
        with db.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM Users "
                "WHERE Users.Email = '''{}'''".format(email)
            )
            user = cursor.fetchone()
            if user is None:
                response = {"error": "User not found"}
                return jsonify(response), 404
            return jsonify(user), 200
    except ClientError:
        response = {'error': 'Unauthorised'}
        return jsonify(response), 401 
    except Exception as e:
        response = {'error': str(e)}
        return jsonify(response), 500 

# Update user by given email
@api_blueprint.route('/user/<string:email>', methods=['PUT'])
def update_user(email):
    from app import cip, db
    try:
        # Get the access token from the header
        access_token = request.headers.get('access_token')
        #Verify the access token 
        # _ = cip.get_user(access_token)
        # Get the user data from the request body
        update_data = request.get_json()

        with db.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM Users "
                "WHERE Users.Email = '''{}'''".format(email)
            )
            user = cursor.fetchone()
            if user is None:
                response = {"error": "User not found"}
                return jsonify(response), 404
            
            updates = update_data['updates']
            for update in updates:
                column = update['column']
                value = update['value']
                sql = f"UPDATE Users SET {column} = %s WHERE Users.Email = %s"
                cursor.execute(sql, (value, f'\'{email}\''))
            db.commit()
        response = {'message': 'Sucessfully update user'}
        return jsonify(response), 200

    except ClientError as ce:
        response = {'error': 'Unauthorised'}
        return jsonify(response), 401        
    except Exception as e:
        response = {'error': str(e)}
        db.rollback()
        return jsonify(response), 500 


# Delete user with given email
@api_blueprint.route('/user/<string:email>', methods=['DELETE'])
def delete_user(email):
    from app import cip, db
    try:
        # Get the access token from the header
        access_token = request.headers.get('access_token')
        #Verify the access token 
        # _ = cip.get_user(access_token)

        with db.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM Users "
                "WHERE Users.Email = '''{}'''".format(email)
            )
            user = cursor.fetchone()
            if user is None:
                response = {"error": "User not found"}
                return jsonify(response), 404
            delete_reports(email)
            cursor.execute("DELETE FROM Users "
                           "WHERE Users.Email = '''{}'''".format(email))
            db.commit()
        response = {'message': 'Sucessfully delete user'}
        return jsonify(response), 200

    except ClientError as ce:
        response = {'error': 'Unauthorised'}
        return jsonify(response), 401        
    except Exception as e:
        response = {'error': str(e)}
        db.rollback()
        return jsonify(response), 500 


@api_blueprint.route('/home/<string:email>', methods=['GET'])
def home(email):
    from app import cip, db
    try:
        # Get the access token from the header
        access_token = request.headers.get('access_token') or "Dummy-String"
        #Verify the access token 
        # _ = cip.get_user(access_token)
        
        with db.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM Users "
                "WHERE Users.Email = '''{}'''".format(email)
            )
            user = cursor.fetchone()
            if user is None:
                response = {"error": "User not found"}
                return jsonify(response), 404
            cursor.execute(
                # "SELECT * FROM past_report "
                # "WHERE email = '''{}'''".format(email)
                "SELECT Users.*, past_report.* FROM Users "
                "JOIN past_report ON Users.Email = past_report.email "
                "WHERE Users.Email = '''{}''' "
                "AND past_report.diagnosis_time IN "
                "(SELECT MAX(diagnosis_time) FROM past_report "
                "WHERE Users.Email = past_report.email "
                "GROUP BY DATE(diagnosis_time)) "
                "ORDER BY past_report.diagnosis_time DESC;".format(email)
            )
            reports = cursor.fetchall()
            risk_score_goal = reports[0]['risk_score_goal']
            # past_report_ref = db.collection("Users").document(person["uid"]).collection("past_report")
            # query = past_report_ref.order_by("diagnosis_time", direction=firestore.Query.DESCENDING).limit(5)
            # results = query.stream()
            #report_list = []

           # latest_report = []
           # second_latest_report = []
           # third_latest_report = []
           # fourth_latest_report = []
           # fifth_latest_report = []
           # latest_diagnosis_date = ''
           # second_diagnosis_date = ''
           # third_diagnosis_date = ''
           # fourth_diagnosis_date = ''
           # fifth_diagnosis_date = ''

            latest_report = None
            second_latest_report = None
            third_latest_report = None
            fourth_latest_report = None
            fifth_latest_report = None
            latest_diagnosis_date = ''
            second_diagnosis_date = ''
            third_diagnosis_date = ''
            fourth_diagnosis_date = ''
            fifth_diagnosis_date = ''
            # for doc in results:
            #     report_list.append(doc.to_dict())
            print(type(reports))
            if len(reports) != 0:
                latest_report = reports[0]
            
                #latest_report = report_list
                print("Diagnosis time:", latest_report['diagnosis_time'])

                date_time_str = latest_report['diagnosis_time'] #+ timedelta(hours=8)
                
                #date_time_str = "2023-09-10 09:21:18"
                latest_diagnosis_date = date_time_str.strftime("%Y-%m-%d")
                
                #print(report_list[0])
                #print(len(report_list))
                if len(reports) >= 2:
                    #print(report_list[1])
                    second_latest_report = reports[1]
                    
                    second_date_time_str = second_latest_report['diagnosis_time'] #+ timedelta(hours=8)
                    second_diagnosis_date = second_date_time_str.strftime("%Y-%m-%d")
                    if len(reports) >= 3:
                        #print(report_list[2])
                        third_latest_report = reports[2]
                        third_date_time_str = third_latest_report['diagnosis_time'] #+ timedelta(hours=8)
                        third_diagnosis_date = third_date_time_str.strftime("%Y-%m-%d")
                        if len(reports) >= 4:
                            #print(report_list[2])
                            fourth_latest_report = reports[3]
                            fourth_date_time_str = fourth_latest_report['diagnosis_time'] #+ timedelta(hours=8)
                            fourth_diagnosis_date = fourth_date_time_str.strftime("%Y-%m-%d")
                            if len(reports) >= 5:
                                #print(report_list[2])
                                fifth_latest_report = reports[4]
                                fifth_date_time_str = fifth_latest_report['diagnosis_time'] #+ timedelta(hours=8)
                                fifth_diagnosis_date = fifth_date_time_str.strftime("%Y-%m-%d")
                            else:
                                fifth_diagnosis_date = "NA"
                        else:
                            fourth_diagnosis_date = "NA"
                            fifth_diagnosis_date = "NA"
                    else:
                        third_diagnosis_date = "NA"
                        fourth_diagnosis_date = "NA"
                        fifth_diagnosis_date = "NA"
                else:
                    second_diagnosis_date = "NA"
                    third_diagnosis_date = "NA"
                    fourth_diagnosis_date = "NA"
                    fifth_diagnosis_date = "NA"
            else:
                latest_diagnosis_date = 'NA'
                second_diagnosis_date = "NA"
                third_diagnosis_date = "NA"
                fourth_diagnosis_date = "NA"
                fifth_diagnosis_date = "NA"
                latest_report = {"risk_score": "not diagnosed yet"}
            # return render_template("home.html", risk_score_goal = int(risk_score_goal), name = person["username"], latest_report=latest_report,
            #                        latest_diagnosis_date=latest_diagnosis_date, second_latest_report=second_latest_report,
            #                        second_diagnosis_date=second_diagnosis_date, third_latest_report=third_latest_report,
            #                        third_diagnosis_date=third_diagnosis_date, fourth_latest_report=fourth_latest_report,
            #                        fourth_diagnosis_date=fourth_diagnosis_date, fifth_latest_report=fifth_latest_report,
            #                        fifth_diagnosis_date=fifth_diagnosis_date)
            # Check if the request accepts JSON, and return JSON if true
            global response_data
            response_data = {
                "risk_score_goal": int(risk_score_goal),  # Convert risk_score_goal to an integer
                #"name": reports[0]["username"],  # User's username from the latest report
                "name": reports[0]["UserName"],  # User's username from the latest report
                "latest_report": latest_report,  # Data for the latest report
                "latest_diagnosis_date": latest_diagnosis_date,  # Date of the latest diagnosis
                "second_latest_report": second_latest_report,  # Data for the second latest report
                "second_diagnosis_date": second_diagnosis_date,  # Date of the second latest diagnosis
                "third_latest_report": third_latest_report,  # Data for the third latest report
                "third_diagnosis_date": third_diagnosis_date,  # Date of the third latest diagnosis
                "fourth_latest_report": fourth_latest_report,  # Data for the fourth latest report
                "fourth_diagnosis_date": fourth_diagnosis_date,  # Date of the fourth latest diagnosis
                "fifth_latest_report": fifth_latest_report,  # Data for the fifth latest report
                "fifth_diagnosis_date": fifth_diagnosis_date,  # Date of the fifth latest diagnosis
            }        
            return jsonify(response_data), 200
    except ClientError:
        response = {'error': 'Unauthorised'}
        return jsonify(response), 401 
    except Exception as e:
        response = {'error': str(e)}
        return jsonify(response), 500 


@api_blueprint.route('/db_simulation/<string:email>', methods=['GET'])
def db_simulation(email):
    from app import cip, db
    try:
        # Get the access token from the header
        access_token = request.headers.get('access_token') or "Dummy-String"
        #Verify the access token 
        # _ = cip.get_user(access_token)
        
        with db.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM Users "
                "WHERE Users.Email = '''{}'''".format(email)
            )
            user = cursor.fetchone()
            if user is None:
                response = {"error": "User not found"}
                return jsonify(response), 404
            cursor.execute(
                # "SELECT * FROM past_report "
                # "WHERE email = '''{}'''".format(email)
                "SELECT Users.*, past_report.* FROM Users "
                "JOIN past_report ON Users.Email = past_report.email "
                "WHERE Users.Email = '''{}''' "
                "AND past_report.diagnosis_time IN "
                "(SELECT MAX(diagnosis_time) FROM past_report "
                "WHERE Users.Email = past_report.email "
                "GROUP BY DATE(diagnosis_time)) "
                "ORDER BY past_report.diagnosis_time DESC;".format(email)
            )
            reports = cursor.fetchone()
            if reports is not None:
                print("NP SIZE")
                #latest_report = report_list[0]
                latest_report = reports
                #print("latest_report",latest_report)
                date_time_str = latest_report['diagnosis_time'] + timedelta(hours=8)

                diagnosis_str = date_time_str.strftime("%Y-%m-%d %H:%M:%S")
                diagnosis_date = date_time_str.strftime("%Y-%m-%d")
                # print(report_list[0])
                #print(report_list)
            else:

                latest_report ={
                                'risk_score_goal': 0, 'age': 0, 'HE_sbp': 0, 'HE_dbp': 0, 
                                'HE_ht': 0, 'HE_wt': 0, 'HE_wc': 0, 'HE_BMI': 0, 'HE_glu': 0, 'HE_HbA1c': 0, 'HE_chol': 0, 'HE_HDL_st2': 0, 
                                'HE_TG': 0, 'DE1_dur': 0, 'eGFR': 0, 'sex': 0, 'DE1_31': 0, 'DE1_32': 0, 'HE_HP_2c': 0, 'HE_HCHOL': 0, 'HE_DMfh': 0, 
                                'sm_presnt_3c': 0, 'HE_obe_6c': 0, 'diagnosis_time': "2023-10-4 13:48:43", 'diagnosed_class': 0, 
                                'risk_score': 0, 'risk_score_glucose_50': None, 'predicted_class_glucose_50': None, 'risk_score_glucose_75': None, 
                                'predicted_class_glucose_75': None, 'risk_score_glucose_100': None, 'predicted_class_glucose_100': None}
                # date_time_str = "0000-00-00 00:00:00"

                # diagnosis_str = date_time_str.strftime("%Y-%m-%d %H:%M:%S")
                # diagnosis_date = date_time_str.strftime("%Y-%m-%d")

            # diagnosis_str = "2023-09-10 09:21:18"
            # diagnosis_date = "2023-09-10"

            global latest_diagnosis
            latest_diagnosis = latest_report

            #print("Latest_Diagnosis:", latest_diagnosis)
            global latest_diagnosis_date
            latest_diagnosis_date = diagnosis_str
            global diagnosis_date_str
            diagnosis_date_str = diagnosis_date
            response_data = {
                "latest_report": latest_report,  # Convert risk_score_goal to an integer
                #"name": reports[0]["username"],  # User's username from the latest report
                "diagnosis_date": diagnosis_str,  # User's username from the latest report
                "diagnosis_date_str": diagnosis_date_str,  # Data for the latest report
            }        
            return jsonify(response_data), 200
    except ClientError:
        response = {'error': 'Unauthorised'}
        return jsonify(response), 401 
    except Exception as e:
        response = {'error': str(e)}
        return jsonify(response), 500





@api_blueprint.route('/db_simulation_user/<string:email>/bloodtest', methods=['POST'])
def db_simulation_user(email):
    from app import cip, db
    try:
        # Get the access token from the header
        access_token = request.headers.get('access_token') or "Dummy-String"
      
        bloodtest = float(request.args.get('bloodtest', default=1))

        user_data = request.get_json()
        result = request.form
        # get basic information
        todays_date = date.today()

        sex = float(user_data['sex'])
        age = float(user_data['age'])
        HE_ht = float(user_data['HE_ht'])
        HE_wt = float(user_data['HE_wt'])
        HE_wc = float(user_data['HE_wc'])
        # calculate BMI
        HE_BMI = HE_wt / ((HE_ht / 100) ** 2)
        DE1_ag = float(user_data['DE1_ag'])
        DE1_dur = float((todays_date.year) - DE1_ag)

        if (HE_BMI<18.5):
            HE_obe_6c = 1
        elif (HE_BMI>=18.5) & (HE_BMI<23.0):
            HE_obe_6c = 2
        elif (HE_BMI>=23.0) & (HE_BMI<25.0):
            HE_obe_6c = 3
        elif (HE_BMI>=25.0) & (HE_BMI<30.0):
            HE_obe_6c = 4
        elif (HE_BMI>=30.0) & (HE_BMI<35.0):
            HE_obe_6c = 5
        elif (HE_BMI>=35.0):
            HE_obe_6c = 6

  
        if bloodtest == 1:
            HE_sbp = float(user_data['HE_sbp'])
            HE_dbp = float(user_data['HE_dbp'])
            HE_chol = float(user_data['HE_chol'])
            HE_HDL_st2 = float(user_data['HE_HDL_st2'])
            HE_TG = float(user_data['HE_TG'])
            HE_glu = float(user_data['HE_glu'])
            HE_HbA1c = float(user_data['HE_HbA1c'])
            eGFR = float(user_data['eGFR'])

        else:
            HE_sbp = None
            HE_dbp = None
            HE_chol = None
            HE_HDL_st2 = None
            HE_TG = None
            HE_glu = None
            HE_HbA1c = None
            eGFR = None
            # HE_BUN = None
            # HE_crea = None
        if bloodtest == 1:
            if (HE_sbp >=140 or HE_dbp >= 90):
                HE_HP_2c = 1
            else: 
                HE_HP_2c = 0
        else:
            HE_HP_2c = None


        sm_presnt_3c = float(user_data['sm_presnt_3c'])

        HE_DMfh = float(user_data['HE_DMfh'])

        DE1_31 = float(user_data['DE1_31']) 
        if DE1_31 is not None:
            DE1_31 = float(DE1_31)
        else:
            DE1_31 = None

        DE1_32 = float(user_data['DE1_32'])
        if DE1_32 is not None:
            DE1_32 = float(DE1_32)
        else:
            DE1_32 = None


        # preprocessing for HE_HCHOL
        HE_HCHOL = 0
        if bloodtest == 1:
            if HE_chol >= 240:
                HE_HCHOL = 1
        else:
            HE_HCHOL = 0
        
        try:
            global simulation_report
        
            # risk score prediction for with blood test
            diagnosed_class = None
            rounded_risk_score = None
            risk_score_glucose_50 = None
            risk_score_glucose_75 = None
            risk_score_glucose_100 = None

            # predicted diabetes class
            predicted_class = None
            predicted_class_glucose_50 = None
            predicted_class_glucose_75 = None
            predicted_class_glucose_100 = None

            if bloodtest == 1:
                
                t = pd.DataFrame(np.array(
                    [age, HE_sbp, HE_dbp, HE_ht, HE_wt, HE_wc, HE_BMI, HE_glu, HE_HbA1c, HE_chol, 
                     HE_HDL_st2, HE_TG, DE1_dur, eGFR, sex, DE1_31, DE1_32, HE_HP_2c, HE_HCHOL, 
                     HE_DMfh, sm_presnt_3c, HE_obe_6c]).reshape(-1,22),
                            columns=['age', 'HE_sbp', 'HE_dbp', 'HE_ht', 'HE_wt', 'HE_wc', 'HE_BMI', 'HE_glu', 'HE_HbA1c', 'HE_chol', 'HE_HDL_st2', 'HE_TG',
                        'DE1_dur', 'eGFR', 'sex', 'DE1_31', 'DE1_32', 'HE_HP_2c', 'HE_HCHOL', 'HE_DMfh', 'sm_presnt_3c', 'HE_obe_6c'])
                diagnosed_class = model.predict(t)
                predicted_class = float(diagnosed_class[0])
                risk_score = model.predict_proba(t)[0][1]
                rounded_risk_score = float(round(risk_score * 100))
                # print(predicted_class)
                # print(rounded_risk_score)

            # risk score prediction for without blood test in confidence interval
            else:
      
                
                t_50 = pd.DataFrame(np.array(
                    [age, 118.98240115718419, 75.55882352941177, HE_ht, HE_wt, HE_wc, HE_BMI, 2, 5.534691417550627, 0, 
                     51.822621449955356, 121.89223722275796, DE1_dur, 88.915500, sex, DE1_31, DE1_32, 1, HE_HCHOL, 
                     HE_DMfh, sm_presnt_3c, HE_obe_6c]).reshape(-1,22),
                            columns=['age', 'HE_sbp', 'HE_dbp', 'HE_ht', 'HE_wt', 'HE_wc', 'HE_BMI', 'HE_glu', 'HE_HbA1c', 'HE_chol', 'HE_HDL_st2', 'HE_TG',
                        'DE1_dur', 'eGFR', 'sex', 'DE1_31', 'DE1_32', 'HE_HP_2c', 'HE_HCHOL', 'HE_DMfh', 'sm_presnt_3c', 'HE_obe_6c'])
                diagnosed_class_50 = model.predict(t_50)
                predicted_class_glucose_50 = float(diagnosed_class_50[0])
                risk_score_50 = model.predict_proba(t_50)[0][1]
                risk_score_glucose_50 = float(round(risk_score_50 * 100))
                # print(predicted_class_glucose_50)
                # print(risk_score_glucose_50)

                t_75 = pd.DataFrame(np.array(
                    [age, 124.16230366492147, 77.2324607329843, HE_ht, HE_wt, HE_wc, HE_BMI, 2, 5.739895287958115, 0, 
                     49.74235817995025, 142.69476439790577, DE1_dur, 102.563000, sex, DE1_31, DE1_32, 3, HE_HCHOL, 
                     HE_DMfh, sm_presnt_3c, HE_obe_6c]).reshape(-1,22),
                        columns=['age', 'HE_sbp', 'HE_dbp', 'HE_ht', 'HE_wt', 'HE_wc', 'HE_BMI', 'HE_glu', 'HE_HbA1c', 'HE_chol', 'HE_HDL_st2', 'HE_TG',
                        'DE1_dur', 'eGFR', 'sex', 'DE1_31', 'DE1_32', 'HE_HP_2c', 'HE_HCHOL', 'HE_DMfh', 'sm_presnt_3c', 'HE_obe_6c'])
                diagnosed_class_75 = model.predict(t_75)
                predicted_class_glucose_75 = float(diagnosed_class_75[0])
                risk_score_75 = model.predict_proba(t_75)[0][1]
                risk_score_glucose_75 = float(round(risk_score_75 * 100))
                # print(predicted_class_glucose_75)
                # print(risk_score_glucose_75)

                t_100 = pd.DataFrame(np.array(
                    [age, 126.92238648363252, 77.0063357972545, HE_ht, HE_wt, HE_wc, HE_BMI, 2, 6.763727560718057, 0, 
                     46.931693880777516, 172.6441393875396, DE1_dur, 115.661, sex, DE1_31, DE1_32, 3, HE_HCHOL, 
                     HE_DMfh, sm_presnt_3c, HE_obe_6c]).reshape(-1,22),
                            columns=['age', 'HE_sbp', 'HE_dbp', 'HE_ht', 'HE_wt', 'HE_wc', 'HE_BMI', 'HE_glu', 'HE_HbA1c', 'HE_chol', 'HE_HDL_st2', 'HE_TG',
                        'DE1_dur', 'eGFR', 'sex', 'DE1_31', 'DE1_32', 'HE_HP_2c', 'HE_HCHOL', 'HE_DMfh', 'sm_presnt_3c', 'HE_obe_6c'])
                diagnosed_class_100 = model.predict(t_100)
                predicted_class_glucose_100 = float(diagnosed_class_100[0])
                risk_score_100 = model.predict_proba(t_100)[0][1]
                risk_score_glucose_100 = float(round(risk_score_100 * 100))

                rounded_risk_score = round((risk_score_glucose_50 + risk_score_glucose_75 + risk_score_glucose_100) / 3)
            #email = "borghare.sb@gmail.com"
            # store all data in dic
            
            diagnosis_time = (datetime.now(timezone.utc) + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")

            simulation_report = {"diagnosis_time": (datetime.now(timezone.utc) + timedelta(hours=8)).strftime("%Y-%m-%d"),
                                "diagnosed_class": predicted_class, "risk_score": rounded_risk_score,
                                "risk_score_glucose_50": risk_score_glucose_50,
                                "predicted_class_glucose_50": predicted_class_glucose_50,
                                "risk_score_glucose_75": risk_score_glucose_75,
                                "predicted_class_glucose_75": predicted_class_glucose_75,
                                "risk_score_glucose_100": risk_score_glucose_100,
                                "predicted_class_glucose_100": predicted_class_glucose_100,
                                "email": email, "age": age, "HE_sbp": HE_sbp, "HE_dbp": HE_dbp,
                                "HE_ht": HE_ht, "HE_wt": HE_wt, "HE_wc": HE_wc,  "HE_BMI": HE_BMI,
                                "HE_glu": HE_glu, "HE_HbA1c": HE_HbA1c, "HE_chol": HE_chol, "HE_HDL_st2": HE_HDL_st2,
                                "HE_TG": HE_TG, "DE1_dur": DE1_dur, "eGFR": eGFR, "sex": sex, "DE1_31": DE1_31,
                                "DE1_32": DE1_32, "HE_HP_2c": HE_HP_2c, "HE_HCHOL": HE_HCHOL, "HE_DMfh": HE_DMfh,
                                "sm_presnt_3c": sm_presnt_3c, "HE_obe_6c": HE_obe_6c}
            
                  
            print("DB SIMULATION email above mysql:",email)
            # Insert user data into the database
            #cursor = mysql.connection.cursor()
            cursor = db.cursor()
            print("AGE=",cursor)
            #print("LINE 809", person['email'])   
            # Define the SQL INSERT statement
            # removed the columns (bloodtest, dr_month, dr_high, pa_vig_tm, pa_mod_tm, pa_walkMET, pa_aerobic, DE1_3) 
            insert_query = """
            INSERT INTO past_report (
                    email,age, HE_sbp, HE_dbp, HE_ht, HE_wt, HE_wc, HE_BMI, HE_glu, HE_HbA1c, HE_chol, HE_HDL_st2, HE_TG,
                    DE1_dur, eGFR, sex, DE1_31, DE1_32, HE_HP_2c, HE_HCHOL, HE_DMfh, sm_presnt_3c, HE_obe_6c, diagnosis_time, 
                    diagnosed_class, risk_score, risk_score_glucose_50, predicted_class_glucose_50, risk_score_glucose_75, 
                    predicted_class_glucose_75, risk_score_glucose_100, predicted_class_glucose_100
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """
            # Execute the INSERT statement with the data
            cursor.execute(insert_query, (
                email ,age, HE_sbp, HE_dbp, HE_ht, HE_wt, HE_wc, HE_BMI, HE_glu, HE_HbA1c, 
                HE_chol, HE_HDL_st2, HE_TG, DE1_dur, eGFR, sex, DE1_31, DE1_32, HE_HP_2c, 
                HE_HCHOL, HE_DMfh, sm_presnt_3c, HE_obe_6c, diagnosis_time,  predicted_class, rounded_risk_score, 
                risk_score_glucose_50, predicted_class_glucose_50, risk_score_glucose_75, predicted_class_glucose_75, risk_score_glucose_100, 
                predicted_class_glucose_100
            ))
            
            
            # db.commit()
            # #mysql.connection.commit()
            # cursor.close()

            cursor.execute(
                # "SELECT * FROM past_report "
                # "WHERE email = '''{}'''".format(email)
                "SELECT Users.*, past_report.* FROM Users "
                "JOIN past_report ON Users.Email = past_report.email "
                "WHERE Users.Email = %s "
                "AND past_report.diagnosis_time IN "
                "(SELECT MAX(diagnosis_time) FROM past_report "
                "WHERE Users.Email = past_report.email "
                "GROUP BY DATE(diagnosis_time)) "
                "ORDER BY past_report.diagnosis_time DESC;", (email)
            )

            db.commit()
            #mysql.connection.commit()
            cursor.close()

            reports = cursor.fetchone()
            if reports is not None:
                print("NP SIZE")
                #latest_report = report_list[0]
                latest_report = reports
                #print("latest_report",latest_report)
                date_time_str = latest_report['diagnosis_time'] + timedelta(hours=8)

                diagnosis_str = date_time_str.strftime("%Y-%m-%d %H:%M:%S")
                diagnosis_date = date_time_str.strftime("%Y-%m-%d")
                # print(report_list[0])
                #print(report_list)
            else:

                latest_report ={
                                'risk_score_goal': 0, 'age': 0, 'HE_sbp': 0, 'HE_dbp': 0, 
                                'HE_ht': 0, 'HE_wt': 0, 'HE_wc': 0, 'HE_BMI': 0, 'HE_glu': 0, 'HE_HbA1c': 0, 'HE_chol': 0, 'HE_HDL_st2': 0, 
                                'HE_TG': 0, 'DE1_dur': 0, 'eGFR': 0, 'sex': 0, 'DE1_31': 0, 'DE1_32': 0, 'HE_HP_2c': 0, 'HE_HCHOL': 0, 'HE_DMfh': 0, 
                                'sm_presnt_3c': 0, 'HE_obe_6c': 0, 'diagnosis_time': "2023-10-4 13:48:43", 'diagnosed_class': 0, 
                                'risk_score': 0, 'risk_score_glucose_50': None, 'predicted_class_glucose_50': None, 'risk_score_glucose_75': None, 
                                'predicted_class_glucose_75': None, 'risk_score_glucose_100': None, 'predicted_class_glucose_100': None}
                # date_time_str = "0000-00-00 00:00:00"

                # diagnosis_str = date_time_str.strftime("%Y-%m-%d %H:%M:%S")
                # diagnosis_date = date_time_str.strftime("%Y-%m-%d")

            # diagnosis_str = "2023-09-10 09:21:18"
            # diagnosis_date = "2023-09-10"

            global latest_diagnosis
            latest_diagnosis = latest_report

            #print("Latest_Diagnosis:", latest_diagnosis)
            global latest_diagnosis_date
            latest_diagnosis_date = diagnosis_str
            global diagnosis_date_str
            diagnosis_date_str = diagnosis_date

            db_simulation_user_report = {"simulated_report": simulation_report,
                                "latest_report": latest_diagnosis, "diagnosis_time": diagnosis_date_str,
                                }
            
            return jsonify(db_simulation_user_report), 401
        except Exception as e:
            print("this is second try")
            print("hello ",e)
    except Exception as e:
        response = {'error': str(e)}
        return jsonify(response), 500



# Get Report
@api_blueprint.route('/report/<string:email>', methods=['GET'])
def report(email):
    from app import cip, db
    try:
        # Get the access token from the header
        access_token = request.headers.get('access_token') or "Dummy-String"
        #Verify the access token 
        # _ = cip.get_user(access_token)
        
        with db.cursor() as cursor:
            cursor.execute(
                "SELECT Users.*, past_report.* FROM Users JOIN past_report "
               "ON Users.Email = past_report.email WHERE Users.Email = '''{}''' "
               "ORDER BY past_report.diagnosis_time DESC LIMIT 2;".format(email)
            )
            user = cursor.fetchall()
            if user is None:
                response = {"error": "User not found"}
                return jsonify(response), 404
            return jsonify(user), 200
    except ClientError:
        response = {'error': 'Unauthorised'}
        return jsonify(response), 401 
    except Exception as e:
        response = {'error': str(e)}
        return jsonify(response), 500


# Get all reports of given email
@api_blueprint.route('/report_detail/<string:email>', methods=['POST', 'GET'])
def report_detail(email):
    from app import cip, db
    try:
        # Get the access token from the header
        access_token = request.headers.get('access_token') or "Dummy-String"

        if request.method == 'POST':
            result = request.form.to_dict()
            print("result api to dictionary", result)

            cursor = db.cursor()
            # Execute an SQL query to fetch the user with the provided email
            cursor.execute(
                 "SELECT Users.*, past_report.* FROM Users JOIN past_report "
                "ON Users.Email = past_report.email WHERE Users.Email = '''{}''' "
                "ORDER BY past_report.diagnosis_time DESC;".format(email) 
            )

            user = cursor.fetchall()

            if not user:
                response = {"error": "User not found"}
                return jsonify(response), 404

            return jsonify(user), 200

    except Exception as e:
        response = {'error': str(e)}
        return jsonify(response), 500


# Get all reports of given email
@api_blueprint.route('/reports/<string:email>', methods=['GET'])
def get_reports(email):
    from app import cip, db
    try:
        # Get the access token from the header
        access_token = request.headers.get('access_token') or "Dummy-String"
        #Verify the access token 
        # _ = cip.get_user(access_token)
        
        with db.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM Users "
                "WHERE Users.Email = '''{}'''".format(email)
            )
            user = cursor.fetchone()
            if user is None:
                response = {"error": "User not found"}
                return jsonify(response), 404
            cursor.execute(
                "SELECT * FROM past_report "
                "WHERE email = '''{}'''".format(email)
            )
            reports = cursor.fetchall()            
            return jsonify(list(reports)), 200
    except ClientError:
        response = {'error': 'Unauthorised'}
        return jsonify(response), 401 
    except Exception as e:
        response = {'error': str(e)}
        return jsonify(response), 500 
    
def delete_reports(email):
    from app import cip, db
    try:
        # Get the access token from the header
        # access_token = request.headers.get('access_token')
        #Verify the access token 
        # _ = cip.get_user(access_token)

        with db.cursor() as cursor:  
            cursor.execute("DELETE FROM past_report "
                           "WHERE email = '''{}'''".format(email))
            db.commit()

        return True       
    except Exception as e:
        db.rollback()
        return False
    
# Get most recent report of given email
@api_blueprint.route('/report/<string:email>', methods=['GET'])
def get_report(email):
    from app import cip, db
    try:
        # Get the access token from the header
        access_token = request.headers.get('access_token') or "Dummy-String"
        #Verify the access token 
        # _ = cip.get_user(access_token)
        
        with db.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM Users "
                "WHERE Users.Email = '''{}'''".format(email)
            )
            user = cursor.fetchone()
            if user is None:
                response = {"error": "User not found"}
                return jsonify(response), 404
            sql = """
            SELECT *
            FROM past_report
            WHERE email = %s
            ORDER BY diagnosis_time DESC
            LIMIT 1
            """
            cursor.execute(sql, f'\'{email}\'')
            report = cursor.fetchone()            
            return jsonify(report), 200
    except ClientError:
        response = {'error': 'Unauthorised'}
        return jsonify(response), 401 
    except Exception as e:
        response = {'error': str(e)}
        return jsonify(response), 500 
# Other routes for getting, updating, and deleting users and reports can be similarly defined.
