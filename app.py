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
#import firebase_admin
#import pyrebase
import json

from apscheduler.schedulers.background import BackgroundScheduler

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
#from flask_mysqldb import MySQL
import pymysql
import threading

from api import api_blueprint

app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba240'
#app.register_blueprint(api_blueprint, url_prefix='/api/v1')

app.config.from_object(Config)

#mysql = MySQL(app)
db = pymysql.connect(
    host=app.config['DB_HOST'],
    user=app.config['DB_USER_NAME'],
    password=app.config['DB_PWD'],
    db=app.config['DB_NAME'],
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)


#Initialze person as dictionary
person = {"is_logged_in": False, "username": "", "fullname": "", "email": "", "uid": "", "dob": "", "risk_score_goal": ""}

#initalize report_id for diagnosis
diagnosis_report = {}
latest_diagnosis_date = ''
report_id = ""
diagnosis_date_str = ''

#initialze for simulation
latest_diagnosis = {}
simulation_report = {}

#load machine learning model
#model = pickle.load(open("finalized_model.pkl", 'rb'))
model = pickle.load(open("DR_finalized_model_01102023.pkl", 'rb'))

#initialize flask-awscognito
aws_auth = flask_awscognito.AWSCognitoAuthentication(app)

#initialize Cognito Wrapper Class
cip = cognito.CognitoIdentityProviderWrapper(app)


@app.route('/')
@app.route('/login', methods=['POST', 'GET'])
def login_page():
    return redirect(aws_auth.get_sign_in_url())

# Function to refresh the database
def refresh_database():
    try:
        db = pymysql.connect(
            host=app.config['DB_HOST'],
            user=app.config['DB_USER_NAME'],
            password=app.config['DB_PWD'],
            db=app.config['DB_NAME'],
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        cursor = db.cursor()

        # Perform database operations here
        # For example, you might want to update user data or fetch information

        # Sample query to fetch all users
        cursor.execute("SELECT * FROM Users")
        users = cursor.fetchall()
        #for user in users:
        #    print(user)  # Process the user data as needed

        cursor.close()
        db.close()

        #print("Database Refreshed")
    except pymysql.Error as e:
        print(f"Error refreshing database: {e}")

# Scheduler setup
scheduler = BackgroundScheduler()
scheduler.add_job(refresh_database, 'interval', minutes=1)
scheduler.start()


@app.route('/login_user', methods=['POST', 'GET'])
def login_user():
    try:
        session["access_token"] = aws_auth.get_access_token(request.args)
        attrs = cip.get_user(session["access_token"])
        print("This is access token:", attrs)
        data = {item["Name"]: item["Value"] for item in attrs}

        global person
        person["is_logged_in"] = True
        person["email"] = f'\'{data["email"]}\''
        print("Cognito Email:", person['email'])
        person["gender"] = data['gender']
        person["username"] = data["preferred_username"]
        person["fullname"] = data["name"]
        person['dob'] = data["birthdate"] 

        db = pymysql.connect(
            host=app.config['DB_HOST'],
            user=app.config['DB_USER_NAME'],
            password=app.config['DB_PWD'],
            db=app.config['DB_NAME'],
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        cursor = db.cursor()

        # Execute an SQL query to fetch the user with the provided email
        cursor.execute("SELECT * FROM Users WHERE Users.Email = %s", (person["email"],))
        user = cursor.fetchone()
        print(user)

        if user is None:
            person["risk_score_goal"] = 40
            insert_query = """
            INSERT INTO Users (UserName, FullName, Email, Birthdate, Gender, risk_score_goal)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (
                person['username'], person['fullname'], person['email'],
                person['dob'], person['gender'], person['risk_score_goal']
            ))
            # Commit the change to the database
            db.commit()
        else:
            person["risk_score_goal"] = user["risk_score_goal"]

        cursor.close()
        db.close()

    except pymysql.Error as e:
        if e.args[0] == 2006 or e.args[0] == 2013:
            print("MySQL server has gone away. Reconnecting...")
            db = pymysql.connect(
                host=app.config['DB_HOST'],
                user=app.config['DB_USER_NAME'],
                password=app.config['DB_PWD'],
                db=app.config['DB_NAME'],
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            cursor = db.cursor()
            # Retry the failed operation here or handle reconnection as needed
        else:
            print("Error: {}".format(e))
    except Exception as e:
        print(f"Failed to add user to the database: {e}")

    return redirect(url_for('home_page'))


@app.route('/logout')
def logout():
    session.pop('access_token', None)
    session.clear()  # Clear the 'access_token' from the session
    # Add other session variables if needed and clear them as well
    return redirect(url_for('login_page'))  # Redirect to the login page after logout


@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


#@app.route('/')
@app.route('/home',methods=['POST', 'GET'])
def home_page():

   # global person
   # person["is_logged_in"] = True
   # person['email'] = "'borghare.sb@gmail.com'"
   # person["risk_score_goal"] = 40
    if person["is_logged_in"] == True:

        #based on new aws rds database
        cursor = db.cursor()
        # Execute an SQL query to fetch the user with the provided username and password
        cursor.execute(
            "SELECT Users.*, past_report.* FROM Users "
            "JOIN past_report ON Users.Email = past_report.email "
            "WHERE Users.Email = %s "
            "AND past_report.diagnosis_time IN "
            "(SELECT MAX(diagnosis_time) FROM past_report "
            "WHERE Users.Email = past_report.email "
            "GROUP BY DATE(diagnosis_time)) "
            "ORDER BY past_report.diagnosis_time DESC;", (person['email'], )
        )


        #past_report_ref = db.collection("Users").document(person["uid"]).collection("past_report")
        #data = cursor.fetchone()
        data = cursor.fetchall()
        # print(type(data[0]['diagnosis_time']))
        # print(data[0]['diagnosis_time'].date())
        #print(data[0])
        #risk_score_goal = data['risk_score_goal']
        past_report_ref = data
        #past_report_ref = cursor.fetchone()

        # query = past_report_ref.order_by("diagnosis_time", direction=firestore.Query.DESCENDING).limit(1)
        # results = query.stream()
        results = past_report_ref
        cursor.close()

        #report_list = []
        report_list = results

        # data = db.collection("Users").document(person["uid"]).get().to_dict()
        risk_score_goal = person['risk_score_goal']
        # past_report_ref = db.collection("Users").document(person["uid"]).collection("past_report")
        # query = past_report_ref.order_by("diagnosis_time", direction=firestore.Query.DESCENDING).limit(5)
        # results = query.stream()
        #report_list = []
        latest_report = []
        second_latest_report = []
        third_latest_report = []
        fourth_latest_report = []
        fifth_latest_report = []
        latest_diagnosis_date = ''
        second_diagnosis_date = ''
        third_diagnosis_date = ''
        fourth_diagnosis_date = ''
        fifth_diagnosis_date = ''
        # for doc in results:
        #     report_list.append(doc.to_dict())
        if len(report_list) != 0:
            latest_report = report_list[0]
            #latest_report = report_list
            print("Diagnosis time:", latest_report['diagnosis_time'])

            date_time_str = latest_report['diagnosis_time'] #+ timedelta(hours=8)
            #date_time_str = "2023-09-10 09:21:18"
            latest_diagnosis_date = date_time_str.strftime("%Y-%m-%d")
            #print(report_list[0])
            #print(len(report_list))
            if len(report_list) >= 2:
                #print(report_list[1])
                second_latest_report = report_list[1]
                second_date_time_str = second_latest_report['diagnosis_time'] #+ timedelta(hours=8)
                second_diagnosis_date = second_date_time_str.strftime("%Y-%m-%d")
                if len(report_list) >= 3:
                    #print(report_list[2])
                    third_latest_report = report_list[2]
                    third_date_time_str = third_latest_report['diagnosis_time'] #+ timedelta(hours=8)
                    third_diagnosis_date = third_date_time_str.strftime("%Y-%m-%d")
                    if len(report_list) >= 4:
                        #print(report_list[2])
                        fourth_latest_report = report_list[3]
                        fourth_date_time_str = fourth_latest_report['diagnosis_time'] #+ timedelta(hours=8)
                        fourth_diagnosis_date = fourth_date_time_str.strftime("%Y-%m-%d")
                        if len(report_list) >= 5:
                            #print(report_list[2])
                            fifth_latest_report = report_list[4]
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
        # Check if the request accepts JSON, and return JSON if true
        response_data = {
            "risk_score_goal": int(risk_score_goal),  # Convert risk_score_goal to an integer
            "name": person["username"],  # User's username from the latest report
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
        if request.headers.get('accept') == 'application/json':
            return jsonify(response_data)
        else:
            return render_template("home.html", risk_score_goal = int(risk_score_goal), name = person["username"], latest_report=latest_report,
                        latest_diagnosis_date=latest_diagnosis_date, second_latest_report=second_latest_report,
                        second_diagnosis_date=second_diagnosis_date, third_latest_report=third_latest_report,
                        third_diagnosis_date=third_diagnosis_date, fourth_latest_report=fourth_latest_report,
                        fourth_diagnosis_date=fourth_diagnosis_date, fifth_latest_report=fifth_latest_report,
                        fifth_diagnosis_date=fifth_diagnosis_date)
    else:
        return redirect(url_for('login_page'))


@app.route('/db_simulation', methods=['POST', 'GET'])
def db_simulation_page():

    if person["is_logged_in"] == True:

        latest_report = {"age": 0, "HE_ht": 0, "HE_wt": 0, "HE_wc": 0,
                         "HE_sbp": 0, "HE_dbp": 0, "HE_chol": 0, "HE_HDL_st2": 0, "HE_TG": 0,
                                "HE_glu": 0, "HE_HbA1c": 0
                                }

        diagnosis_str = ''
        diagnosis_date = ''

        cursor = db.cursor()
       # Execute an SQL query to fetch the user with the provided username and password
        print("Person Email:",person['email'])
        cursor.execute(
            "SELECT Users.*, past_report.* FROM Users "
            "JOIN past_report ON Users.Email = past_report.email "
            "WHERE Users.Email = %s "
            "AND past_report.diagnosis_time IN "
            "(SELECT MAX(diagnosis_time) FROM past_report "
            "WHERE Users.Email = past_report.email "
            "GROUP BY DATE(diagnosis_time)) "
            "ORDER BY past_report.diagnosis_time DESC;", (person['email'])
        )


        #past_report_ref = db.collection("Users").document(person["uid"]).collection("past_report")
        past_report_ref = cursor.fetchone()

        #print("Past Report Reference: ", past_report_ref)

        # query = past_report_ref.order_by("diagnosis_time", direction=firestore.Query.DESCENDING).limit(1)
        # results = query.stream()
        results = past_report_ref


        cursor.close()

        print("Results: pp", results)

        #report_list = []
        report_list = results
        print("Reposrt_List:", report_list)
        #print('Report List = ', len(report_list))
        # for doc in results:
        #     #report_list.append(doc.to_dict())
        #     report_list.append(doc)
        # print(report_list)
        # print("------------------------")
        # print(len(report_list))
        # if report_list or len(report_list) != 0:
        # latest_report =
        if report_list is not None:
            print("NP SIZE")
            #latest_report = report_list[0]
            latest_report = report_list
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
        #print("This is latest report:", latest_report)
        return render_template('db_simulation.html', latest_report = latest_report, diagnosis_date = diagnosis_str, diagnosis_date_str = diagnosis_date_str)
    else:
        return redirect(url_for('login_page'))


@app.route('/db_simulation_user', methods=['POST', 'GET'])
def db_simulation_user():
    error = None
    if request.method == 'POST':
        result = request.form
        # get basic information
        todays_date = date.today()

        sex = float(result.get('sex'))
        age = float(result.get('age'))
        HE_ht = float(result.get('HE_ht'))
        HE_wt = float(result.get('HE_wt'))
        HE_wc = float(result.get('HE_wc'))
        # calculate BMI
        HE_BMI = HE_wt / ((HE_ht / 100) ** 2)
        DE1_ag = float(result.get('DE1_ag'))

        #DE1_ag = result.get('DE1_ag')
        #    if DE1_ag:  # Checking for empty string
        #        DE1_ag = float(DE1_ag)
        #        if DE1_ag <= todays_date.year:
        #            DE1_dur = todays_date.year - DE1_ag
        #$        else:
         #           DE1_dur = None  # Handle as appropriate for your use case
         #   else:
         #       DE1_dur = None  # Handle if the input is empty 

        DE1_dur = float((todays_date.year) - DE1_ag)

       #eGFR = float(result.get('eGFR'))
        print("Duration:",DE1_dur)
        # pre-processing for obesity
        # if HE_BMI <= 18.5:
        #     HE_obe_6c = 1
        # elif HE_BMI <= 25:
        #     HE_obe = 2
        # else:
        #     HE_obe = 3
        # HE_BMI required
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

        #print("HE OBE 6:", HE_obe_6c)

        # Get blood test reusults
        #removed the question (Have you done blood testing? in UI)
        bloodtest = float(result.get('bloodtest'))
        #bloodtest = float(1)
        if bloodtest == 1:
            HE_sbp = float(result.get('HE_sbp'))
            HE_dbp = float(result.get('HE_dbp'))
            HE_chol = float(result.get('HE_chol'))
            HE_HDL_st2 = float(result.get('HE_HDL_st2'))
            HE_TG = float(result.get('HE_TG'))
            HE_glu = float(result.get('HE_glu'))
            HE_HbA1c = float(result.get('HE_HbA1c'))
            eGFR = float(result.get('eGFR'))
            # HE_BUN = float(result.get('HE_BUN'))
            #HE_crea = float(result.get('HE_crea'))
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
        # get lifestyles
        # N_PROT = float(result.get('N_PROT'))
        # N_FAT = float(result.get('N_FAT'))
        # N_CHO = float(result.get('N_CHO'))

        # dr_month = float(result.get('dr_month'))
        # dr_high = float(result.get('dr_high'))

        sm_presnt_3c = float(result.get('sm_presnt_3c'))
        # pa_vig_tm = float(result.get('pa_vig_tm'))
        # pa_mod_tm = float(result.get('pa_mod_tm'))
        # pa_walkMET = float(result.get('pa_walkMET'))
        # pa_aerobic = float(result.get('pa_aerobic'))

        # preprocess for physical activity
        # pa_vigMET = round(8 * pa_vig_tm, 2)
        # pa_modMET = round(4 * pa_mod_tm, 2)
        # pa_totMET = round(pa_walkMET * 3.3 + pa_modMET + pa_vigMET, 2)

        # get history disease
        # DI3_dg = float(result.get('DI3_dg'))
        # DI4_dg = float(result.get('DI4_dg'))
        HE_DMfh = float(result.get('HE_DMfh'))
        #DE1_3 = float(result.get('DE1_3'))
        # DI1_2 = float(result.get('DI1_2'))
        # DI2_2 = float(result.get('DI2_2'))
        DE1_31 = result.get('DE1_31')
        if DE1_31 is not None:
            DE1_31 = float(DE1_31)
        else:
            DE1_31 = None

        DE1_32 = result.get('DE1_32')
        if DE1_32 is not None:
            DE1_32 = float(DE1_32)
        else:
            DE1_32 = None

        # preproccessing for HE_HP
        #HE_HP_2c = 1
        # if bloodtest == 1:
        #     if 0 < HE_sbp < 120 and 0 < HE_dbp < 80:
        #         HE_HP_2c = 1
        #     elif 120 <= HE_sbp < 140 or 80 <= HE_dbp < 90:
        #         HE_HP_2c = 2
        #     elif 140 <= HE_sbp or 90 <= HE_dbp :
        #         HE_HP_2c = 3
        #     else:
        #         HE_HP_2c = None
        # else:
        #     HE_HP_2c = None

        # preprocessing for HE_HCHOL
        HE_HCHOL = 0
        if bloodtest == 1:
            if HE_chol >= 240:
                HE_HCHOL = 1
        else:
            HE_HCHOL = 0

        # # preprocessign for HE_HTG
        # HE_HTG = 0
        # if bloodtest == 1:
        #     if HE_TG >= 200:
        #         HE_HTG = 1
        # else:
        #     HE_HTG = 0

        try:
            global simulation_report

            # Run machine learning model to generate risk score
            # 23 independent variables to input in the model
            # ['pa_totMET', 'N_PROT', 'N_CHO', 'N_FAT', 'HE_wc', 'HE_BMI',
            # 'HE_sbp', 'HE_dbp', 'HE_HbA1c', 'HE_BUN', 'HE_crea', 'HE_HDL_st2',
            # 'HE_TG', 'age', 'DI3_dg', 'DI4_dg', 'HE_DMfh', 'sm_presnt',
            # 'HE_obe', 'HE_HP', 'HE_HCHOL', 'HE_HTG', 'sex']

            # risk score prediction for with blood test
            rounded_risk_score = None
            risk_score_glucose_50 = None
            risk_score_glucose_75 = None
            risk_score_glucose_100 = None

            # predicted diabetes class
            diagnosed_class = None
            predicted_class = None
            predicted_class_glucose_50 = None
            predicted_class_glucose_75 = None
            predicted_class_glucose_100 = None

            if bloodtest == 1:

                # t = pd.DataFrame(np.array(
                #     [HE_wc, HE_BMI, HE_sbp, HE_dbp, HE_HbA1c, HE_BUN, HE_crea,
                #      HE_HDL_st2, HE_TG, age, HE_DMfh, HE_obe, HE_HP, HE_HCHOL,
                #      HE_HTG, sm_presnt, sex]).reshape(-1, 17),
                #                  columns=['HE_wc', 'HE_BMI', 'HE_sbp',
                #                           'HE_dbp', 'HE_HbA1c','HE_BUN', 'HE_crea', 'HE_HDL_st2',
                #                           'HE_TG', 'age', 'HE_DMfh', 'HE_obe',
                #                           'HE_HP', 'HE_HCHOL', 'HE_HTG', 'sm_presnt', 'sex'])
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
                # generate risk score if HE_glu being in the 25 th percentile to 50 th percentile
                # removed the feature (pa_totMET)
                # t_50 = pd.DataFrame(np.array(
                #     [HE_wc, HE_BMI, 118.98240115718419, 75.55882352941177,
                #      5.534691417550627, 15.428881388621022, 0.8001157184185149, 51.822621449955356, 121.89223722275796,
                #      age, HE_DMfh, HE_obe, 1, 0, 0, sm_presnt, sex]).reshape(-1, 17), columns=['HE_wc', 'HE_BMI', 'HE_sbp',
                #                                                         'HE_dbp', 'HE_HbA1c','HE_BUN', 'HE_crea', 'HE_HDL_st2',
                #                                                         'HE_TG', 'age', 'HE_DMfh', 'HE_obe',
                #                                                         'HE_HP', 'HE_HCHOL', 'HE_HTG', 'sm_presnt', 'sex'])

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

                # generate risk score if HE_glu being in the 50 th percentile to 75 th percentile
                # t_75 = pd.DataFrame(np.array(
                #     [pa_totMET, HE_wc, HE_BMI, N_PROT, N_CHO, N_FAT, 124.16230366492147, 77.2324607329843,
                #      5.739895287958115, 16.05759162303665, 0.8326178010471204, 49.74235817995025, 142.69476439790577,
                #      age, DI3_dg, DI4_dg, HE_DMfh, HE_obe, 3, 0,
                #      0, sm_presnt, sex]).reshape(-1, 23), columns=['pa_totMET', 'HE_wc', 'HE_BMI', 'N_PROT', 'N_CHO',
                #                                                    'N_FAT', 'HE_sbp',
                #                                                    'HE_dbp', 'HE_HbA1c', 'HE_BUN', 'HE_crea',
                #                                                    'HE_HDL_st2',
                #                                                    'HE_TG', 'age', 'DI3_dg', 'DI4_dg', 'HE_DMfh',
                #                                                    'HE_obe',
                #                                                    'HE_HP', 'HE_HCHOL', 'HE_HTG', 'sm_presnt', 'sex'])
                # t_75 = pd.DataFrame(np.array(
                #     [HE_wc, HE_BMI, 124.16230366492147, 77.2324607329843,
                #      5.739895287958115, 16.05759162303665, 0.8326178010471204, 49.74235817995025, 142.69476439790577,
                #      age, HE_DMfh, HE_obe, 3, 0,
                #      0, sm_presnt, sex]).reshape(-1, 17), columns=['HE_wc', 'HE_BMI', 'HE_sbp',
                #                                                    'HE_dbp', 'HE_HbA1c','HE_BUN', 'HE_crea',
                #                                                    'HE_HDL_st2',
                #                                                    'HE_TG', 'age', 'HE_DMfh',
                #                                                    'HE_obe',
                #                                                    'HE_HP', 'HE_HCHOL', 'HE_HTG', 'sm_presnt', 'sex'])
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

                # generate risk score if HE_glu being in the 75 th percentile to 100th percentile
                # t_100 = pd.DataFrame(np.array(
                #     [HE_wc, HE_BMI, 126.92238648363252, 77.0063357972545,
                #      6.763727560718057, 16.50897571277719, 0.8618532206969378, 46.931693880777516, 172.6441393875396,
                #      age, HE_DMfh, HE_obe, 3, 0,
                #      0, sm_presnt, sex]).reshape(-1, 17), columns=['HE_wc', 'HE_BMI', 'HE_sbp',
                #                                                    'HE_dbp', 'HE_HbA1c','HE_BUN', 'HE_crea',
                #                                                    'HE_HDL_st2',
                #                                                    'HE_TG', 'age', 'HE_DMfh',
                #                                                    'HE_obe',
                #                                                    'HE_HP', 'HE_HCHOL', 'HE_HTG', 'sm_presnt', 'sex'])
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

                # print("LINE ")
                # print(predicted_class_glucose_100)
                # print(risk_score_glucose_100)

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
                                "email": person['email'], "age": age, "HE_sbp": HE_sbp, "HE_dbp": HE_dbp,
                                "HE_ht": HE_ht, "HE_wt": HE_wt, "HE_wc": HE_wc,  "HE_BMI": HE_BMI,
                                "HE_glu": HE_glu, "HE_HbA1c": HE_HbA1c, "HE_chol": HE_chol, "HE_HDL_st2": HE_HDL_st2,
                                "HE_TG": HE_TG, "DE1_dur": DE1_dur, "eGFR": eGFR, "sex": sex, "DE1_31": DE1_31,
                                "DE1_32": DE1_32, "HE_HP_2c": HE_HP_2c, "HE_HCHOL": HE_HCHOL, "HE_DMfh": HE_DMfh,
                                "sm_presnt_3c": sm_presnt_3c, "HE_obe_6c": HE_obe_6c}


            print("DB SIMULATION email above mysql:",person['email'])
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
                person['email'],age, HE_sbp, HE_dbp, HE_ht, HE_wt, HE_wc, HE_BMI, HE_glu, HE_HbA1c, HE_chol, HE_HDL_st2, HE_TG,
                DE1_dur, eGFR, sex, DE1_31, DE1_32, HE_HP_2c, HE_HCHOL, HE_DMfh, sm_presnt_3c, HE_obe_6c, diagnosis_time,
                predicted_class, rounded_risk_score, risk_score_glucose_50, predicted_class_glucose_50, risk_score_glucose_75,
                predicted_class_glucose_75, risk_score_glucose_100, predicted_class_glucose_100
            ))

            db.commit()
            #mysql.connection.commit()
            cursor.close()
            print("Line 827", latest_diagnosis)
            return render_template('db_simulated_score.html', simulated_report = simulation_report, latest_report = latest_diagnosis,
                                   diagnosis_time = diagnosis_date_str)
        except Exception as e:
            print(e)
            return render_template('db_simulation.html')


@app.route('/report')
def report_page():
    #based on new aws rds database
    cursor = db.cursor()
    # Execute an SQL query to fetch the user with the provided username and password
    # cursor.execute("SELECT Users.*, past_report.* FROM Users JOIN past_report " 
    #                "ON Users.Email = past_report.email WHERE Users.Email = %s "
    #                "ORDER BY past_report.diagnosis_time DESC LIMIT 2;"(person['email'], ))
    cursor.execute("SELECT Users.*, past_report.* FROM Users JOIN past_report "
               "ON Users.Email = past_report.email WHERE Users.Email = %s "
               "ORDER BY past_report.diagnosis_time DESC LIMIT 2;", (person['email'], ))



    #past_report_ref = db.collection("Users").document(person["uid"]).collection("past_report")
    #query = past_report_ref.order_by("diagnosis_time", direction=firestore.Query.DESCENDING)
    #past_report = query.stream()
    past_report = cursor.fetchall()
    #report_list = []
    report_list = past_report
    # for report in past_report:
    #     report_list.append(report.to_dict())
    #     #print(f'{report.id} => {report.to_dict()}')
    # #print(report_list)
    for report in report_list:
        #report['diagnosis_time'] = (report['diagnosis_time'] + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")
        report['diagnosis_time'] = (report['diagnosis_time']).strftime("%Y-%m-%d %H:%M:%S")
        print(report['diagnosis_time'])
    return render_template('report.html', past_report = report_list)
    #return render_template('report.html')

@app.route('/report_detail', methods=['POST', 'GET'])
def report_detail_page():
    if request.method == 'POST':
        #print(request.form)
        result = request.form
        #print(result.to_dict())
        #store the report time when view the report button is clicked, convert to right format
        report_time = result.get('report_time')
        #report_date_time = datetime.datetime.strptime(report_time, '%Y-%m-%d %H:%M:%S') - timedelta(hours=8)
        report_date_time = datetime.strptime(report_time, '%Y-%m-%d %H:%M:%S') #- timedelta(hours=8)
        #print("desired report time in date fromat: ", report_date_time)
        #get the report values specicfic to the report time
        #based on new aws rds database
        cursor = db.cursor()
        # Execute an SQL query to fetch the user with the provided username and password
        cursor.execute("SELECT Users.*, past_report.* FROM Users JOIN past_report " 
                       "ON Users.Email = past_report.email " 
                       "WHERE Users.Email = %s "
                       "ORDER BY past_report.diagnosis_time DESC;", person['email'], )
        #past_report = db.collection("Users").document(person["uid"]).collection("past_report").stream()
        past_report = cursor.fetchall()
        report = {}
        for doc in past_report:
            to_str = doc.get("diagnosis_time").strftime("%Y-%m-%d %H:%M:%S")
            #to_date = datetime.datetime.strptime(to_str, '%Y-%m-%d %H:%M:%S')
            to_date = datetime.strptime(to_str, '%Y-%m-%d %H:%M:%S')
            #print("each doc time in date fromat: ", to_date)
            if to_date == report_date_time:
                #report = doc.to_dict()
                report = doc
        print("The app.py report",report)

        report['bloodtest'] = 0
        if report['HE_sbp'] is not None:
            report['bloodtest'] = 1

        #report['bloodtest'] = 1
        if report['bloodtest'] == 1:
            #print(report['bloodtest'])
            #print(report['advice_list'][1], report['advice_link_list'][1])
            return render_template('report_detail_BT.html', report = report)
        else :
            #print(report['bloodtest'])
            #print(report['advice_list'][1], report['advice_link_list'][1])
            return render_template('report_detail_noBT.html', report = report)

# @app.route('/appointment')
# def appointment_page():
#     return render_template('appointment.html')
@app.route('/about')
def about_page():
    return render_template('about.html')


@app.route('/contact')
def contact_page():
    return render_template('contact.html')


app.register_blueprint(api_blueprint, url_prefix = '/api/v1')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
