import time
import datetime
from datetime import date,timedelta
from datetime import datetime, timedelta, timezone

from forms import RegistrationForm, LoginForm
import random
from flask import Flask, render_template, url_for, session, request, redirect, jsonify
import requests
#import firebase_admin
#import pyrebase
import json
#from firebase_admin import credentials, auth, firestore
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
from flask_mysqldb import MySQL
import pymysql

from flask import Flask, request
from flask_restful import Api, Resource


app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba240'

app = Flask(__name__)
app.config['MYSQL_HOST'] = 'diabet-test-shubham.c9cog5gepkr5.ap-southeast-1.rds.amazonaws.com'
app.config['MYSQL_USER'] = 'admin'
app.config['MYSQL_PASSWORD'] = 'Shubham123'
app.config['MYSQL_DB'] = 'diabetic_app'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'  # This makes results return as dictionaries

mysql = MySQL(app)

# Create a connection to the RDS database
db = pymysql.connect(
    host=app.config['MYSQL_HOST'],
    user=app.config['MYSQL_USER'],
    password=app.config['MYSQL_PASSWORD'],
    db=app.config['MYSQL_DB'],
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

#initialize firebase
# cred = credentials.Certificate('fbAdminConfig.json')
# firebase = firebase_admin.initialize_app(cred)
# pb = pyrebase.initialize_app(json.load(open('fbconfig.json')))
# db = firestore.client()

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
model = pickle.load(open("finalized_model.pkl", 'rb'))

# @app.route('/')
# @app.route('/home')
# def home_page():
#     return render_template('home.html')

@app.route('/login', methods=['POST', 'GET'])
def login_page():
    form = LoginForm()
    return render_template('login.html', title='Login', form=form)

# @app.route('/register', methods=['POST', 'GET'])
# def register_page():
#     form = RegistrationForm()
#     return render_template('register.html', title='Register', form=form)

# @app.route('/login_user', methods=['POST', 'GET'])
# def login_user():
#     form = LoginForm()
#     if form.validate_on_submit():
#         if request.method == 'POST':
#             result = request.form
#             email = result.get('email')
#             password = result.get('password')
#             try:
#                 user = pb.auth().sign_in_with_email_and_password(email, password)
#                 #Insert the user data in the global person
#                 global person
#                 person["is_logged_in"] = True
#                 person["email"] = email
#                 person["uid"] = user['localId']
#                 #Get the name of the user
#                 data = db.collection("users").document(person["uid"]).get().to_dict()
#                 person["username"] = data["username"]
#                 person["fullname"] = data["fullname"]
#                 person['dob'] = data["dob"]
#                 person["risk_score_goal"] = data["risk_score_goal"]
#                 #Redirect to home page
#                 return redirect(url_for('home_page'))
#             except Exception as e:
#                 print(e)
#                 return redirect(url_for('login_page'))
#     else:
#         return render_template('login.html', title='Login', form=form)

# @app.route('/register_user', methods=['POST', 'GET'])
# def register_user():
#     form = RegistrationForm()
#     if form.validate_on_submit():
#         if request.method == 'POST':
#             result = request.form
#             email = result.get('email')
#             password = result.get('password')
#             fullname = result.get('fullname')
#             username = result.get('username')
#             dob = result.get('birthday')
#             try:
#                 #Try creating the user account using the provided data
#                 auth.create_user(email=email, password=password)
#                 user = pb.auth().sign_in_with_email_and_password(email, password)
#                 #Add data to global person
#                 global person
#                 person["is_logged_in"] = True
#                 person["email"] = email
#                 person["uid"] = user['localId']
#                 person["fullname"] = fullname
#                 person["username"] = username
#                 person['dob'] = dob
#                 person["risk_score_goal"] = 40
#                 pb.auth().send_email_verification(user['idToken'])
#                 #Append data to the firebase realtime database
#                 data = {"fullname": fullname, "username": username, "email": email, "password": password, 'dob': dob, "risk_score_goal":40}
#                 db.collection("users").document(person["uid"]).set(data)
#                 #Go to home page
#                 return redirect(url_for('home_page'))
#             # if the email is registered, redirect to the login page
#             except firebase_admin._auth_utils.EmailAlreadyExistsError as e:
#                 return redirect(url_for('login_page'))
#             except Exception as e:
#                 print(e)
#                 return redirect(url_for('register_page'))
#     return render_template('register.html', title='Register', form=form)


@app.route('/')
@app.route('/home')
def home_page():

    global person
    person["is_logged_in"] = True

    if person["is_logged_in"] == True:

        #based on new aws rds database
        cursor = db.cursor()
        # Execute an SQL query to fetch the user with the provided username and password
        cursor.execute(
            "SELECT users.*, past_reports.* FROM users "
            "JOIN past_reports ON users.email = past_reports.email "
            "WHERE users.email = 'borghare.sb@gmail.com' "
            "AND past_reports.diagnosis_time IN "
            "(SELECT MAX(diagnosis_time) FROM past_reports "
            "WHERE users.email = past_reports.email "
            "GROUP BY DATE(diagnosis_time)) "
            "ORDER BY past_reports.diagnosis_time DESC;"
        )

        
        #past_reports_ref = db.collection("users").document(person["uid"]).collection("past_reports")
        #data = cursor.fetchone()
        data = cursor.fetchall()
        # print(type(data[0]['diagnosis_time']))
        # print(data[0]['diagnosis_time'].date())
        #print(data[0])
        #risk_score_goal = data['risk_score_goal']
        past_reports_ref = data
        #past_reports_ref = cursor.fetchone()

        # query = past_reports_ref.order_by("diagnosis_time", direction=firestore.Query.DESCENDING).limit(1)
        # results = query.stream()
        results = past_reports_ref
        cursor.close()

        #report_list = []
        report_list = results

        # data = db.collection("users").document(person["uid"]).get().to_dict()
        risk_score_goal = data[0]['risk_score_goal']
        # past_reports_ref = db.collection("users").document(person["uid"]).collection("past_reports")
        # query = past_reports_ref.order_by("diagnosis_time", direction=firestore.Query.DESCENDING).limit(5)
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
            
            print("Diagnosis time:", latest_report['diagnosis_time'])

            date_time_str = latest_report['diagnosis_time'] + timedelta(hours=8)
            latest_diagnosis_date = date_time_str.strftime("%Y-%m-%d")
            #print(report_list[0])
            #print(len(report_list))
            if len(report_list) >= 2:
                #print(report_list[1])
                second_latest_report = report_list[1]
                second_date_time_str = second_latest_report['diagnosis_time'] + timedelta(hours=8)
                second_diagnosis_date = second_date_time_str.strftime("%Y-%m-%d")
                if len(report_list) >= 3:
                    #print(report_list[2])
                    third_latest_report = report_list[2]
                    third_date_time_str = third_latest_report['diagnosis_time'] + timedelta(hours=8)
                    third_diagnosis_date = third_date_time_str.strftime("%Y-%m-%d")
                    if len(report_list) >= 4:
                        #print(report_list[2])
                        fourth_latest_report = report_list[3]
                        fourth_date_time_str = fourth_latest_report['diagnosis_time'] + timedelta(hours=8)
                        fourth_diagnosis_date = fourth_date_time_str.strftime("%Y-%m-%d")
                        if len(report_list) == 5:
                            #print(report_list[2])
                            fifth_latest_report = report_list[4]
                            fifth_date_time_str = fifth_latest_report['diagnosis_time'] + timedelta(hours=8)
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
        response_data = {
            "risk_score_goal": int(risk_score_goal),  # Convert risk_score_goal to an integer
            "name": latest_report["username"],  # User's username from the latest report
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
            return render_template("home.html", risk_score_goal = int(risk_score_goal), name = latest_report["username"], latest_report=latest_report,
                        latest_diagnosis_date=latest_diagnosis_date, second_latest_report=second_latest_report,
                        second_diagnosis_date=second_diagnosis_date, third_latest_report=third_latest_report,
                        third_diagnosis_date=third_diagnosis_date, fourth_latest_report=fourth_latest_report,
                        fourth_diagnosis_date=fourth_diagnosis_date, fifth_latest_report=fifth_latest_report,
                        fifth_diagnosis_date=fifth_diagnosis_date)
    else:
        return redirect(url_for('login'))


@app.route('/db_simulation', methods=['POST', 'GET'])
def db_simulation_page():
    global person
    person["is_logged_in"] = True
    if person["is_logged_in"] == True:
        # latest_report = {"age": 0, "HE_ht": 0, "HE_wt": 0, "HE_wc": 0,
        #                  "HE_sbp": 0, "HE_dbp": 0, "HE_chol": 0, "HE_HDL_st2": 0, "HE_TG": 0,
        #                         "HE_glu": 0, "HE_HbA1c": 0, "HE_BUN": 0, "HE_crea": 0,
        #                         "N_PROT": 0, "N_FAT": 0, "N_CHO": 0,
        #                         "pa_vig_tm": 0,
        #                         "pa_mod_tm": 0, "pa_walkMET": 0, "pa_aerobic": 0,
        #                         "pa_vigMET": 0, "pa_modMET": 0, "pa_totMET": 0}
        latest_report = {"age": 0, "HE_ht": 0, "HE_wt": 0, "HE_wc": 0,
                         "HE_sbp": 0, "HE_dbp": 0, "HE_chol": 0, "HE_HDL_st2": 0, "HE_TG": 0,
                                "HE_glu": 0, "HE_HbA1c": 0, "HE_BUN": 0, "HE_crea": 0,
                                "pa_vig_tm": 0,
                                "pa_mod_tm": 0, "pa_walkMET": 0, "pa_aerobic": 0,
                                "pa_vigMET": 0, "pa_modMET": 0, "pa_totMET": 0}
        diagnosis_str = ''
        diagnosis_date = ''
        global email
        email = 'borghare.sb@gmail.com'
        #based on new aws rds database
        cursor = db.cursor()
       # Execute an SQL query to fetch the user with the provided username and password
        cursor.execute(
            "SELECT users.*, past_reports.* FROM users "
            "JOIN past_reports ON users.email = past_reports.email "
            "WHERE users.email = %s "
            "AND past_reports.diagnosis_time IN "
            "(SELECT MAX(diagnosis_time) FROM past_reports "
            "WHERE users.email = past_reports.email "
            "GROUP BY DATE(diagnosis_time)) "
            "ORDER BY past_reports.diagnosis_time DESC;", (email)
        )

        #past_reports_ref = db.collection("users").document(person["uid"]).collection("past_reports")
        past_reports_ref = cursor.fetchone()

        # query = past_reports_ref.order_by("diagnosis_time", direction=firestore.Query.DESCENDING).limit(1)
        # results = query.stream()
        results = past_reports_ref
        cursor.close()

        #report_list = []
        report_list = results
        # for doc in results:
        #     #report_list.append(doc.to_dict())
        #     report_list.append(doc)
        #print(results['diagnosis_time'])
        if len(report_list) != 0:
            #latest_report = report_list[0]
            latest_report = report_list
            
            date_time_str = latest_report['diagnosis_time'] + timedelta(hours=8)


            diagnosis_str = date_time_str.strftime("%Y-%m-%d %H:%M:%S")
            diagnosis_date = date_time_str.strftime("%Y-%m-%d")
            # print(report_list[0])
            print(report_list)


        # diagnosis_str = "2023-09-10 09:21:18"
        # diagnosis_date = "2023-09-10"

        global latest_diagnosis
        latest_diagnosis = latest_report
        global latest_diagnosis_date
        latest_diagnosis_date = diagnosis_str
        global diagnosis_date_str
        diagnosis_date_str = diagnosis_date
        return render_template('db_simulation.html', latest_report = latest_report, diagnosis_date = diagnosis_str,
                               diagnosis_date_str = diagnosis_date_str)
    # else:
    #     return redirect(url_for('login'))


@app.route('/db_simulation_user', methods=['POST', 'GET'])
def db_simulation_user():
    error = None
    if request.method == 'POST':
        result = request.form
        # get basic information
        sex = float(result.get('sex'))
        age = float(result.get('age'))
        HE_ht = float(result.get('HE_ht'))
        HE_wt = float(result.get('HE_wt'))
        HE_wc = float(result.get('HE_wc'))
        # calculate BMI
        HE_BMI = HE_wt / (HE_ht / 100) ** 2
        # pre-processing for obesity
        if HE_BMI <= 18.5:
            HE_obe = 1
        elif HE_BMI <= 25:
            HE_obe = 2
        else:
            HE_obe = 3

        # Get blood test reusults
        bloodtest = float(result.get('bloodtest'))
        if bloodtest == 1:
            HE_sbp = float(result.get('HE_sbp'))
            HE_dbp = float(result.get('HE_dbp'))
            HE_chol = float(result.get('HE_chol'))
            HE_HDL_st2 = float(result.get('HE_HDL_st2'))
            HE_TG = float(result.get('HE_TG'))
            HE_glu = float(result.get('HE_glu'))
            HE_HbA1c = float(result.get('HE_HbA1c'))
            HE_BUN = float(result.get('HE_BUN'))
            HE_crea = float(result.get('HE_crea'))
        else:
            HE_sbp = None
            HE_dbp = None
            HE_chol = None
            HE_HDL_st2 = None
            HE_TG = None
            HE_glu = None
            HE_HbA1c = None
            HE_BUN = None
            HE_crea = None

        # get lifestyles
        # N_PROT = float(result.get('N_PROT'))
        # N_FAT = float(result.get('N_FAT'))
        # N_CHO = float(result.get('N_CHO'))
        dr_month = float(result.get('dr_month'))
        dr_high = float(result.get('dr_high'))
        sm_presnt = float(result.get('sm_presnt'))
        pa_vig_tm = float(result.get('pa_vig_tm'))
        pa_mod_tm = float(result.get('pa_mod_tm'))
        pa_walkMET = float(result.get('pa_walkMET'))
        pa_aerobic = float(result.get('pa_aerobic'))

        # preprocess for physical activity
        pa_vigMET = round(8 * pa_vig_tm, 2)
        pa_modMET = round(4 * pa_mod_tm, 2)
        pa_totMET = round(pa_walkMET * 3.3 + pa_modMET + pa_vigMET, 2)

        # get history disease
        # DI3_dg = float(result.get('DI3_dg'))
        # DI4_dg = float(result.get('DI4_dg'))
        HE_DMfh = float(result.get('HE_DMfh'))
        DE1_3 = float(result.get('DE1_3'))
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
        HE_HP = None
        if bloodtest == 1:
            if 0 < HE_sbp < 120 and 0 < HE_dbp < 80:
                HE_HP = 1
            elif 120 <= HE_sbp < 140 or 80 <= HE_dbp < 90:
                HE_HP = 2
            elif 140 <= HE_sbp or 90 <= HE_dbp :
                HE_HP = 3
            else:
                HE_HP = None
        else:
            HE_HP = None

        # preprocessing for HE_HCHOL
        HE_HCHOL = 0
        if bloodtest == 1:
            if HE_chol >= 240:
                HE_HCHO = 1
        else:
            HE_HCHOL = 0

        # preprocessign for HE_HTG
        HE_HTG = 0
        if bloodtest == 1:
            if HE_TG >= 200:
                HE_HTG = 1
        else:
            HE_HTG = 0

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
            predicted_class = None
            predicted_class_glucose_50 = None
            predicted_class_glucose_75 = None
            predicted_class_glucose_100 = None

            if bloodtest == 1:
                t = pd.DataFrame(np.array(
                    [pa_totMET, HE_wc, HE_BMI, HE_sbp, HE_dbp, HE_HbA1c, HE_BUN, HE_crea,
                     HE_HDL_st2, HE_TG, age, HE_DMfh, HE_obe, HE_HP, HE_HCHOL,
                     HE_HTG, sm_presnt, sex]).reshape(-1, 18),
                                 columns=['pa_totMET', 'HE_wc', 'HE_BMI', 'HE_sbp',
                                          'HE_dbp', 'HE_HbA1c','HE_BUN', 'HE_crea', 'HE_HDL_st2',
                                          'HE_TG', 'age', 'HE_DMfh', 'HE_obe',
                                          'HE_HP', 'HE_HCHOL', 'HE_HTG', 'sm_presnt', 'sex'])
                diagnosed_class = model.predict(t)
                predicted_class = float(diagnosed_class[0])
                risk_score = model.predict_proba(t)[0][1]
                rounded_risk_score = float(round(risk_score * 100))
                # print(predicted_class)
                # print(rounded_risk_score)

            # risk score prediction for without blood test in confidence interval
            else:
                # generate risk score if HE_glu being in the 25 th percentile to 50 th percentile
                t_50 = pd.DataFrame(np.array(
                    [pa_totMET, HE_wc, HE_BMI, 118.98240115718419, 75.55882352941177,
                     5.534691417550627, 15.428881388621022, 0.8001157184185149, 51.822621449955356, 121.89223722275796,
                     age, HE_DMfh, HE_obe, 1, 0, 0, sm_presnt, sex]).reshape(-1, 18), columns=['pa_totMET','HE_wc', 'HE_BMI', 'HE_sbp',
                                                                        'HE_dbp', 'HE_HbA1c','HE_BUN', 'HE_crea', 'HE_HDL_st2',
                                                                        'HE_TG', 'age', 'HE_DMfh', 'HE_obe',
                                                                        'HE_HP', 'HE_HCHOL', 'HE_HTG', 'sm_presnt', 'sex'])
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
                t_75 = pd.DataFrame(np.array(
                    [pa_totMET, HE_wc, HE_BMI, 124.16230366492147, 77.2324607329843,
                     5.739895287958115, 16.05759162303665, 0.8326178010471204, 49.74235817995025, 142.69476439790577,
                     age, HE_DMfh, HE_obe, 3, 0,
                     0, sm_presnt, sex]).reshape(-1, 18), columns=['pa_totMET', 'HE_wc', 'HE_BMI', 'HE_sbp',
                                                                   'HE_dbp', 'HE_HbA1c','HE_BUN', 'HE_crea',
                                                                   'HE_HDL_st2',
                                                                   'HE_TG', 'age', 'HE_DMfh',
                                                                   'HE_obe',
                                                                   'HE_HP', 'HE_HCHOL', 'HE_HTG', 'sm_presnt', 'sex'])
                diagnosed_class_75 = model.predict(t_75)
                predicted_class_glucose_75 = float(diagnosed_class_75[0])
                risk_score_75 = model.predict_proba(t_75)[0][1]
                risk_score_glucose_75 = float(round(risk_score_75 * 100))
                # print(predicted_class_glucose_75)
                # print(risk_score_glucose_75)

                # generate risk score if HE_glu being in the 75 th percentile to 100th percentile
                t_100 = pd.DataFrame(np.array(
                    [pa_totMET, HE_wc, HE_BMI, 126.92238648363252, 77.0063357972545,
                     6.763727560718057, 16.50897571277719, 0.8618532206969378, 46.931693880777516, 172.6441393875396,
                     age, HE_DMfh, HE_obe, 3, 0,
                     0, sm_presnt, sex]).reshape(-1, 18), columns=['pa_totMET', 'HE_wc', 'HE_BMI', 'HE_sbp',
                                                                   'HE_dbp', 'HE_HbA1c','HE_BUN', 'HE_crea',
                                                                   'HE_HDL_st2',
                                                                   'HE_TG', 'age', 'HE_DMfh',
                                                                   'HE_obe',
                                                                   'HE_HP', 'HE_HCHOL', 'HE_HTG', 'sm_presnt', 'sex'])
                diagnosed_class_100 = model.predict(t_100)
                predicted_class_glucose_100 = float(diagnosed_class_100[0])
                risk_score_100 = model.predict_proba(t_100)[0][1]
                risk_score_glucose_100 = float(round(risk_score_100 * 100))
                print(predicted_class_glucose_100)
                print(risk_score_glucose_100)

                rounded_risk_score = round((risk_score_glucose_50 + risk_score_glucose_75 + risk_score_glucose_100) / 3)

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
                                "sex": sex, "age": age, "HE_ht": HE_ht, "HE_wt": HE_wt, "HE_wc": HE_wc,
                                "HE_BMI": HE_BMI, "HE_obe": HE_obe,
                                "bloodtest": bloodtest, "HE_sbp": HE_sbp, "HE_dbp": HE_dbp, "HE_chol": HE_chol,
                                "HE_HDL_st2": HE_HDL_st2, "HE_TG": HE_TG,
                                "HE_glu": HE_glu, "HE_HbA1c": HE_HbA1c, "HE_BUN": HE_BUN, "HE_crea": HE_crea,
                                "dr_month": dr_month, "dr_high": dr_high, "sm_presnt": sm_presnt,
                                "pa_vig_tm": pa_vig_tm,
                                "pa_mod_tm": pa_mod_tm, "pa_walkMET": pa_walkMET, "pa_aerobic": pa_aerobic,
                                "pa_vigMET": pa_vigMET, "pa_modMET": pa_modMET, "pa_totMET": pa_totMET,
                                "HE_DMfh": HE_DMfh, "DE1_3": DE1_3, "DE1_31": DE1_31, "DE1_32": DE1_32, "HE_HP": HE_HP,
                                "HE_HCHOL": HE_HCHOL, "HE_HTG": HE_HTG}
            
            # Insert user data into the database
            cursor = mysql.connection.cursor()
                
            # Define the SQL INSERT statement
            insert_query = """
            INSERT INTO past_reports (
                email, sex, age, diagnosis_time, diagnosed_class, HE_ht, HE_wt, HE_wc,
                HE_BMI, HE_obe, bloodtest, HE_sbp, HE_dbp, HE_chol, HE_HDL_st2, HE_TG, HE_glu,
                HE_HbA1c, HE_BUN, HE_crea, dr_month, dr_high, sm_presnt, pa_vig_tm, pa_mod_tm,
                pa_walkMET, pa_aerobic, pa_vigMET, pa_modMET, pa_totMET, HE_DMfh, DE1_3, DE1_31,
                DE1_32, HE_HP, HE_HCHOL, HE_HTG, risk_score, risk_score_glucose_50,
                predicted_class_glucose_50, risk_score_glucose_75, predicted_class_glucose_75,
                risk_score_glucose_100, predicted_class_glucose_100
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s
            )
            """
            # Execute the INSERT statement with the data
            cursor.execute(insert_query, (
                email, sex, age, diagnosis_time, predicted_class, HE_ht, HE_wt, HE_wc, HE_BMI, HE_obe, 
                bloodtest, HE_sbp, HE_dbp, HE_chol, HE_HDL_st2, HE_TG, HE_glu, HE_HbA1c, 
                HE_BUN, HE_crea, dr_month, dr_high, sm_presnt, pa_vig_tm, pa_mod_tm, pa_walkMET, 
                pa_aerobic, pa_vigMET, pa_modMET, pa_totMET, HE_DMfh, DE1_3, DE1_31, DE1_32, HE_HP, 
                HE_HCHOL, HE_HTG, rounded_risk_score, risk_score_glucose_50, predicted_class_glucose_50, 
                risk_score_glucose_75, predicted_class_glucose_75, risk_score_glucose_100, 
                predicted_class_glucose_100
            ))

            mysql.connection.commit()
            cursor.close()


            return render_template('db_simulated_score.html', simulated_report = simulation_report, latest_report = latest_diagnosis,
                                   diagnosis_time = diagnosis_date_str)
        except Exception as e:
            print(e)
            return render_template('db_simulation.html')
        
@app.route('/dr_simulation', methods=['POST', 'GET'])
def dr_simulation_page():
    global person
    person["is_logged_in"] = True
    if person["is_logged_in"] == True:
        # latest_report = {"age": 0, "HE_ht": 0, "HE_wt": 0, "HE_wc": 0,
        #                  "HE_sbp": 0, "HE_dbp": 0, "HE_chol": 0, "HE_HDL_st2": 0, "HE_TG": 0,
        #                         "HE_glu": 0, "HE_HbA1c": 0, "HE_BUN": 0, "HE_crea": 0,
        #                         "N_PROT": 0, "N_FAT": 0, "N_CHO": 0,
        #                         "pa_vig_tm": 0,
        #                         "pa_mod_tm": 0, "pa_walkMET": 0, "pa_aerobic": 0,
        #                         "pa_vigMET": 0, "pa_modMET": 0, "pa_totMET": 0}
        latest_report = {"age": 0, "HE_ht": 0, "HE_wt": 0, "HE_wc": 0,
                         "HE_sbp": 0, "HE_dbp": 0, "HE_chol": 0, "HE_HDL_st2": 0, "HE_TG": 0,
                                "HE_glu": 0, "HE_HbA1c": 0, "HE_BUN": 0, "HE_crea": 0,
                                "pa_vig_tm": 0,
                                "pa_mod_tm": 0, "pa_walkMET": 0, "pa_aerobic": 0,
                                "pa_vigMET": 0, "pa_modMET": 0, "pa_totMET": 0}
        diagnosis_str = ''
        diagnosis_date = ''
        global email
        email = 'borghare.sb@gmail.com'
        #based on new aws rds database
        cursor = db.cursor()
       # Execute an SQL query to fetch the user with the provided username and password
        cursor.execute(
            "SELECT users.*, past_reports.* FROM users "
            "JOIN past_reports ON users.email = past_reports.email "
            "WHERE users.email = %s "
            "AND past_reports.diagnosis_time IN "
            "(SELECT MAX(diagnosis_time) FROM past_reports "
            "WHERE users.email = past_reports.email "
            "GROUP BY DATE(diagnosis_time)) "
            "ORDER BY past_reports.diagnosis_time DESC;", (email)
        )
        #past_reports_ref = db.collection("users").document(person["uid"]).collection("past_reports")
        past_reports_ref = cursor.fetchone()

        # query = past_reports_ref.order_by("diagnosis_time", direction=firestore.Query.DESCENDING).limit(1)
        # results = query.stream()
        results = past_reports_ref
        cursor.close()

        #report_list = []
        report_list = results
        # for doc in results:
        #     #report_list.append(doc.to_dict())
        #     report_list.append(doc)
        #print(results['diagnosis_time'])
        if len(report_list) != 0:
            #latest_report = report_list[0]
            latest_report = report_list
            
            date_time_str = latest_report['diagnosis_time'] + timedelta(hours=8)


            diagnosis_str = date_time_str.strftime("%Y-%m-%d %H:%M:%S")
            diagnosis_date = date_time_str.strftime("%Y-%m-%d")
            # print(report_list[0])
            print(report_list)


        # diagnosis_str = "2023-09-10 09:21:18"
        # diagnosis_date = "2023-09-10"

        global latest_diagnosis
        latest_diagnosis = latest_report
        global latest_diagnosis_date
        latest_diagnosis_date = diagnosis_str
        global diagnosis_date_str
        diagnosis_date_str = diagnosis_date
        return render_template('dr_simulation.html', latest_report = latest_report, diagnosis_date = diagnosis_str,
                               diagnosis_date_str = diagnosis_date_str)
    # else:
    #     return redirect(url_for('login'))


@app.route('/dr_simulation_user', methods=['POST', 'GET'])
def dr_simulation_user():
    error = None
    if request.method == 'POST':
        result = request.form
        # get basic information
        sex = float(result.get('sex'))
        age = float(result.get('age'))
        HE_ht = float(result.get('HE_ht'))
        HE_wt = float(result.get('HE_wt'))
        HE_wc = float(result.get('HE_wc'))
        # calculate BMI
        HE_BMI = HE_wt / (HE_ht / 100) ** 2
        # pre-processing for obesity
        if HE_BMI <= 18.5:
            HE_obe = 1
        elif HE_BMI <= 25:
            HE_obe = 2
        else:
            HE_obe = 3

        # Get blood test reusults
        bloodtest = float(result.get('bloodtest'))
        if bloodtest == 1:
            HE_sbp = float(result.get('HE_sbp'))
            HE_dbp = float(result.get('HE_dbp'))
            HE_chol = float(result.get('HE_chol'))
            HE_HDL_st2 = float(result.get('HE_HDL_st2'))
            HE_TG = float(result.get('HE_TG'))
            HE_glu = float(result.get('HE_glu'))
            HE_HbA1c = float(result.get('HE_HbA1c'))
            HE_BUN = float(result.get('HE_BUN'))
            HE_crea = float(result.get('HE_crea'))
        else:
            HE_sbp = None
            HE_dbp = None
            HE_chol = None
            HE_HDL_st2 = None
            HE_TG = None
            HE_glu = None
            HE_HbA1c = None
            HE_BUN = None
            HE_crea = None

        # get lifestyles
        # N_PROT = float(result.get('N_PROT'))
        # N_FAT = float(result.get('N_FAT'))
        # N_CHO = float(result.get('N_CHO'))
        dr_month = float(result.get('dr_month'))
        dr_high = float(result.get('dr_high'))
        sm_presnt = float(result.get('sm_presnt'))
        pa_vig_tm = float(result.get('pa_vig_tm'))
        pa_mod_tm = float(result.get('pa_mod_tm'))
        pa_walkMET = float(result.get('pa_walkMET'))
        pa_aerobic = float(result.get('pa_aerobic'))

        # preprocess for physical activity
        pa_vigMET = round(8 * pa_vig_tm, 2)
        pa_modMET = round(4 * pa_mod_tm, 2)
        pa_totMET = round(pa_walkMET * 3.3 + pa_modMET + pa_vigMET, 2)

        # get history disease
        # DI3_dg = float(result.get('DI3_dg'))
        # DI4_dg = float(result.get('DI4_dg'))
        HE_DMfh = float(result.get('HE_DMfh'))
        DE1_3 = float(result.get('DE1_3'))
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
        HE_HP = None
        if bloodtest == 1:
            if 0 < HE_sbp < 120 and 0 < HE_dbp < 80:
                HE_HP = 1
            elif 120 <= HE_sbp < 140 or 80 <= HE_dbp < 90:
                HE_HP = 2
            elif 140 <= HE_sbp or 90 <= HE_dbp :
                HE_HP = 3
            else:
                HE_HP = None
        else:
            HE_HP = None

        # preprocessing for HE_HCHOL
        HE_HCHOL = 0
        if bloodtest == 1:
            if HE_chol >= 240:
                HE_HCHO = 1
        else:
            HE_HCHOL = 0

        # preprocessign for HE_HTG
        HE_HTG = 0
        if bloodtest == 1:
            if HE_TG >= 200:
                HE_HTG = 1
        else:
            HE_HTG = 0

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
            predicted_class = None
            predicted_class_glucose_50 = None
            predicted_class_glucose_75 = None
            predicted_class_glucose_100 = None

            if bloodtest == 1:
                t = pd.DataFrame(np.array(
                    [pa_totMET, HE_wc, HE_BMI, HE_sbp, HE_dbp, HE_HbA1c, HE_BUN, HE_crea,
                     HE_HDL_st2, HE_TG, age, HE_DMfh, HE_obe, HE_HP, HE_HCHOL,
                     HE_HTG, sm_presnt, sex]).reshape(-1, 18),
                                 columns=['pa_totMET', 'HE_wc', 'HE_BMI', 'HE_sbp',
                                          'HE_dbp', 'HE_HbA1c','HE_BUN', 'HE_crea', 'HE_HDL_st2',
                                          'HE_TG', 'age', 'HE_DMfh', 'HE_obe',
                                          'HE_HP', 'HE_HCHOL', 'HE_HTG', 'sm_presnt', 'sex'])
                diagnosed_class = model.predict(t)
                predicted_class = float(diagnosed_class[0])
                risk_score = model.predict_proba(t)[0][1]
                rounded_risk_score = float(round(risk_score * 100))
                # print(predicted_class)
                # print(rounded_risk_score)

            # risk score prediction for without blood test in confidence interval
            else:
                # generate risk score if HE_glu being in the 25 th percentile to 50 th percentile
                t_50 = pd.DataFrame(np.array(
                    [pa_totMET, HE_wc, HE_BMI, 118.98240115718419, 75.55882352941177,
                     5.534691417550627, 15.428881388621022, 0.8001157184185149, 51.822621449955356, 121.89223722275796,
                     age, HE_DMfh, HE_obe, 1, 0, 0, sm_presnt, sex]).reshape(-1, 18), columns=['pa_totMET','HE_wc', 'HE_BMI', 'HE_sbp',
                                                                        'HE_dbp', 'HE_HbA1c','HE_BUN', 'HE_crea', 'HE_HDL_st2',
                                                                        'HE_TG', 'age', 'HE_DMfh', 'HE_obe',
                                                                        'HE_HP', 'HE_HCHOL', 'HE_HTG', 'sm_presnt', 'sex'])
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
                t_75 = pd.DataFrame(np.array(
                    [pa_totMET, HE_wc, HE_BMI, 124.16230366492147, 77.2324607329843,
                     5.739895287958115, 16.05759162303665, 0.8326178010471204, 49.74235817995025, 142.69476439790577,
                     age, HE_DMfh, HE_obe, 3, 0,
                     0, sm_presnt, sex]).reshape(-1, 18), columns=['pa_totMET', 'HE_wc', 'HE_BMI', 'HE_sbp',
                                                                   'HE_dbp', 'HE_HbA1c','HE_BUN', 'HE_crea',
                                                                   'HE_HDL_st2',
                                                                   'HE_TG', 'age', 'HE_DMfh',
                                                                   'HE_obe',
                                                                   'HE_HP', 'HE_HCHOL', 'HE_HTG', 'sm_presnt', 'sex'])
                diagnosed_class_75 = model.predict(t_75)
                predicted_class_glucose_75 = float(diagnosed_class_75[0])
                risk_score_75 = model.predict_proba(t_75)[0][1]
                risk_score_glucose_75 = float(round(risk_score_75 * 100))
                # print(predicted_class_glucose_75)
                # print(risk_score_glucose_75)

                # generate risk score if HE_glu being in the 75 th percentile to 100th percentile
                t_100 = pd.DataFrame(np.array(
                    [pa_totMET, HE_wc, HE_BMI, 126.92238648363252, 77.0063357972545,
                     6.763727560718057, 16.50897571277719, 0.8618532206969378, 46.931693880777516, 172.6441393875396,
                     age, HE_DMfh, HE_obe, 3, 0,
                     0, sm_presnt, sex]).reshape(-1, 18), columns=['pa_totMET', 'HE_wc', 'HE_BMI', 'HE_sbp',
                                                                   'HE_dbp', 'HE_HbA1c','HE_BUN', 'HE_crea',
                                                                   'HE_HDL_st2',
                                                                   'HE_TG', 'age', 'HE_DMfh',
                                                                   'HE_obe',
                                                                   'HE_HP', 'HE_HCHOL', 'HE_HTG', 'sm_presnt', 'sex'])
                diagnosed_class_100 = model.predict(t_100)
                predicted_class_glucose_100 = float(diagnosed_class_100[0])
                risk_score_100 = model.predict_proba(t_100)[0][1]
                risk_score_glucose_100 = float(round(risk_score_100 * 100))
                print(predicted_class_glucose_100)
                print(risk_score_glucose_100)

                rounded_risk_score = round((risk_score_glucose_50 + risk_score_glucose_75 + risk_score_glucose_100) / 3)

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
                                "sex": sex, "age": age, "HE_ht": HE_ht, "HE_wt": HE_wt, "HE_wc": HE_wc,
                                "HE_BMI": HE_BMI, "HE_obe": HE_obe,
                                "bloodtest": bloodtest, "HE_sbp": HE_sbp, "HE_dbp": HE_dbp, "HE_chol": HE_chol,
                                "HE_HDL_st2": HE_HDL_st2, "HE_TG": HE_TG,
                                "HE_glu": HE_glu, "HE_HbA1c": HE_HbA1c, "HE_BUN": HE_BUN, "HE_crea": HE_crea,
                                "dr_month": dr_month, "dr_high": dr_high, "sm_presnt": sm_presnt,
                                "pa_vig_tm": pa_vig_tm,
                                "pa_mod_tm": pa_mod_tm, "pa_walkMET": pa_walkMET, "pa_aerobic": pa_aerobic,
                                "pa_vigMET": pa_vigMET, "pa_modMET": pa_modMET, "pa_totMET": pa_totMET,
                                "HE_DMfh": HE_DMfh, "DE1_3": DE1_3, "DE1_31": DE1_31, "DE1_32": DE1_32, "HE_HP": HE_HP,
                                "HE_HCHOL": HE_HCHOL, "HE_HTG": HE_HTG}
            
            # Insert user data into the database
            cursor = mysql.connection.cursor()
                
            # Define the SQL INSERT statement
            insert_query = """
            INSERT INTO past_reports (
                email, sex, age, diagnosis_time, diagnosed_class, HE_ht, HE_wt, HE_wc,
                HE_BMI, HE_obe, bloodtest, HE_sbp, HE_dbp, HE_chol, HE_HDL_st2, HE_TG, HE_glu,
                HE_HbA1c, HE_BUN, HE_crea, dr_month, dr_high, sm_presnt, pa_vig_tm, pa_mod_tm,
                pa_walkMET, pa_aerobic, pa_vigMET, pa_modMET, pa_totMET, HE_DMfh, DE1_3, DE1_31,
                DE1_32, HE_HP, HE_HCHOL, HE_HTG, risk_score, risk_score_glucose_50,
                predicted_class_glucose_50, risk_score_glucose_75, predicted_class_glucose_75,
                risk_score_glucose_100, predicted_class_glucose_100
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s
            )
            """
            # Execute the INSERT statement with the data
            cursor.execute(insert_query, (
                "borghare.sb@gmail.com", sex, age, diagnosis_time, predicted_class, HE_ht, HE_wt, HE_wc, HE_BMI, HE_obe, 
                bloodtest, HE_sbp, HE_dbp, HE_chol, HE_HDL_st2, HE_TG, HE_glu, HE_HbA1c, 
                HE_BUN, HE_crea, dr_month, dr_high, sm_presnt, pa_vig_tm, pa_mod_tm, pa_walkMET, 
                pa_aerobic, pa_vigMET, pa_modMET, pa_totMET, HE_DMfh, DE1_3, DE1_31, DE1_32, HE_HP, 
                HE_HCHOL, HE_HTG, rounded_risk_score, risk_score_glucose_50, predicted_class_glucose_50, 
                risk_score_glucose_75, predicted_class_glucose_75, risk_score_glucose_100, 
                predicted_class_glucose_100
            ))

            mysql.connection.commit()
            cursor.close()


            return render_template('dr_simulated_score.html', simulated_report = simulation_report, latest_report = latest_diagnosis,
                                   diagnosis_time = diagnosis_date_str)
        except Exception as e:
            print(e)
            return render_template('dr_simulation.html')

@app.route('/report')
def report_page():
    #based on new aws rds database
    cursor = db.cursor()
    # Execute an SQL query to fetch the user with the provided username and password
    cursor.execute("SELECT users.*, past_reports.* FROM users JOIN past_reports ON users.email = past_reports.email WHERE users.email = 'borghare.sb@gmail.com' ORDER BY past_reports.diagnosis_time DESC LIMIT 2;")
    #past_reports_ref = db.collection("users").document(person["uid"]).collection("past_reports")
    #query = past_reports_ref.order_by("diagnosis_time", direction=firestore.Query.DESCENDING)
    #past_reports = query.stream()
    past_reports = cursor.fetchall()
    #report_list = []
    report_list = past_reports
    # for report in past_reports:
    #     report_list.append(report.to_dict())
    #     #print(f'{report.id} => {report.to_dict()}')
    # #print(report_list)
    for report in report_list:
        report['diagnosis_time'] = (report['diagnosis_time'] + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")
        print(report['diagnosis_time'])
    return render_template('report.html', past_reports = report_list)
    #return render_template('report.html')
    

@app.route('/report_detail', methods=['POST', 'GET'])
def report_detail_page():
    if request.method == 'POST':
        result = request.form
        #print(result.to_dict())
        #store the report time when view the report button is clicked, convert to right format
        report_time = result.get('report_time')
        #report_date_time = datetime.datetime.strptime(report_time, '%Y-%m-%d %H:%M:%S') - timedelta(hours=8)
        report_date_time = datetime.strptime(report_time, '%Y-%m-%d %H:%M:%S') - timedelta(hours=8)
        #print("desired report time in date fromat: ", report_date_time)
        #get the report values specicfic to the report time
        #based on new aws rds database
        cursor = db.cursor()
        # Execute an SQL query to fetch the user with the provided username and password
        cursor.execute("SELECT users.*, past_reports.* FROM users JOIN past_reports ON users.email = past_reports.email WHERE users.email = 'borghare.sb@gmail.com' ORDER BY past_reports.diagnosis_time DESC;")
        
        #past_reports = db.collection("users").document(person["uid"]).collection("past_reports").stream()
        past_reports = cursor.fetchall()
        report = {}
        for doc in past_reports:
            to_str = doc.get("diagnosis_time").strftime("%Y-%m-%d %H:%M:%S")
            #to_date = datetime.datetime.strptime(to_str, '%Y-%m-%d %H:%M:%S')
            to_date = datetime.strptime(to_str, '%Y-%m-%d %H:%M:%S')
            #print("each doc time in date fromat: ", to_date)
            if to_date == report_date_time:
                #report = doc.to_dict()
                report = doc
        #print(report)
        if report['bloodtest'] == 1:
            #print(report['bloodtest'])
            #print(report['advice_list'][1], report['advice_link_list'][1])
            return render_template('report_detail_BT.html', report = report)
        else :
            #print(report['bloodtest'])
            #print(report['advice_list'][1], report['advice_link_list'][1])
            return render_template('report_detail_noBT.html', report = report)

#@app.route('/diagnosis_report')
#def diagnosis_report_page():
#    return render_template('report_detail.html', report = diagnosis_report)

# #take in  feature values
# #return top 3 most important features with values exceeding diabetic level
# #test
# # bloodtest = 0
# # HE_HbA1c = 7
# # HE_TG = 170
# # HE_HP = 4
# # pa_totMET = 1534
# # HE_HDL_st2 = 47
# # HE_HTG = 0
# # HE_dbp = 74.6373182552504
# # HE_sbp = 126.7011308562197
# # HE_BUN = 17.04604200323102
# # sm_presnt = 1
# # HE_crea = 0.8868659127625202 
# # HE_obe = 4
# # HE_HCHOL = 0
# # HE_BMI = 26
# def top_advice(bloodtest,HE_HbA1c,HE_TG,HE_HP,pa_totMET,HE_HDL_st2,HE_HTG,HE_dbp,HE_sbp,HE_BUN,sm_presnt,HE_crea,HE_obe,HE_HCHOL,HE_BMI):
#     featureValue = []
#     if bloodtest == 1:
#         featureValue = [HE_HbA1c,HE_TG,HE_HP,pa_totMET,HE_HDL_st2,HE_HTG,HE_dbp,HE_sbp,HE_BUN,sm_presnt,HE_crea]

#         HE_HbA1c_normal = 5.6238456955615135
#         HE_TG_normal  = 133.24337205838546
#         HE_HP_normal  = 1
#         pa_totMET_normal  = 1729.2093535895146
#         HE_HDL_st2_normal  = 50.920450758467354
#         HE_HTG_normal  = 0
#         HE_dbp_normal  = 76.6131963062258
#         HE_sbp_normal  = 121.27256478999107
#         HE_BUN_normal  =  15.61423890378314
#         sm_presnt_normal  = 0
#         HE_crea_normal  = 0.8107819481680072
#         normalValue = [HE_HbA1c_normal, HE_TG_normal, HE_HP_normal, pa_totMET_normal, HE_HDL_st2_normal, 
#                         HE_HTG_normal, HE_dbp_normal, HE_sbp_normal, HE_BUN_normal, sm_presnt_normal,HE_crea_normal] #ordered by importance
#         featureName = ["Hemoglobin_A1c (%)", "Triglycerides (mg/dL)", "Hypertension status",
#                        "Total MET (min/week)", "HDL cholesterol (mg/dL)", "Hyper triglycerides status", "Diastolic blood pressure (mmHg)", 
#                         "Systolic blood pressure (mmHg)", "Blood urea nitrogen (mg/dL)", "Current smoking status", 
#                         "Blood serum creatinine (mg/dL)"] #ordered by importance
#         adviceList = ["Reduce Hemoglobin_A1c:", "Reduce Triglycerides:", "About hypertension:","Physical activity and diabetes:",
#                       "Cholesteral and diabetes:", "Reduce hypertriglycerides:", "Diastolic blood pressure:",
#                       "Systolic blood pressure:", "About blood urea nitrogen", "Smoking and diabetes:", "About blood serum creatinine"] #same order
#         adviceLink = ["https://www.everydayhealth.com/type-2-diabetes/treatment/ways-lower-your-a1c/",
#                      "https://www.webmd.com/cholesterol-management/lowering-triglyceride-levels",
#                      "https://www.medicalnewstoday.com/articles/150109#diet",
#                      "https://www.cdc.gov/diabetes/managing/active.html#:~:text=If%20you%20have%20diabetes%2C%20being,heart%20disease%20and%20nerve%20damage.",
#                      "https://www.heart.org/en/health-topics/diabetes/diabetes-complications-and-risks/cholesterol-abnormalities--diabetes",
#                      "https://my.clevelandclinic.org/health/diseases/23942-hypertriglyceridemia#:~:text=A%20normal%20triglyceride%20level%20in,150%20mg%2FdL%20or%20higher.",
#                      "https://www.uab.edu/news/research/item/10393-diastolic-blood-pressure-how-low-is-too-low",
#                      "https://www.cdc.gov/bloodpressure/about.htm",
#                      "https://labs.selfdecode.com/blog/causes-of-high-or-low-blood-urea-nitrogen-bun/",
#                      "https://www.cdc.gov/tobacco/campaign/tips/diseases/diabetes.html#:~:text=We%20now%20know%20that%20smoking%20is%20one%20cause%20of%20type%202%20diabetes.&text=In%20fact%2C%20people%20who%20smoke,people%20who%20don't%20smoke.&text=People%20with%20diabetes%20who%20smoke,and%20with%20managing%20their%20condition.",
#                      "https://www.medicalnewstoday.com/articles/322380"]
#     else:
#         featureValue = [pa_totMET,HE_HTG,sm_presnt,HE_obe,HE_HCHOL,HE_BMI]
#         pa_totMET_normal  = 1729.2093535895146
#         HE_HTG_normal  = 0
#         sm_presnt_normal  = 0
#         HE_obe_normal = 2
#         HE_HCHOL_normal = 0
#         HE_BMI_normal = 23.879052007717448
#         normalValue = [pa_totMET_normal, HE_HTG_normal, sm_presnt_normal, HE_obe_normal, HE_HCHOL_normal, HE_BMI_normal]
#         featureName = ["Total MET (min/week)", "Hyper triglycerides status", "Current smoking status", 
#                        "Obesity status", "Hyperlipidemia status", "Body mass index (kg/m2)"]
#         adviceList = ["Physical activity and diabetes:", "Reduce hypertriglycerides:", "Smoking and diabetes:", 
#                      "Obesity and diabetes:","About hyperlipidemia:", "BMI as a risk factor of diabetes:"] #same order
#         adviceLink = ["https://www.cdc.gov/diabetes/managing/active.html#:~:text=If%20you%20have%20diabetes%2C%20being,heart%20disease%20and%20nerve%20damage.",
#                       "https://my.clevelandclinic.org/health/diseases/23942-hypertriglyceridemia#:~:text=A%20normal%20triglyceride%20level%20in,150%20mg%2FdL%20or%20higher.",
#                       "https://www.cdc.gov/tobacco/campaign/tips/diseases/diabetes.html#:~:text=We%20now%20know%20that%20smoking%20is%20one%20cause%20of%20type%202%20diabetes.&text=In%20fact%2C%20people%20who%20smoke,people%20who%20don't%20smoke.&text=People%20with%20diabetes%20who%20smoke,and%20with%20managing%20their%20condition.",
#                       "https://www.diabetes.co.uk/diabetes-and-obesity.html",
#                       "https://www.healthhub.sg/a-z/diseases-and-conditions/622/hyperlipidemia",
#                       "https://www.escardio.org/The-ESC/Press-Office/Press-releases/Body-mass-index-is-a-more-powerful-risk-factor-for-diabetes-than-genetics#:~:text=The%20highest%20BMI%20group%20had,groups%2C%20regardless%20of%20genetic%20risk."]
#     topAdvice = []
#     topAdviceLink = []
#     count = 0
#     for i in range(len(featureValue)):
#         print(featureValue[i])
#         if count == 3:
#             break
#         if  featureValue[i] > normalValue[i]:
#             topAdvice.append(adviceList[i])
#             topAdviceLink.append(adviceLink[i])            
#             count += 1
#     if len(topAdvice) == 0:
#         topAdvice.append("")
#         topAdvice.append("")
#         topAdvice.append("")
#         topAdviceLink.append("")
#         topAdviceLink.append("")
#         topAdviceLink.append("")
#     elif len(topAdvice) == 1:
#         topAdvice.append("")
#         topAdvice.append("")
#         topAdviceLink.append("")
#         topAdviceLink.append("")
#     elif len(topAdvice) == 2:
#         topAdvice.append("")
#         topAdviceLink.append("")

#     return topAdvice, topAdviceLink

# # print(top_advice(bloodtest,HE_HbA1c,HE_TG,HE_HP,pa_totMET,HE_HDL_st2,HE_HTG,HE_dbp,HE_sbp,HE_BUN,sm_presnt,HE_crea,HE_obe,HE_HCHOL,HE_BMI))

# @app.route('/appointment')
# def appointment_page():
#     return render_template('appointment.html')

@app.route('/about')
def about_page():
    return render_template('about.html')


@app.route('/contact')
def contact_page():
    return render_template('contact.html')


# @app.route('/profile')
# def profile_page():
#     data = db.collection("users").document(person["uid"]).get().to_dict()
#     risk_score_goal = data['risk_score_goal']
#     return render_template('profile.html', risk_score_goal = risk_score_goal, username = person["username"], fullname = person["fullname"], email = person["email"], dob = person["dob"])

# @app.route('/change_username', methods=['GET', 'POST'])
# def change_username():
#     if request.method == 'POST':
#         result = request.form
#         username = result.get('username')
#         person['username'] = username
#         try: 
#             #change username to the firebase realtime database
#             data = {"username": username}
#             db.collection("users").document(person["uid"]).update(data)
#             return redirect(url_for('profile_page'))
#         except Exception as e:
#             print(e)
#             return redirect(url_for('profile_page'))
    
# @app.route('/change_riskscore', methods=['GET', 'POST'])
# def change_riskscore():
#     if request.method == 'POST':
#         result = request.form
#         risk_score_goal = result.get('risk_score_goal')
#         person['risk_score_goal'] = risk_score_goal
#         try: 
#             #change username to the firebase realtime database
#             data = {"risk_score_goal": risk_score_goal}
#             db.collection("users").document(person["uid"]).update(data)
#             return redirect(url_for('profile_page'))
#         except Exception as e:
#             print(e)
#             return redirect(url_for('profile_page'))

# @app.route('/delete_account')
# def delete_account():
#     try: 
#         user = auth.get_user_by_email(person['email'])
#         auth.delete_user(user.uid)
#         db.collection("users").document(person["uid"]).delete()
#         return redirect(url_for('login_page'))
#     except Exception as e:
#         print(e)
#         return redirect(url_for('profile_page'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)