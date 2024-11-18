from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
app = Flask(__name__)

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            customerID=request.form['customerID'],
            gender=request.form['gender'],
            SeniorCitizen=int(request.form['SeniorCitizen']),
            Partner=request.form['Partner'],
            Dependents=request.form['Dependents'],
            tenure=int(request.form['tenure']),
            PhoneService=request.form['PhoneService'],
            MultipleLines=request.form['MultipleLines'],
            InternetService=request.form['InternetService'],
            OnlineSecurity=request.form['OnlineSecurity'],
            OnlineBackup=request.form['OnlineBackup'],
            DeviceProtection=request.form['DeviceProtection'],
            TechSupport=request.form['TechSupport'],
            StreamingTV=request.form['StreamingTV'],
            StreamingMovies=request.form['StreamingMovies'],
            Contract=request.form['Contract'],
            PaperlessBilling=request.form['PaperlessBilling'],
            PaymentMethod=request.form['PaymentMethod'],
            MonthlyCharges=float(request.form['MonthlyCharges']),
            TotalCharges=float(request.form['TotalCharges'])
        )
        data_df = data.to_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(data_df)
        message = 'Customer is likely to churn' if pred == 1 else 'Customer is not likely to churn'
        return render_template('home.html',message=message)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)