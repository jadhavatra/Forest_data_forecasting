import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler



app = Flask(__name__)

ridge_model = pickle.load(open('models/ridge_model.pkl','rb'))
scaler = pickle.load(open('models/scalar.pkl','rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled=scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)
        result = round(result[0],2)

        return render_template('home.html',result=result)

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")
