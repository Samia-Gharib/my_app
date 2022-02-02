# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from flask import Flask,request, url_for, redirect, render_template, jsonify

import pandas as pd
import pickle
import numpy as np

# Initalise the Flask app
app = Flask(__name__ , template_folder = 'templates')
# Loads pre-trained model
model = pickle.load(open ('./models/model_rf_2.pkl' , 'rb'))
cols = ['his_val', 'age_val', 'ki67_val', 'g_val', 'mamogram_s_val', 'symp_val', 'medical_h' , 'quad_mamo' , 'quad_symp']

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/y_predict',methods=['POST'])
def y_predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    if int(output)== 1:
            output_text ='Metastatic'
    else:
           output_text ='Non-Metastatic'
    return render_template('index.html', prediction_text=' The Patient is More Likely To Be : {}'.format(output_text))

if __name__ == '__main__':
    app.run()








