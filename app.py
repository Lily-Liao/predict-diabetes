from flask import Flask,render_template,request
from diabetes_prediction import PreprocessData
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.framework import ops

#import tensorflow.python.keras.backend as K 

app=Flask(__name__)


@app.route('/')
@app.route('/index-zh')
def index():
    title="歡迎使用糖尿病風險預測網站"
    return render_template('index.html',title=title)

@app.route('/index-en')
def index1():
    title="Welcome to the diabetes risk prediction web"
    return render_template('index1.html',title=title)    

@app.route('/predict-zh')
def pred():
    return render_template('predict.html')

@app.route('/predict-en')
def pred1():
    return render_template('predict1.html')

def prediction_rate(new_data):

    global graph, model
    model=load_model("train_diabetes_model.h5")
    graph = tf.get_default_graph()

    all_df=pd.read_csv('diabetes_data_upload.csv')
    cols=['Age','Gender','Polyuria','Polydipsia','sudden weight loss','weakness','Polyphagia','Genital thrush','visual blurring','Itching','Irritability','delayed healing','partial paresis','muscle stiffness','Alopecia','Obesity','class']
    dic={}
    for j in range(len(cols)):
        dic[str(cols[j])]=new_data[j]
    new_dataframe=pd.DataFrame(dic,index=[1])
    new_all_df=all_df.append(new_dataframe,ignore_index=True)
    predict_feacture, predict_label = PreprocessData(new_all_df)

    with graph.as_default():
        predict_all = model.predict(predict_feacture)
    prediction = predict_all[-1]
    return prediction

@app.route('/predict-zh',methods=['GET','POST'])
def submit():
    new_data=[]
    age = request.values['age']
    sex = request.values['sex']
    polyuria = request.values['polyuria']
    polydipsia = request.values['polydipsia']
    sudden_weight_loss = request.values['sudden weight loss']
    weakness = request.values['weakness']
    polyphagia = request.values['polyphagia']
    genital_thrush = request.values['genital thrush']
    visual_blurring = request.values['visual blurring']
    itching = request.values['itching']
    irritability = request.values['irritability']
    delayed_healing = request.values['delayed healing']
    partial_paresis = request.values['partial paresis']
    muscle_stiffness = request.values['muscle stiffness']
    alopecia = request.values['alopecia']
    obesity = request.values['obesity']

    new_data.extend((age,sex,polyuria,polydipsia,sudden_weight_loss,weakness,polyphagia,
    genital_thrush,visual_blurring,itching,irritability,delayed_healing,partial_paresis,
    muscle_stiffness,alopecia,obesity,'Negative'))
    print(new_data)
    predict_rate=prediction_rate(new_data)
    risk_rate=float(format(predict_rate[0]*100 , '.2f'))
    print(risk_rate)
    return render_template('predict.html',risk_rate=risk_rate)


@app.route('/predict-en',methods=['GET','POST'])
def submit1():
    new_data=[]
    age = request.values['age1']
    sex = request.values['sex1']
    polyuria = request.values['polyuria1']
    polydipsia = request.values['polydipsia1']
    sudden_weight_loss = request.values['sudden weight loss1']
    weakness = request.values['weakness1']
    polyphagia = request.values['polyphagia1']
    genital_thrush = request.values['genital thrush1']
    visual_blurring = request.values['visual blurring1']
    itching = request.values['itching1']
    irritability = request.values['irritability1']
    delayed_healing = request.values['delayed healing1']
    partial_paresis = request.values['partial paresis1']
    muscle_stiffness = request.values['muscle stiffness1']
    alopecia = request.values['alopecia1']
    obesity = request.values['obesity1']

    new_data.extend((age,sex,polyuria,polydipsia,sudden_weight_loss,weakness,polyphagia,
    genital_thrush,visual_blurring,itching,irritability,delayed_healing,partial_paresis,
    muscle_stiffness,alopecia,obesity,'Negative'))
    print(new_data)
    predict_rate=prediction_rate(new_data)
    risk_rate=float(format(predict_rate[0]*100 , '.2f'))
    print(risk_rate)
    return render_template('predict1.html',risk_rate=risk_rate)   

    
if __name__=='__main__':
    app.run(debug=False)#,host='0.0.0.0',port=8080