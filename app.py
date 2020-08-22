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

# def build_model():
#     model = Sequential()

#     model.add(Dense(40,activation='relu', input_shape=(16,)))
#     model.add(Dense(60,activation='relu'))
#     model.add(Dropout(0.3))
#     model.add(Dense(40,activation='relu'))
#     model.add(Dropout(0.3))
#     model.add(Dense(1,activation='sigmoid'))

#     model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])

#     return model
# model=build_model()
# model.load_weights("train_diabetes_model.h5")


@app.route('/')
@app.route('/index')
def index():
    title="Welcome to my web"
    return render_template('index.html',title=title)

@app.route('/predict')
def pred():
    return render_template('predict.html')


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

@app.route('/predict',methods=['GET','POST'])
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

    # if float(risk_rate) > 50.0:
    #     return render_template('predict.html',view= "Yor prediction of the diabetes risk rate is "+risk_rate+
    #     " % ."+"<div>I suggest you go to the hospital to check your health quickly!!</div>")
    #     #
    # else:
    #     return render_template('predict.html',view= "Yor prediction of the diabetes risk rate is "+risk_rate+
    #     " % .")

    

    
if __name__=='__main__':
    app.run(debug=False)#,host='0.0.0.0',port=8080