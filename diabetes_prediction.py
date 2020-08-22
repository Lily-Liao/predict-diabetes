# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 18:13:15 2020

@author: user
"""


import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
import tensorflow.compat.v1 as tf #使用1.0版本的方法


model = load_model('train_diabetes_model.h5')

def PreprocessData(all_df):
  #將描述數字化
  all_df['Gender'] = all_df['Gender'].map({'Female':0,'Male':1}).astype(int)
  cols=['Polyuria','Polydipsia','sudden weight loss','weakness','Polyphagia','Genital thrush','visual blurring','Itching','Irritability','delayed healing','partial paresis','muscle stiffness','Alopecia','Obesity']
  for col in cols:
    all_df[col] = all_df[col].map({'No':0,'Yes':1}).astype(int)
  all_df['class'] = all_df['class'].map({'Negative':0,'Positive':1}).astype(int)
  #將dataframe轉成array
  ndarray = all_df.values
  #擷取 feacture & label
  label = ndarray[:,-1]
  feacture = ndarray[:,:-1]
  #將特徵標準化在0~1之間，能夠更有效收斂
  feacture_normalization = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(feacture)

  return feacture_normalization,label

def prediction_rate(new_data):
  all_df=pd.read_csv('diabetes_data_upload.csv')
  cols=['Age','Gender','Polyuria','Polydipsia','sudden weight loss','weakness','Polyphagia','Genital thrush','visual blurring','Itching','Irritability','delayed healing','partial paresis','muscle stiffness','Alopecia','Obesity','class']
  dic={}
  for j in range(len(cols)):
    dic[str(cols[j])]=new_data[j]
  new_dataframe=pd.DataFrame(dic,index=[1])
  new_all_df=all_df.append(new_dataframe,ignore_index=True)
  predict_feacture, predict_label = PreprocessData(new_all_df)
  predict_all = model.predict(predict_feacture)
  prediction = predict_all[-1]
  return prediction

if __name__ == '__main__':
    new_data=['40','Female','No','Yes','No','No','No','No','No','No','No','No','No','No','Yes','No','Negative']#77.34
    predict_rate=prediction_rate(new_data)
    print(predict_rate)
    
    new_data5=['60','Male','Yes','No','Yes','Yes','No','Yes','No','Yes','Yes','No','Yes','No','Yes','Yes','Negative']
    predict_rate2=prediction_rate(new_data5)
    print(predict_rate2)
