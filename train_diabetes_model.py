# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 18:19:01 2020

@author: user
"""


import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

# import tensorflow as tf
# from tensorflow.python.keras import backend as K
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# config.gpu_options.allow_growth = True
# K.set_session(tf.compat.v1.Session(config=config)) 

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

def show_train_history(train_history,train,validation):
  plt.plot(train_history.history[train])
  plt.plot(train_history.history[validation])
  plt.title('Train History')
  plt.ylabel(train)
  plt.xlabel('Epoch')
  plt.legend(['train','validation'],loc='best')
  plt.show()
  
##############################################################################
df=pd.read_csv('diabetes_data_upload.csv')
X,Y = PreprocessData(df)
x_train,x_test,y_train,y_test=train_test_split(X, Y, test_size=0.2, random_state=42)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

##############################################################################
model = Sequential()

model.add(Dense(40,activation='relu', input_shape=(16,)))
model.add(Dense(60,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(40,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
train_history = model.fit(x=x_train, y=y_train, validation_split=0.2, epochs=10, batch_size=16, verbose=2)

model.save('train_diabetes_model.h5')

##############################################################################
show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')

scores = model.evaluate(x=x_test,y=y_test)
print(scores[1])