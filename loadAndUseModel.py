#music handling libraries
import librosa
import pandas as pd
import numpy as np

import time

import matplotlib.pyplot as plt
import os
from PIL import Image
import pathlib
import csv

#data preprocessing libraries
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split

#libraries for the neural network
from keras import models
from keras import layers

from keras.models import model_from_json


data = pd.read_csv('data.csv')
data = data.drop(['filename'],axis=1)

#this just gets the labels and assigns integers to them
genre_list = data.iloc[:,-1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)

#scaling feature coloumns
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:,:-1],dtype=float))

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

x_val = X_train[:200]
partial_x_train = X_train[200:]

y_val = y_train[:200]
partial_y_train = y_train[200:]

json_file = open("trainedModel.json",'r')
loaded_model = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model)
loaded_model.load_weights("model.h5")

print("[INFO]: Loaded model")

model = loaded_model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


newExample = X_test[0:1]
predictions = model.predict(newExample)

genres = "blues classical country disco hiphop jazz metal pop reggae rock".split()
print(genres[np.argmax(predictions[0])])
