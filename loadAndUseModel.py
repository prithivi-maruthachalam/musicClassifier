#music handling libraries
import librosa
import pandas as pd
import numpy as np

import time
import sys

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

data = pd.read_csv('testFile.csv')
data = data.drop(['filename'],axis=1)



#importing the trained model
json_file = open("trainedModel.json",'r')
loaded_model = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model)
loaded_model.load_weights("model.h5")

print("[INFO]: Loaded model")

model = loaded_model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

data = data.iloc[:,:-1]
print("-----------------------------------------")
songFile = data[0:1]
print(songFile)

predictions = model.predict(songFile)
print(np.argmax(predictions[0]))
