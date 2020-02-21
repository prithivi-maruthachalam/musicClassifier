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


timeA = time.time()
header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1,21):
	header += ' mfcc'+str(i)
header += ' label'
header =  header.split()

with open('data.csv','w') as file:
	writer = csv.writer(file)
	writer.writerow(header)
	genres = "blues classical country disco hiphop jazz metal pop reggae rock".split()
	for g in genres:
		currentFilePath = "dataset/" + g + "/"
		for filename in os.listdir(currentFilePath):
			rmse  = 1
			songname = currentFilePath + filename
			y,sr = librosa.load(songname,mono=True,duration=30)
			chroma_stft = librosa.feature.chroma_stft(y=y,sr=sr)
			spec_cent = librosa.feature.spectral_centroid(y=y,sr=sr)
			spec_bw = librosa.feature.spectral_bandwidth(y=y,sr=sr)
			rolloff = librosa.feature.spectral_rolloff(y=y,sr=sr)
			zcr = librosa.feature.zero_crossing_rate(y)
			mfcc = librosa.feature.mfcc(y=y,sr=sr)
			
			to_append = filename + ' ' + str(np.mean(chroma_stft)) + " " + str(np.mean(rmse)) + " " + str(np.mean(spec_cent)) + " " +  str(np.mean(spec_bw)) + " " + str(np.mean(rolloff)) + " " + " " + str(np.mean(zcr))
			for e in mfcc:
				to_append += " " + str(np.mean(e))
			to_append += " " + g
			to_append = to_append.split()
			writer.writerow(to_append)

print("[INFO]: Done.")
print("[INFO]: That took " + str(time.time()-timeA) + " seconds")
