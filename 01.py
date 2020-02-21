import librosa
import matplotlib.pyplot as plt
import librosa.display as libPlt
import numpy as np
import sklearn

SR = 44100

filePath = "dataset/blues/blues.00001.wav"
x,SR = librosa.load(filePath,sr=SR)

chromagram = librosa.feature.chroma_stft(x,sr=SR,hop_length=512)
plt.figure(figsize=(15,5))
libPlt.specshow(chromagram,x_axis='time',y_axis='chroma',hop_length=512,cmap='coolwarm')

plt.show()
