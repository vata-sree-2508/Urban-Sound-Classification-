import librosa
import matplotlib.pyplot as plt
import librosa.display
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import tensorflow 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
from sklearn.model_selection import train_test_split

audio_dataset_path='C:\\Users\\ramak\\Documents\\Datasets\\Audio Datasets\\UrbanSound8K.tar\\UrbanSound8K\\audio'
metadata_path = 'C:\\Users\\ramak\\Documents\\Datasets\\Audio Datasets\\UrbanSound8K.tar\\UrbanSound8K\\metadata\\UrbanSound8k.csv'
metadata = pd.read_csv(metadata_path)
# print(metadata.head())

def feature_extractor(file_name):
  audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
  mfccs_feature = librosa.feature.mfcc(audio, sample_rate, n_mfcc = 40)
  mfcc_scaled = np.mean(mfccs_feature.T, axis=0)
  return mfcc_scaled

extracted_features_path = r'C:\\Users\\ramak\\Documents\\Sounds Project\\npy\\extracted_features_np.npy'
extracted_features_pkl = np.load(extracted_features_path,allow_pickle=True)

# print(extracted_features_pkl.shape)

extracted_features_df =pd.DataFrame(extracted_features_pkl,columns=['feature','class'])
# print(extracted_features_df.head())

X = np.array(extracted_features_df['feature'].tolist())
y = np.array(extracted_features_df['class'].tolist())
print(X.shape,y.shape)

labelencoder = LabelEncoder()
y1 = to_categorical(labelencoder.fit_transform(y))
print(y1.shape)

X_train,X_test,y_train,y_test=train_test_split(X,y1,test_size=0.25,random_state=0)
print(y_train.shape,y_test.shape)

def get_model_1():
  model = Sequential()
  model.add(Dense(100,input_shape=(40,)))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))

  model.add(Dense(200))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))

  model.add(Dense(100))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))

  model.add(Dense(10))
  model.add(Activation('softmax'))
  return model

# model = get_model_1()
# print(model.summary())  
def compile_model(model):
  model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

saved_model_path = 'C:\\Users\\ramak\\Documents\\Sounds Project\\models\\audio_classification.hdf5'
checkpointer = ModelCheckpoint(filepath=saved_model_path,verbose=1,save_best_only=True)
num_epochs = 100
num_batch_size = 32

def get_reconstructed_model(saved_model_path):
  reconstructed_model = tensorflow.keras.models.load_model(saved_model_path)
  return reconstructed_model

reconstructed_model = get_reconstructed_model(saved_model_path)
print(reconstructed_model.summary())
print('model reconstructed from disk')



def model_fit(reconstructed_model):
  reconstructed_model.fit(X_train, y_train, batch_size= num_batch_size,epochs=num_epochs, validation_data=(X_test, y_test),callbacks=[checkpointer])

# print('Training start!')
# model_fit(reconstructed_model)
# print('Training end!')

testing = reconstructed_model.evaluate(X_test,y_test, verbose=0)
print(testing)

new_audio_path = 'C:\\Users\\ramak\\Documents\\Datasets\\Audio Datasets\\Mini Urban Sound8K\\Class 5\\146186-5-0-13.wav'
def testing_new_audio(new_audio_path):
  prediction_feature = feature_extractor(new_audio_path)
  prediction_feature = prediction_feature.reshape(-1,1).T
  # prediction_feature.shape
  predicted_class = np.argmax(reconstructed_model.predict(prediction_feature))
  print(predicted_class)

testing_new_audio(new_audio_path)  
