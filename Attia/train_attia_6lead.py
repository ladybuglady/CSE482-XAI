# Attia et. al. model with 6-lead ECG Data
# 6 leads are reduced to 2 independent leads

# ~~~~~~~~~~~~~~~ IMPORTS ~~~~~~~~~~~~~~~
from collections import Counter
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from keras.models import Sequential,Model
from keras.layers import concatenate,Activation,Dropout,Dense,ZeroPadding2D
from keras.layers import Input,add,Conv2D, MaxPooling2D,Flatten,BatchNormalization,LSTM
import os
import zipfile
from pandas import *
from sklearn.model_selection import train_test_split


# ~~~~~~~~~~~~~~~ DATA FETCH ~~~~~~~~~~~~~~~
dir_path = '/karman_netid/local1'

count = 0
# Iterate directory
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        count += 1
print('File count:', count)

with zipfile.ZipFile("/content/drive/MyDrive/Colab Notebooks/XAI_DATA/Copy of study60_recordings_json.zip","r") as zip_ref:
    zip_ref.extractall("/content/drive/MyDrive/Colab Notebooks/XAI_DATA/recordings")

import json
import os
from keras.utils import to_categorical

dir_path = '/content/drive/MyDrive/Colab Notebooks/XAI_DATA/recordings/recording_batch/'
filenames = []

X = None
y_labels = []
y = None

# Iterate directory
for path in os.listdir(dir_path):
  patient_X = np.empty((2, 5000))

  jsonFile = open(dir_path + path, 'r')
  fileContents = json.load(jsonFile)

  # digging into the dictionaries to get lead data
  lead_1_samples = fileContents['samples']
  lead_2_samples = fileContents['extraLeads'][0]['samples']

  # Crop the data to 5000 data points.
  patient_X[0,:] = lead_1_samples[0:5000]
  patient_X[1,:] = lead_2_samples[0:5000]
  
  if X is None:
    X = np.expand_dims(patient_X, axis=0)
  else:
    X = np.concatenate((X, np.expand_dims(patient_X, axis=0)), axis=0)

  recording_id = fileContents['filename'][:-9]
  filenames.append(recording_id)
  diagnostics_file = filename_csv.index(recording_id)
  filenames.append(diagnostics_file)
  if y_csv[diagnostics_file] == 'Sinus Rhythm':
    y_labels.append(0)
  else:
    y_labels.append(1)
  jsonFile.close()

y_labels = np.asarray(y_labels)

# ~~~~~~~~~~~~~~~ DATA PREPROCESS ~~~~~~~~~~~~~~~

# Change X dims for model compatibility
# expected shape=(None, 5000, 2, 1)
X = np.swapaxes(X,1,2)
X = np.expand_dims(X, axis=3)

# Change y dims to one-hot encoding, should be (300, 2)
y = to_categorical(y_labels, dtype ="uint8")

X_train, X_rem, y_train, y_rem = train_test_split(X,y, train_size=0.8)
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)

N_Val = y_valid.shape[0]
print ('Training on : ' + str(N_Train) + ' and validating on : ' +str(N_Val))

n_classes = 2

# ~~~~~~~~~~~~~~~ TRAIN MODEL ~~~~~~~~~~~~~~~
class_weight={}

model = BuildModel(segmentLength=int(5000),
                           padTo=int(5120),n_classes=n_classes,reluLast=True)


earlyStopCallback = EarlyStopping(monitor='val_loss', min_delta=0, patience=9,  mode='auto')
saveBestCallback = ModelCheckpoint(modelName,monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
reduceLR =ReduceLROnPlateau(monitor = 'val_loss',factor = 0.5,patience = 3,verbose=1,min_lr = 0.00001)
history = model.fit(X_train, y_train,validation_data=(X_valid, y_valid),epochs=20, batch_size=128, verbose=1, 
                    callbacks=[saveBestCallback,earlyStopCallback,reduceLR]) #class_weight=class_weight

# ~~~~~~~~~~~~~~~ SAVE MODEL ~~~~~~~~~~~~~~~
