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
from attia_6lead_model import BuildModel
from pandas import *
import json
import os
from keras.utils import to_categorical
import argparse
import time


# ~~~~~~~~~~~~~~~ CONNECT TO GPU ~~~~~~~~~~~~~~~

os.environ["OMP_NUM_THREADS"] = "0"

if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    #Choose GPU 0 as a default if not specified (can set this in Python script that calls this)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

# ~~~~~~~~~~~~~~~ DATA PREPROCESS ~~~~~~~~~~~~~~~
def preprocess(X, y_labels):
  # Change X dims for model compatibility
  # expected shape=(None, 5000, 2, 1)
  X = np.swapaxes(X,1,2)
  X = np.expand_dims(X, axis=3)

  # Change y dims to one-hot encoding, should be (300, 2)
  y = to_categorical(y_labels, dtype ="uint8")

  X_train, X_rem, y_train, y_rem = train_test_split(X,y, train_size=0.8)
  X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)

  N_Val = y_valid.shape[0]
  N_Train = y_train.shape[0]

  print ('Training on : ' + str(N_Train) + ' and validating on : ' +str(N_Val))

  n_classes = 2
  return X_train, X_rem, y_train, y_rem, X_valid, X_test, y_valid, y_test, n_classes

# ~~~~~~~~~~~~~~~ DATA FETCH ~~~~~~~~~~~~~~~

def fetch_data(size):
  if size == "full":
    dir_path =  '../../../../../../local1/CSE_XAI/study60_recordings_json/'
  else:
    dir_path = '../../../../../../local1/CSE_XAI/small_data/'
  # reading CSV file
  data = read_csv("Copy of study60_patient_recordings.csv")
  
  # converting column data to list
  filename_csv = data['recording_public_id'].tolist()
  y_csv = data['determination'].tolist()
  
  # printing list data
  print('Recording ID:', filename_csv[10])
  print('Determination:', y_csv[10])

  filenames = []

  X = None
  y_labels = []
  y = None
  count = 0
  # Iterate directory
  for path in os.listdir(dir_path):
    
    count += 1
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
  
  print("Count: ", count)
  y_labels = np.asarray(y_labels)
  return preprocess(X, y_labels)

# ~~~~~~~~~~~~~~~ TRAIN MODEL ~~~~~~~~~~~~~~~
def train_model(X_train, X_rem, y_train, y_rem, X_valid, X_test, y_valid, y_test, n_classes, model_name):
  class_weight={}

  model = BuildModel(segmentLength=int(5000),
                            padTo=int(5120),n_classes=n_classes,reluLast=True)


  earlyStopCallback = EarlyStopping(monitor='val_loss', min_delta=0, patience=9,  mode='auto')
  saveBestCallback = ModelCheckpoint(model_name+'weights_only_checkpoint.h5',monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
  reduceLR =ReduceLROnPlateau(monitor = 'val_loss',factor = 0.5,patience = 3,verbose=1,min_lr = 0.00001)
  history = model.fit(X_train, y_train,validation_data=(X_valid, y_valid),epochs=20, batch_size=128, verbose=1, 
                      callbacks=[saveBestCallback,earlyStopCallback,reduceLR]) #class_weight=class_weight

  
  return model

# ~~~~~~~~~~~~~~~ CALCULATE TEST ACCURACY ~~~~~~~~~~~~~~~
def get_test_acc(model, X_test, y_test):
  score = model.evaluate(X_test, y_test, verbose = 0) 

  print('Test loss:', score[0]) 
  print('Test accuracy:', score[1])
  print()
  print("For reference, Attia model reports 83.3% test accuracy.")
  return score[1]

# ~~~~~~~~~~~~~~~ SAVE MODEL ~~~~~~~~~~~~~~~
def save_model(model, name):
  model.save(name)
  print("Saved model to ", name)



def main():
  start_time = time.time()
  parser = argparse.ArgumentParser()
  parser.add_argument("-f", "--full", help="If you would like to use the full dataset (not recommended) use --full, otherwise, sample is used.", action="store_true")
  args = parser.parse_args()
  size = 'sample'
  if args.full:
    size = 'full'
  model_name = 'attia_6lead_'+size+'_dataset_'
  X_train, X_rem, y_train, y_rem, X_valid, X_test, y_valid, y_test, n_classes = fetch_data(size)

  model = train_model(X_train, X_rem, y_train, y_rem, X_valid, X_test, y_valid, y_test, n_classes, model_name)

  test_acc = get_test_acc(model, X_test, y_test)
  model_name += '_'+str(test_acc)[2:]

  save_model(model, model_name)
  end_time = time.time()
  print("Elapsed time: ", str(end_time-start_time))

    

if __name__ == "__main__":
    main()

