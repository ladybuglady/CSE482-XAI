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
import random
import numpy_indexed as npi
from numpy.random import default_rng


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
  print("Y before categorical:")
  print(y_labels[0:5])
  y = to_categorical(y_labels, dtype ="uint8")
  print("Y after categorical:")
  print(y[0:5])

  X_train, X_rem, y_train, y_rem = train_test_split(X,y, train_size=0.8)
  X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)

  N_Val = y_valid.shape[0]
  N_Train = y_train.shape[0]

  print ('Training on : ' + str(N_Train) + ' and validating on : ' +str(N_Val))

  n_classes = 2
  return X_train, X_rem, y_train, y_rem, X_valid, X_test, y_valid, y_test, n_classes

# ~~~~~~~~~~~~~~~ DATA FETCH ~~~~~~~~~~~~~~~

def combine_sets(refs, afib_path, control_path, size=30000):
  np.random.seed(777) # Lucky number 7. 
  
  # Filter out only episodes of sinus rhythm in the afib patients:
  sr_episodes = refs[refs["determination"]=="Sinus Rhythm"]["recording_public_id"]

  # Of these, select a random sample:
  afib_recordings = random.sample(list(sr_episodes), 30000) # has to be hardcoded in for some reason??!?!?!

  # Get complete filename:
  afib_recordings = np.array(list(map(lambda s: s+"_raw.json", afib_recordings)))

  print(afib_recordings[0:10])
  # Now, use these IDs to reference the afib ECG dataset
  afib_ecgs =  np.asarray(os.listdir(afib_path))
  afib_ecgs = afib_ecgs[npi.indices(afib_ecgs, afib_recordings)]

  # Verify: (These should print the same thing)
  print("Recording ID:", afib_recordings[120])
  print("Recording ID:", afib_ecgs[120])

  # Set up y values. These are all afib patients.
  y_labels_afib = np.ones(30000) # an array of all 1's

  # Now, let's get the healthy patients:
  control_ecgs = random.choices(os.listdir(control_path), k=30000)
  y_labels_control = np.zeros(30000)

  # Concatenate the 2 arrays:
  ecg_filenames = np.concatenate([afib_ecgs, control_ecgs])
  y_labels = np.concatenate([y_labels_afib, y_labels_control])

  # Now randomize and make sure correct labels line up with ecgs:
  random_indices = np.random.randint(low=0,high=30000*2-1,size = (30000*2,))
  ecg_filenames = ecg_filenames[random_indices]
  y_labels = y_labels[random_indices]

  # Let's verify:
  print(ecg_filenames[600])
  print(y_labels[600])
  print(ecg_filenames[600] in afib_ecgs) # if y-label is 1, then this should be true. if y-label is 0, should be false.

  return ecg_filenames, y_labels


def fetch_data(size=30000):

  # ECG Recordings
  afib_path =  '../../../../../../../local1/CSE_XAI/study60_recordings_json/'
  control_path = '../../../../../../../local1/CSE_XAI/control_small/'

  # CSV file
  refs = read_csv("../Copy of study60_patient_recordings.csv")

  ecg_filenames, Y = combine_sets(refs, afib_path, control_path, size)


  X = np.zeros((60000, 2, 5000))
  counter = 0
  for file in ecg_filenames:
    
    if counter %1000 == 0:
      print("Retrieved files: ", counter)
    patient_X = np.empty((2, 5000))

    try:
      jsonFile = open(afib_path + file, 'r')
    except:
      jsonFile = open(control_path + file, 'r')

    fileContents = json.load(jsonFile)

    # digging into the dictionaries to get lead data
    lead_1_samples = fileContents['samples']
    lead_2_samples = fileContents['extraLeads'][0]['samples']

    # Crop the data to 5000 data points.
    patient_X[0,:] = lead_1_samples[0:5000]
    patient_X[1,:] = lead_2_samples[0:5000]
    
    X[counter] = patient_X
    counter += 1
    jsonFile.close()

  return preprocess(X, Y)

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
  size = 30000
  if args.full:
    size = 'full'
  model_name = 'attia_6lead_'+str(size)+'_dataset_'
  X_train, X_rem, y_train, y_rem, X_valid, X_test, y_valid, y_test, n_classes = fetch_data(size)

  model = train_model(X_train, X_rem, y_train, y_rem, X_valid, X_test, y_valid, y_test, n_classes, model_name)

  test_acc = get_test_acc(model, X_test, y_test)
  model_name += '_'+str(test_acc)[2:]

  save_model(model, model_name)
  end_time = time.time()
  print("Elapsed time: ", str(end_time-start_time))

    

if __name__ == "__main__":
    main()
