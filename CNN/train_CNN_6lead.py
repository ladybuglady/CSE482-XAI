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
from CNN_6lead_model import ResNet18
from pandas import *
import json
import os
from keras.utils import to_categorical
import argparse
import time
import random
import numpy_indexed as npi
from numpy.random import default_rng

from PIL import Image, ImageOps
import os, os.path
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models
from tqdm import tqdm_notebook as tqdm


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

def combine_sets(afib_path, control_path, size=30000):
  np.random.seed(777) # Lucky number 7. 
  
  # Of these, select a random sample:
  afib_ecgs =random.choices(os.listdir(afib_path), k=int(size)) # has to be hardcoded in for some reason??!?!?!

  # Set up y values. These are all afib patients.
  y_labels_afib = np.ones(300) # an array of all 1's

  # Now, let's get the healthy patients:
  control_ecgs = random.choices(os.listdir(control_path), k=300)
  y_labels_control = np.zeros(300)

  # Concatenate the 2 arrays:
  image_filenames = np.concatenate([afib_ecgs, control_ecgs])
  y_labels = np.concatenate([y_labels_afib, y_labels_control])

  # Now randomize and make sure correct labels line up with ecgs:
  random_indices = np.random.randint(low=0,high=300*2-1,size = (300*2,))
  image_filenames = image_filenames[random_indices]
  y_labels = y_labels[random_indices]

  # Let's verify:
  print(image_filenames[400])
  print(y_labels[400])
  print(image_filenames[400] in afib_ecgs) # if y-label is 1, then this should be true. if y-label is 0, should be false.

  return image_filenames, y_labels


def fetch_data(datatype, size=30000):

  if datatype == 'spectrogram':
    afib_path =  '../../../../../../../local1/CSE_XAI/CSE482-XAI/image_data_processing/control_ecgs_as_spectro/'
    control_path = '../../../../../../../local1/CSE_XAI/CSE482-XAI/image_data_processing/afib_ecgs_as_spectro/'
  else: 
    afib_path =  '../../../../../../../local1/CSE_XAI/CSE482-XAI/image_data_processing/control_ecgs_as_plots/'
    control_path = '../../../../../../../local1/CSE_XAI/CSE482-XAI/image_data_processing/afib_ecgs_as_plots/'


  image_filenames, Y = combine_sets(afib_path, control_path, size)


  X = np.zeros((int(size*2), 2, 5000))
  counter = 0
  for file in image_filenames:
    
    if counter %1000 == 0:
      print("Retrieved files: ", counter)
    patient_X = np.empty((2, 5000))

    try:
      image = Image.open(afib_path + file)
    except:
      image = Image.open(control_path + file)

    image = ImageOps.grayscale(image)
    data_ary = np.asarray(image)
    print(data_ary.shape)

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
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = ResNet18().to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=0.1, decay=0.5, )
  #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=0.05)
  criterion = nn.CrossEntropyLoss()
  batch_size=32

  model.summary()
  model.compile(loss=criterion,  optimizer=optimizer,  metrics=['accuracy'])

  earlyStopCallback = EarlyStopping(monitor='val_loss', min_delta=0, patience=9,  mode='auto')
  saveBestCallback = ModelCheckpoint(model_name+'weights_only_checkpoint.h5',monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
  reduceLR =ReduceLROnPlateau(monitor = 'val_loss',factor = 0.5,patience = 3,verbose=1,min_lr = 0.00001)
  history = model.fit(X_train, y_train,validation_data=(X_valid, y_valid),epochs=20, batch_size=batch_size, verbose=1, 
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
  parser.add_argument("-s", "--spectrogram", help="Use this tag to use spectrogram images instead of plots.", action="store_true")
  args = parser.parse_args()
  size = 300 #30000
  datatype = "plot"
  if args.full:
    size = 'full'
  if args.spectrogram:
    datatype = "spectrogram"

  model_name = 'CNN_6lead_'+str(size)+'_dataset_'
  X_train, X_rem, y_train, y_rem, X_valid, X_test, y_valid, y_test, n_classes = fetch_data(datatype, size)

  model = train_model(X_train, X_rem, y_train, y_rem, X_valid, X_test, y_valid, y_test, n_classes, model_name)

  test_acc = get_test_acc(model, X_test, y_test)
  model_name += '_'+str(test_acc)[2:]

  save_model(model, model_name)
  end_time = time.time()
  print("Elapsed time: ", str(end_time-start_time))

    

if __name__ == "__main__":
    main()
