# Attia et. al. model with 6-lead ECG Data
# 6 leads are reduced to 2 independent leads

# ~~~~~~~~~~~~~~~ IMPORTS ~~~~~~~~~~~~~~~
from collections import Counter
import numpy as np
#import keras
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import concatenate,Activation,Dropout,Dense,ZeroPadding2D
from tensorflow.keras.layers import Input,add,Conv2D, MaxPooling2D,Flatten,BatchNormalization,LSTM
from tensorflow.keras.optimizers import Adam
import os
import zipfile
from pandas import *
from sklearn.model_selection import train_test_split
from CNN_6lead_model_keras import ResNet18
from pandas import *
import json
import os
from tensorflow.keras.utils import to_categorical
import argparse
import time
import random
# import numpy_indexed as npi
from numpy.random import default_rng

from IPython import display

from PIL import Image, ImageOps
import os, os.path
from tqdm import tqdm_notebook as tqdm
import shap

# ~~~~~~~~~~~~~~~ CONNECT TO GPU ~~~~~~~~~~~~~~~

os.environ["OMP_NUM_THREADS"] = "1"

if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    #Choose GPU 0 as a default if not specified (can set this in Python script that calls this)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

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

def combine_sets(afib_path, control_path, size=20000):
  np.random.seed(777) # Lucky number 7. 
  
  # Of these, select a random sample:
  afib_ecgs =random.choices(os.listdir(afib_path), k=int(size)) # has to be hardcoded in for some reason??!?!?!

  # Set up y values. These are all afib patients.
  y_labels_afib = np.ones(int(size)) # an array of all 1's

  # Now, let's get the healthy patients:
  control_ecgs = random.choices(os.listdir(control_path), k=int(size))
  y_labels_control = np.zeros(int(size))

  # Concatenate the 2 arrays:
  image_filenames = np.concatenate([afib_ecgs, control_ecgs])
  y_labels = np.concatenate([y_labels_afib, y_labels_control]) #.to(device)

  # Now randomize and make sure correct labels line up with ecgs:
  random_indices = np.random.randint(low=0,high=int(size)*2-1,size = (int(size)*2,))
  image_filenames = image_filenames[random_indices]
  y_labels = y_labels[random_indices]

  # Let's verify:
  print(image_filenames[4])
  print(y_labels[4])
  print(image_filenames[4] in afib_ecgs) # if y-label is 1, then this should be true. if y-label is 0, should be false.

  return image_filenames, y_labels


def fetch_data(datatype, size=20000, load=True):

  if datatype == 'spectrogram':
    if load:
      return preprocess(np.load("spectro_X.npy")[0:3000], np.load("spectro_Y.npy")[0:3000])
    afib_path =  '/local1/CSE_XAI/CSE482-XAI/image_data_processing/control_ecgs_as_spectro/'
    control_path = '/local1/CSE_XAI/CSE482-XAI/image_data_processing/afib_ecgs_as_spectro/'
  else: 
    afib_path =  '../../../../../../../local1/CSE_XAI/CSE482-XAI/image_data_processing/control_ecgs_as_plots/'
    control_path = '../../../../../../../local1/CSE_XAI/CSE482-XAI/image_data_processing/afib_ecgs_as_plots/'


  image_filenames, Y = combine_sets(afib_path, control_path, size)


  X = np.zeros((int(size*2), 300, 300))
  counter = 0
  for file in image_filenames:
    
    if counter %1000 == 0:
      print("Retrieved files: ", counter)

    try:
      image = Image.open(afib_path + file)
    except:
      image = Image.open(control_path + file)

    image = ImageOps.grayscale(image)

    if datatype == 'spectrogram':

      im = Image.open("/local1/CSE_XAI/CSE482-XAI/image_data_processing/afib_ecgs_as_spectro/"+file)
      im = ImageOps.grayscale(im)
      im = im.crop((140, 140, 1800, 1400))
      im = im.resize((300, 300), Image.ANTIALIAS)
      
      patient_X = np.asarray(im)

    
    X[counter] = patient_X
    counter += 1
  
  print("saving data into numpy arrays...")
  """
  if datatype == 'spectrogram':
    np.save("spectro_X.npy", X)
    np.save("spectro_Y.npy", Y)
  else:
    np.save("plot_X.npy", X)
    np.save("plot_Y.npy", Y)

    """
  return preprocess(X, Y)

# ~~~~~~~~~~~~~~~ TRAIN MODEL ~~~~~~~~~~~~~~~
def train_model(X_train, X_rem, y_train, y_rem, X_valid, X_test, y_valid, y_test, n_classes, model_name):
  
  model = ResNet18(2)

  opt0 = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.5)
  model.compile(loss='categorical_crossentropy',  optimizer=opt0,  metrics=['accuracy'])

  earlyStopCallback = EarlyStopping(monitor='val_loss', min_delta=0, patience=9,  mode='auto')
  saveBestCallback = ModelCheckpoint(model_name+'weights_only_checkpoint.h5',monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
  reduceLR =ReduceLROnPlateau(monitor = 'val_loss',factor = 0.5,patience = 3,verbose=1,min_lr = 0.00001)
  history = model.fit(X_train, y_train,validation_data=(X_valid, y_valid),epochs=5, batch_size=128, verbose=1, 
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

def load_model(filename, input_size):
  model = ResNet18(2)
  model.build(input_shape=input_size)
  model.load_weights(filename)
  return model

start_time = time.time()
#parser = argparse.ArgumentParser()
#parser.add_argument("-f", "--full", help="If you would like to use the full dataset (not recommended) use --full, otherwise, sample is used.", action="store_true")
#parser.add_argument("-s", "--spectrogram", help="Use this tag to use spectrogram images instead of plots.", action="store_true")
#parser.add_argument("-l", "--load_data",help="Use this load the data from the npy files.", action="store_true")
#args = parser.parse_args()
size = 5
datatype = "plot"
#if args.full:
#  size = 'full'
#if args.spectrogram:
datatype = "spectrogram"

model_name = 'CNN_6lead_'+str(size)+'_dataset_'
X_train, X_rem, y_train, y_rem, X_valid, X_test, y_valid, y_test, n_classes = fetch_data(datatype, size, False)

# model = train_model(X_train, X_rem, y_train, y_rem, X_valid, X_test, y_valid, y_test, n_classes, model_name)

#test_acc = get_test_acc(model, X_test, y_test)
#model_name += '_'+str(test_acc)[2:]

modelfile = modelfile = '/local1/CSE_XAI/CSE482-XAI/CNN/CNN_6lead_3000_dataset_weights_only_checkpoint.h5'
input_size = (1, 300, 300, 1)
model = load_model(modelfile, input_size)
#model = load_model(modelfile, input_size)

#save_model(model, model_name)
#end_time = time.time()
#print("Elapsed time: ", str(end_time-start_time))

res50_model = load_model('/local1/CSE_XAI/CSE482-XAI/CNN/CNN_6lead_3000_dataset_weights_only_checkpoint.h5', (2, 300, 300, 1))
res50_model.layers.pop(0)

newInput = Input(batch_shape=(1, 300,300,1))    # let us say this new InputLayer
newOutputs = res50_model(newInput)
newModel = Model(newInput, newOutputs)
newModel.set_weights(res50_model.get_weights())

newModel.summary()
res50_model.summary()

#Dataset should be a (sample x feature) shape
# X_test = X_test.reshape(-1, 90000)


#Model prediction, two cases depending on if its a single predicition or an array of predictions
def f(x):
    if len(x.shape) > 1:
        return newModel.predict(x.reshape(-1, 300, 300, 1))
    else:
        return newModel.predict(x.reshape(300, 300, 1))
    
# masker = shap.maskers.Image("inpaint_telea", (1, 300, 300, 1))

# By default the Partition explainer is used for all  partition explainer
# explainer = shap.Explainer(f, masker, output_names=class_names)

print(X_test.shape)

samples = shap.utils.sample(X_train[1:3], 2)
# samples = np.squeeze(samples)
print(samples.shape)
samples = np.reshape(samples, (2, 90000))
print(samples.shape)
bg = shap.maskers.Partition(samples)
print("GOt BG")
explainer = shap.explainers.Partition(f, bg)
print("GOT explainer")
shap_values = explainer(X_train[1])
print("GOT SHAP")

np.save(shap_values, "image_shaps.npy")

# shap_values = explainer(X_test[500], max_evals=500, batch_size=50, outputs=shap.Explanation.argsort.flip[:1])