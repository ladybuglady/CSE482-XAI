
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
import json
import os
from keras.utils import to_categorical
from pandas import *

# ~~~~~~~~~~~~~~~ DATA FETCH ~~~~~~~~~~~~~~~
dir_path = '../../../../../../local1/CSE_XAI/study60_recordings_json/'

data = read_csv("/content/drive/MyDrive/Colab Notebooks/XAI_DATA/Copy of study60_patient_recordings.csv")
 
# converting column data to list
filename_csv = data['recording_public_id'].tolist()
y_csv = data['determination'].tolist()

# get 1 file

'''
After getting a file, put it into a numpy array
Be able use the bio spy python library to get wave components from that numpy array
Output a list of wave component values with timestamp

'''