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
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import os
from keras.utils import to_categorical
import argparse
import time
import random
import numpy_indexed as npi
from numpy.random import default_rng
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import signal 
import matplotlib as mpl



# ~~~~~~~~~~~~~~~ CONNECT TO GPU ~~~~~~~~~~~~~~~

os.environ["OMP_NUM_THREADS"] = "0"

if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    #Choose GPU 0 as a default if not specified (can set this in Python script that calls this)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

# ~~~~~~~~~~~~~~~ GLOBAL VARIABLES ~~~~~~~~~~~~~~~

size = 1 # should do 300,000
afib_path =  '../../../../../../../local1/CSE_XAI/study60_recordings_json/'
control_path = '../../../../../../../local1/CSE_XAI/control_small/'

# ~~~~~~~~~~~~~ ACQUIRE ORIGINAL DATA ~~~~~~~~~~~~~
def get_ecgs(ecg_filenames):
    X = np.zeros((size, 2, 5000))
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
    return X

# ~~~~~~~~~~~~~ PLOTTING FUNCTIONS ~~~~~~~~~~~~~
def make_plot_images(ecgs, dest):
    i = 0
    for ecg in ecgs:
        fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(50,4))
        sns.lineplot(ecg[0], ax=axes[0], linewidth = 0.5).set(ylim=[-2,2])
        sns.lineplot(ecg[1], ax=axes[1], color='orange', linewidth = 0.5).set(ylim=[-2,2])
        plt.tight_layout()
        plt.savefig(dest+"_"+str(i), dpi=300) #DEST SHOULD BE  DIRECTTORY INSIDE OF LOCAL1
        plt.close()
        i+=1
        print(i)

def make_spectro_images(ecgs, dest):
    i=0
    for ecg in ecgs:
        print(ecg.shape)
        # flatten the ecg
        ecg = ecg.flatten()
        print(ecg.shape)
        
        fig, ax = plt.subplots() 
        ax.set_ylim(0,0.5)
        f, t, Sxx = signal.spectrogram(ecg)
        pc = ax.pcolormesh(t, f, Sxx, norm=mpl.colors.LogNorm(vmin=10e-10, vmax=100), cmap='gray')
        plt.savefig(dest+"_"+str(i), dpi=300)
        plt.close()
        i+=1
        print(i)

def main():
    refs = pd.read_csv("../Copy of study60_patient_recordings.csv")
    sr_episodes = refs[refs["determination"]=="Sinus Rhythm"]["recording_public_id"]

    afib_recordings = list(sr_episodes)[:size]
    afib_recordings = np.array(list(map(lambda s: s+"_raw.json", afib_recordings)))
    np.savetxt('afib_recordings.out', afib_recordings, delimiter=',', fmt='%s')

    afib_files =  np.asarray(os.listdir(afib_path))
    afib_files = afib_files[npi.indices(afib_files, afib_recordings)]
    afib_ecgs = get_ecgs(afib_files)

    control_files = np.asarray(os.listdir(control_path))[:size]
    control_ecgs = get_ecgs(control_files)


    """make_plot_images(afib_ecgs, "/../../../../../local1/afib_ecgs_as_plots/")
    make_plot_images(control_ecgs, "/../../../../../local1/control_ecgs_as_plots/")"""
    #make_plot_images(afib_ecgs, "afib_ecgs_as_plots/")
    #make_plot_images(control_ecgs, "control_ecgs_as_plots/")
    make_spectro_images(afib_ecgs)
    make_spectro_images(control_ecgs)


if __name__ == "__main__":
    main()