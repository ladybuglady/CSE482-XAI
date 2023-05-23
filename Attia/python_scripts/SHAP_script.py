# Read these to understand how the model is saved and how you can load it:
# https://www.tensorflow.org/guide/keras/save_and_serialize
# https://www.tensorflow.org/guide/saved_model#the_savedmodel_format_on_disk

# I have used the NEW KERAS SAVEDMODEL filesystem (updated January 2022).
# This is a directory that contains MODEL ARCHITECTURE as well as WEIGHTS.

# saved_model.pb IS THE MODEL ARCHITECTURE
# variables/variables.data-000000-of-00001 are the WEIGHTS.

# You will have to load the architecture then load it with the weights to initialize.


# the long string of numbers at the end of the directory (88......) is the test accuracy as
# a percent, missing the decimal point. Assume the decimal follows the first two digits (so 88.something %)
# This accuracy is obtained via only 3,000 recording files.

# The attia model reports 83% accuracy, so this should be sufficient to do a good SHAPLY analysis.

# good luck! -Zeynep



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#USAGE NOTES
#Run constructor sx = Shap_Explainer()
#Run loader of explainer by sx.load_explainer
#To get shap values of an ECG run sx.getShapValues(ECG INPUT)
#add parameter reshape = True if you have it in the (5000, 2, 1) format
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
from tensorflow import keras
import shap
import os
import numpy as np
import json
from pandas import *
import random


class Shap_Explainer:
    def __init__(self):
        self.explainer = None

    def loadExplainer(self, entryCount = 10):

        data_dir_path = '../../../../../../local1/CSE_XAI/control_small/' # changed this to control so that background data is all normal

        #Load model
        
        model = keras.models.load_model('./attia_6lead_sample_dataset__8899999856948853/')
        model.load_weights('./attia_6lead_sample_dataset__8899999856948853/variables/variables')

        #Load 10 data entries by default 
        data_entries = entryCount
        X = None

        # For background, it's best if the whole set is for "sinus rythm" (i.e. normal) recordings if our aim
        # is to see the important features of a afib recording ...

        paths = random.choices(os.listdir(data_dir_path), k=data_entries)

        for path in paths:
            patient_X = np.empty((2, 5000))
            jsonFile = open(data_dir_path + path, 'r')
            fileContents = json.load(jsonFile)
            lead_1_samples = fileContents['samples']
            lead_2_samples = fileContents['extraLeads'][0]['samples']

            patient_X[0, :] = lead_1_samples[0:5000]
            patient_X[1, :] = lead_2_samples[0:5000]

            if X is None:
                X = np.expand_dims(patient_X, axis=0)
            else:   
                X = np.concatenate((X, np.expand_dims(patient_X, axis=0)), axis=0)

        X = np.swapaxes(X,1,2)
        X = np.expand_dims(X, axis=3)

        print(X)
        print(model.summary())

        #Dataset should be a (sample x feature) shape
        X = X.reshape(-1, 10000)


        #Model prediction, two cases depending on if its a single predicition or an array of predictions
        def f(x):
            if len(x.shape) > 1:
                return model.predict(x.reshape(-1, 5000, 2, 1))
            else:
                return model.predict(x.reshape(5000, 2, 1))


        #first ten entries form "background" dataset which helps establish perturbations when finding shapley values
        self.explainer = shap.KernelExplainer(f, X)
        print("LOADED EXPLAINER")


    #X should be a (#samples, 10000) sized array (10000 features)
    #OR set reshape to True if you pass in a (#samples, 5000, 2, 1) array
    def getShapValues(self, X, reshape = False):

        if reshape:
            if len(X.shape) > 3:
                X = X.reshape(-1, 10000)
            else:
                X = X.reshape(10000)
            
        #Solves for all feature importance (one for every entry I think so like 5000..)
        shap_values = self.explainer.shap_values(X)
        return shap_values


#Make into module which outputs shapley values in right format tuple of ((5000), (5000)) nparray.
#Input: Feed me an ECG recording or an array of ECG recordings
#Ouput: The tuple for either single or array of ECG recordings' shapley values



#shap.summary_plot(explainer.expected_value, shap_values)

# Note*** There is also another file called "weights_only_checkpoint.h5". Ignore this. It is
# just to save intermediate progress.