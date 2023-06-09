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
        self.model = None

    def buildExplainer(self, modeltype, entryCount = 10):

        control_data_dir_path = '../../../../../../local1/CSE_XAI/control_small/'
        afib_data_dir_path = '../../../../../../local1/CSE_XAI/small_data/'

        #Load model
        if modeltype == 'Attia':
            modelfile = './Attia/attia_6lead_sample_dataset__9726666808128357/'
            weightsfile = './Attia/attia_6lead_sample_dataset__9726666808128357/variables/variables'
        elif modeltype == 'LSTM':
            modelfile = './LSTM/lstm_6lead_30000_dataset__5203333497047424/'
            weightsfile = './LSTM/lstm_6lead_30000e_dataset__5203333497047424/variables/variables'
        else: 
            print("This model does not exist.")
            return


        model = keras.models.load_model(modelfile)
        model.load_weights(weightsfile)
        print("Building explainer with model file found at: ", modelfile)

        #Load 10 data entries by default 
        data_entries = entryCount
        X = np.zeros((data_entries, 2, 5000))
        print(model.summary())

        self.model = model

        # For background, it's best if the whole set is for "sinus rythm" (i.e. normal) recordings if our aim
        # is to see the important features of a afib recording ...

        control_paths = random.choices(os.listdir(control_data_dir_path), k=int(data_entries/2))
        afib_paths = random.choices(os.listdir(afib_data_dir_path), k=int(data_entries/2))
        paths = control_paths + afib_paths

        counter = 0
        for path in paths:
            try:
                jsonFile = open(control_data_dir_path + path, 'r')
            except:
                jsonFile = open(afib_data_dir_path + path, 'r')
            fileContents = json.load(jsonFile)
            curr_X = np.empty((2, 5000))
            lead_1_samples = fileContents['samples']
            lead_2_samples = fileContents['extraLeads'][0]['samples']

            curr_X[0, :] = lead_1_samples[0:5000]
            curr_X[1, :] = lead_2_samples[0:5000]

            X[counter] = curr_X
            counter += 1
            jsonFile.close()

        X = np.swapaxes(X,1,2)
        X = np.expand_dims(X, axis=3)

        

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

    #X should be a (#samples, 10000) sized array (10000 features)
    #OR set reshape to True if you pass in a (#samples, 5000, 2, 1) array
    def getShapValues(self, X, reshape = False):

        print("Original size of X: ", X.shape)

        if reshape:
            if len(X.shape) > 3:
                X = X.reshape(-1, 10000)
            else:
                X = X.reshape((1, 10000))

        print("Reshaped X: ", X.shape)
            
        #Solves for all feature importance (one for every entry I think so like 5000..)
        shap_values = self.explainer.shap_values(X)
        expected_value = self.explainer.expected_value
        return shap_values, expected_value

    def getActual(self, x):
        if len(x.shape) > 1:
            return self.model.predict(x.reshape(-1, 5000, 2, 1))
        else:
            return self.model.predict(x.reshape(5000, 2, 1))