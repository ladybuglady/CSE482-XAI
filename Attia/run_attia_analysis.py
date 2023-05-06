import argparse
import numpy as np
import os
import zipfile
import json
import joblib
from decomposer import decompose
from SHAP_script import Shap_Explainer

argParser = argparse.ArgumentParser()
argParser.add_argument("-ecg", "--ECG path", help="Filepath for ECG recording")

def get_patient_data(path=None):
    dir_path = '../../../../../../local1/CSE_XAI/small_data/'
    if path is None:
        path = os.listdir(dir_path)[0]
    
    patient_X = np.empty((2, 5000))
    jsonFile = open(dir_path + path, 'r')
    fileContents = json.load(jsonFile)

    # digging into the dictionaries to get lead data
    lead_1_samples = fileContents['samples']
    lead_2_samples = fileContents['extraLeads'][0]['samples']
    # Crop the data to 5000 data points (5 seconds).
    patient_X[0,:] = lead_1_samples[0:5000]
    patient_X[1,:] = lead_2_samples[0:5000]

    return patient_X

'''
Sadly, these don't work right now

def saveExplainerToFile(explainer, path="first_recording_shap_vals.bz2"):
    joblib.dump(explainer, filename=path, compress=('bz2', 9))

def loadExplainerFromFile(path):
    explainer = joblib.load(filename=path)
'''

def save_shap_vals(vals, path):
    # This is to load in later
    np.save(path, vals, allow_pickle=False, fix_imports=True)

    # This is for us to just be able to see what is in the array
    np.savetxt(path+".txt", vals)

    print("Saved!")


def main():
    patient_X = get_patient_data()

    shap = Shap_Explainer()
    shap.loadExplainer(entryCount=1)

    print("XAI for Attia - Prototype 1")
    print("Input: First ECG file")
    print(decompose(patient_X))
    print("Model: Attia")
    print("Explainer: SHAP")
    vals = shap.getShapValues(patient_X, reshape=True)
    print(len(vals))
    print(len(vals[0]))
    print(len(vals[1]))
    save_shap_vals(vals, "first_recording_shap_vals_entry1")
    print(vals)

if __name__ == "__main__":
    main()
