# ~~~~~~~~~~~~~~~ IMPORTS ~~~~~~~~~~~~~~~
from collections import Counter
import numpy as np
import os
import zipfile
import json
from pandas import *
import matplotlib.pyplot as plt
from biosppy.signals import ecg
import neurokit2 as nk
from scipy import signal

# ~~~~~~~~~~~~~~~ DATA FETCH ~~~~~~~~~~~~~~~
dir_path = '../../../../../../local1/CSE_XAI/small_data/'

# get 1 file
count = 0
# Iterate directory
for path in os.listdir(dir_path):
    patient_X = np.empty((2, 6000))

    jsonFile = open(dir_path + path, 'r')
    fileContents = json.load(jsonFile)

    # digging into the dictionaries to get lead data
    lead_1_samples = fileContents['samples']
    lead_2_samples = fileContents['extraLeads'][0]['samples']

    # Crop the data to 6000 data points (6 seconds).
    patient_X[0,:] = lead_1_samples[0:6000]
    patient_X[1,:] = lead_2_samples[0:6000]
    count += 1
    if count == 1: # just get 1.
        break

'''
After getting a file, put it into a numpy array
Be able use the bio spy python library to get wave components from that numpy array
Output a list of wave component values with timestamp

'''

def decompose():
    features = {"VR": None, "R": [], "P":[], "T":[]}
    frequency = fileContents["frequency"]
    duration = fileContents["duration"]

    # smooth and straighten lead I for R-peak detection
    filtered_lead_1 = ecg.ecg(patient_X[0], sampling_rate=frequency, show=False)[1]

    # check if lead 1 is inverted
    inverted = nk.ecg_invert(patient_X[0], sampling_rate=frequency, show=False)[1]

    # extract r-wave peak timestamps from filtered lead I using Hamilton's algorithm
    r_peaks_lead_1 = ecg.hamilton_segmenter(filtered_lead_1, sampling_rate=frequency)[0]

    if inverted:
        # correct r-wave peak timestamps to fit inverted lead I (the inversion of the originally inverted lead)
        r_peaks_lead_1 = ecg.correct_rpeaks(np.negative(patient_X[0]), r_peaks_lead_1, sampling_rate=frequency)[0]
    else:
        # correct r-wave peak timestamps to fit original lead I
        r_peaks_lead_1 = ecg.correct_rpeaks(patient_X[0], r_peaks_lead_1, sampling_rate=frequency)[0]

    features["VR"] = len(r_peaks_lead_1) * (60 / (len(patient_X[0]) / 1000))
    features["R"] = r_peaks_lead_1

    # filter out baseline wander in lead II
    filtered_lead_2 = nk.signal_filter(patient_X[1], sampling_rate=frequency, lowcut=3, highcut=None, method="butterworth", order=2, show=False)

    # filter out powerline interference in lead II
    filtered_lead_2 = nk.signal_filter(filtered_lead_2, sampling_rate=frequency, lowcut=None, highcut=None, method="powerline", powerline=50, show=False)

    # filter out EMG noise in lead II

    # filter out electrode motion artifacts in lead II

    # extract p-wave peak timestamps from filtered lead II
    peaks_lead_2 = nk.ecg_delineate(filtered_lead_2, r_peaks_lead_1, sampling_rate=frequency, method="cwt", show=False, check=False)[0]

    p_peaks_lead_2 = peaks_lead_2["ECG_P_Peaks"]
    t_peaks_lead_2 = peaks_lead_2["ECG_T_Peaks"]

    # estimate start and end timestamps based on p-wave peak and add to features map
    for i, p in enumerate(p_peaks_lead_2):
        if p == 0:
            continue
        if i - 60 < 0:
            features["P"].append([0, i])
        elif i + 60 > duration:
            features["P"].append([i, duration])
        else:
            features["P"].append([i - 30, i + 30])

    # estimate start and end timestamps based on t-wave peak and add to features map
    for i, t in enumerate(t_peaks_lead_2):
        if t == 0:
            continue
        if i - 60 < 0:
            features["T"].append([0, i])
        elif i + 60 > duration:
            features["T"].append([i, duration])
        else:
            features["T"].append([i - 30, i + 30])

    return features