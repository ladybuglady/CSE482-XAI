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
dir_path = '../../../../../../../local1/CSE_XAI/small_data/'

# get 1 file
count = 0
# Iterate directory
for path in os.listdir(dir_path):
    patient_X = np.empty((2, 5000))

    jsonFile = open(dir_path + path, 'r')
    fileContents = json.load(jsonFile)

    # digging into the dictionaries to get lead data
    lead_1_samples = fileContents['samples']
    lead_2_samples = fileContents['extraLeads'][0]['samples']

    # Crop the data to 5000 data points (5 seconds).
    patient_X[0,:] = lead_1_samples[0:5000]
    patient_X[1,:] = lead_2_samples[0:5000]
    count += 1
    if count == 1: # just get 1.
        break

'''
After getting a file, put it into a numpy array
Be able use the bio spy python library to get wave components from that numpy array
Output a list of wave component values with timestamp

'''

def decompose(patient_X):

    features = {"R1": [], "R2":[], "P":[], "PR":[], "QRS":[], "ST":[], "T":[], "TP":[]}
    frequency = fileContents["frequency"]

    # smooth and straighten leads for inversion detection
    filtered_lead_1 = ecg.ecg(patient_X[0], sampling_rate=frequency, show=False)[1]
    filtered_lead_2 = ecg.ecg(patient_X[1], sampling_rate=frequency, show=False)[1]

    # check if leads are inverted
    inverted_lead_1 = nk.ecg_invert(filtered_lead_1, sampling_rate=frequency, show=False)[1]
    inverted_lead_2 = nk.ecg_invert(filtered_lead_2, sampling_rate=frequency, show=False)[1]

    if inverted_lead_1:
        # extract r-wave peak timestamps from inverted lead I (the inversion of the originally inverted lead)
        r_peaks_lead_1 = nk.ecg_peaks(np.negative(patient_X[0]), sampling_rate=frequency, method="neurokit", correct_artifacts=True)[1]["ECG_R_Peaks"]
        # correct r-wave peak timestamps to fit inverted lead I (the inversion of the originally inverted lead)
        r_peaks_lead_1 = ecg.correct_rpeaks(np.negative(patient_X[0]), r_peaks_lead_1, sampling_rate=frequency)[0]
    else:
        # extract r-wave peak timestamps from lead I
        r_peaks_lead_1 = nk.ecg_peaks(patient_X[0], sampling_rate=frequency, method="neurokit", correct_artifacts=True)[1]["ECG_R_Peaks"]
        # correct r-wave peak timestamps
        r_peaks_lead_1 = ecg.correct_rpeaks(patient_X[0], r_peaks_lead_1, sampling_rate=frequency)[0]

    if inverted_lead_2:
        # correct r-wave peak timestamps to fit inverted lead II (the inversion of the originally inverted lead)
        r_peaks_lead_2 = ecg.correct_rpeaks(np.negative(patient_X[1]), r_peaks_lead_1, sampling_rate=frequency)[0]
    else:
        # correct r-wave peak timestamps to fit lead II
        r_peaks_lead_2 = ecg.correct_rpeaks(patient_X[1], r_peaks_lead_1, sampling_rate=frequency)[0]

    # filter out baseline wander in lead II
    filtered_lead_2 = nk.signal_filter(patient_X[1], sampling_rate=frequency, lowcut=3, highcut=None, method="butterworth", order=2, show=False)

    # filter out powerline interference in lead II
    filtered_lead_2 = nk.signal_filter(filtered_lead_2, sampling_rate=frequency, lowcut=None, highcut=None, method="powerline", powerline=50, show=False)

    # filter out EMG noise in lead II

    # filter out electrode motion artifacts in lead II

    # extract p_positions-wave peak timestamps from filtered lead II
    if inverted_lead_2:
        peaks_lead_2 = nk.ecg_delineate(np.negative(filtered_lead_2), r_peaks_lead_2, sampling_rate=frequency, method="dwt", show=False, check=False)[0]
    else:
        peaks_lead_2 = nk.ecg_delineate(filtered_lead_2, r_peaks_lead_2, sampling_rate=frequency, method="dwt", show=False, check=False)[0]

    p_starts_lead_2 = []
    p_ends_lead_2 = []

    for i, v in enumerate(peaks_lead_2["ECG_P_Onsets"]):
        if v == 1:
            p_starts_lead_2.append(i)

    for i, v in enumerate(peaks_lead_2["ECG_P_Offsets"]):
        if v == 1:
            p_ends_lead_2.append(i)

    p_positions = list(zip(p_starts_lead_2, p_ends_lead_2))

    q_starts_lead_2 = []
    s_ends_lead_2 = []

    for i, v in enumerate(peaks_lead_2["ECG_R_Onsets"]):
        if v == 1:
            q_starts_lead_2.append(i)

    for i, v in enumerate(peaks_lead_2["ECG_R_Offsets"]):
        if v == 1:
            s_ends_lead_2.append(i)

    if q_starts_lead_2[0] < p_ends_lead_2[0]:
        pr_positions = list(zip(p_ends_lead_2, q_starts_lead_2[1:]))
    else:
        pr_positions = list(zip(p_ends_lead_2, q_starts_lead_2))

    if q_starts_lead_2[0] > s_ends_lead_2[0]:
        qrs_positions = list(zip(q_starts_lead_2, s_ends_lead_2[1:]))
    else:
        qrs_positions = list(zip(q_starts_lead_2, s_ends_lead_2))

    t_starts_lead_2 = []
    t_ends_lead_2 = []

    for i, v in enumerate(peaks_lead_2["ECG_T_Onsets"]):
        if v == 1:
            t_starts_lead_2.append(i)

    for i, v in enumerate(peaks_lead_2["ECG_T_Offsets"]):
        if v == 1:
            t_ends_lead_2.append(i)

    if t_starts_lead_2[0] < s_ends_lead_2[0]:
        st_positions = list(zip(s_ends_lead_2, t_starts_lead_2[1:]))
    else:
        st_positions = list(zip(s_ends_lead_2, t_starts_lead_2))

    t_positions = list(zip(t_starts_lead_2, t_ends_lead_2))

    '''
    for i in t_positions:
        plt.axvspan(i[0], i[1], facecolor="r", alpha=0.25)
    '''

    if t_starts_lead_2[0] > p_ends_lead_2[0]:
        tp_positions = list(zip(t_ends_lead_2, p_starts_lead_2[1:]))
    else:
        tp_positions = list(zip(t_ends_lead_2, p_starts_lead_2))
    
    features["R1"] = r_peaks_lead_1
    features["R2"] = r_peaks_lead_2
    features["P"] = p_positions
    features["PR"] = pr_positions
    features["QRS"] = qrs_positions
    features["ST"] = st_positions
    features["T"] = t_positions
    features["TP"] = tp_positions

    return features