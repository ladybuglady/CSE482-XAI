import os
import sys
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import zipfile
import json
import joblib
from python_scripts.decomposer import decompose
from python_scripts.SHAP_script import Shap_Explainer

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ecg', '--ecg', default=None, help='Filepath for ECG readings')
    parser.add_argument('-diag', '--diag', default='attia', help='Diagnostic model to be used')
    parser.add_argument('-xai', '--xai', default='shap', help='XAI model to be used')
    parser.add_argument('-saveTo', '--saveTo', default='plot.png', help='Desired file name of plot')
    return parser

parsedArgs = setup_parser().parse_args()

def get_patient_data(path=None):
    dir_path = '../../../../../../local1/CSE_XAI/small_data/'
    if path is None:
        path = dir_path + os.listdir(dir_path)[0]
    
    patient_X = np.empty((2, 5000))
    jsonFile = open(path, 'r')
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

def plotShap(filename):
    shap_vals = np.load(filename + "_shap_vals_entry.npy")
    shap_vals = shap_vals[0] # IM JUST GONNA ASSUME THAT BOTH THE LEADS ARE JUST REPEATED IN EACH ROW ):
    shap_vals = shap_vals.reshape((2, 5000))
    print(shap_vals.shape)
    shaps_df = pd.DataFrame(shap_vals, index=['Lead1', 'Lead2'])

    waves = decompose(get_patient_data())

    print("decomp complete")
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(12,4))
    sns.lineplot(get_patient_data()[0], ax=axes[0], linewidth = 0.5).set(ylim=[-2,2])
    sns.lineplot(get_patient_data()[1], ax=axes[1], color='orange', linewidth = 0.5).set(ylim=[-2,2])
    
    # Plot

    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(17,8))
    plt.tick_params(bottom='on')
    fig.suptitle('Sample SHAP Analysis of 2-lead ECG')

    cbar_ax = fig.add_axes([.94, 0.05, .05, .8])
    sns.heatmap(np.asarray(shaps_df.loc['Lead1']).reshape(1, 5000), cmap="PiYG", ax=axes[0], vmin=-0.001, vmax=0.001, 
                yticklabels=False, cbar_ax = cbar_ax, cbar_kws={'label': 'Importance According to SHAP'})
    sns.heatmap(np.asarray(shaps_df.loc['Lead2']).reshape(1, 5000), cmap="PiYG", ax=axes[1], vmin=-0.001, vmax=0.001, 
                yticklabels=False, cbar_ax = cbar_ax, cbar_kws={'label': 'Importance According to SHAP'})


    ax_l1 = axes[0].twinx()
    ax_l2 = axes[1].twinx()

    sns.lineplot(get_patient_data()[0], ax=ax_l1, color='orange', linewidth = 0.5).set(ylim=[-2,2])
    sns.lineplot(get_patient_data()[1], ax=ax_l2, color='orange', linewidth = 0.5).set(ylim=[-2,2])


    axes[0].grid(False)
    axes[1].grid(False)
    axes[0].set_xticks(axes[0].get_xticks()[::2])
    axes[1].set_xticks(axes[1].get_xticks()[::2]) 

    ax_l1.yaxis.set_ticks_position('left')
    ax_l2.yaxis.set_ticks_position('left')
    ax_l1.set(ylabel="Lead1")
    ax_l2.set(ylabel="Lead2")
    ax_l1.grid(False)
    ax_l2.grid(False)

    print("good")

    sns.scatterplot(x=waves['R1'], y=np.zeros(len(waves['R1'])), ax=ax_l1, color='red', label='R-waves')
    sns.scatterplot(x=waves['R2'], y=np.zeros(len(waves['R2'])), ax=ax_l2, color='red')

    sns.lineplot(x=waves['P'][0], y=np.zeros(len(waves['P'][0])), ax=ax_l1, c=sns.xkcd_rgb['dark periwinkle'], label='P-waves')
    for wave in waves['P']:
        sns.lineplot(x=wave, y=np.zeros(len(wave)), ax=ax_l1,  c=sns.xkcd_rgb['dark periwinkle'])
        sns.lineplot(x=wave, y=np.zeros(len(wave)), ax=ax_l2,  c=sns.xkcd_rgb['dark periwinkle'])

    sns.lineplot(x=waves['T'][0], y=np.zeros(len(waves['T'][0])), ax=ax_l1, c=sns.xkcd_rgb['bright cyan'], label='T-waves')
    for wave in waves['T']:
        sns.lineplot(x=wave, y=np.zeros(len(wave)), ax=ax_l1,  c=sns.xkcd_rgb['bright cyan'])
        sns.lineplot(x=wave, y=np.zeros(len(wave)), ax=ax_l2,  c=sns.xkcd_rgb['bright cyan'])

    plt.savefig(filename)


def main():
    print(parsedArgs.ecg)
    patient_X = get_patient_data(parsedArgs.ecg)

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
    save_shap_vals(vals, parsedArgs.saveTo + "_shap_vals_entry")
    print(vals)

    plotShap(parsedArgs.saveTo)

if __name__ == "__main__":
    main()
