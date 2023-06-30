from eeg_funcs import *
from mne_bids import BIDSPath, write_raw_bids, print_dir_tree
import mne_bids
import logging
import numpy as np
import os
import subprocess
import sys
import json 

def turn_off_verbosity(bids_path):
    # Save the current standard output file descriptor
    original_stdout = os.dup(1)
    original_stderr = os.dup(2)
    # Open a null device for output
    null_device = open(os.devnull, 'w')
    err_file = open('file', 'w')
    # Redirect the standard output to the null device
    os.dup2(null_device.fileno(), 1)
    os.dup2(err_file.fileno(), 2)
    # Call the mne_bids.read_raw_bids function
    raw = mne_bids.read_raw_bids(bids_path)
    # Restore the original standard output
    os.dup2(original_stdout, 1)
    os.dup2(original_stderr, 2)
    # Close the null device
    null_device.close()
    err_file.close()
    return raw
    

clips = []

# arr1 = [[1,2], [3,4]]
# arr2 = [[5,6], [7,8]]

# clips.append(arr1)
# clips.append(arr2)
# clips = np.concatenate(clips, axis=1)
# print(clips)

# bids_path = BIDSPath(subject='0001', session='preimplant'+str(1).zfill(2), run=1, 
#     datatype='eeg', root='./bids_dataset', task='positive')

# raw = turn_off_verbosity(bids_path)
# clip = raw.get_data()
# ind = 0
# sfreq = raw.info['sfreq']
# duration = 5
# clips.append(clip[:][ind:int(ind + duration * sfreq)])

runs = [0, 8, 18, 10, 12, 11, 12, 14, 7, 13, 9, 0, 11, 5]
ch_names = []
sfreq = 0

for i in range(1, 14) :
    if (i == 11) : 
        continue
    for j in range(1, runs[i] + 1) :
        bids_path = BIDSPath(subject='0001', session='preimplant'+str(i).zfill(2), run=j, 
            datatype='eeg', root='./bids_dataset', task='positive')
        # raw = mne_bids.read_raw_bids(bids_path)
        raw = turn_off_verbosity(bids_path)
        clip = raw.get_data()
        ind = 0
        sfreq = raw.info['sfreq']
        ch_names = raw.info['ch_names']
        duration = 5
        while (ind + duration * sfreq <= clip.shape[1]) :
            clips.append(clip[:][ind:int(ind + duration * sfreq)])
            ind += 60 * sfreq

clips = np.concatenate(clips, axis=1)

# Variable for number of frequency bands 
numBands = 100

# Function to return array of averagedPsd arrays (one per time window). Returns 
# a numpy array of shape (numBands, numWindows), where each row is an array of 
# a particular frequency for all time windows
def averagePsds(signal, sfreq, timeWindow) :
    averagedPsds = []
    ind = 0
    while (ind + timeWindow * sfreq <= len(signal)) :
        # Calculate the psd for the current time window
        spectrum = psd(signal[int(ind) : int(ind + timeWindow * sfreq)], sfreq)[1]
        # Variable that is the discrete frequency associated with real frequency of 1 Hz,
        # based on the formula k = N * f_k / f_s
        N = timeWindow * sfreq
        k = int(N * 1 / sfreq)
        # Get the averaged psd of spectrum
        averagedPsd = [0 for i in range(numBands)]
        for i in range(len(averagedPsd)) :
            averagedPsd[i] = np.mean(spectrum[int(k * i) : int(k * (i + 1))])
        # Append averagedPsd to averagedPsds
        averagedPsds.append(averagedPsd)
        # Increment ind
        ind += timeWindow * sfreq
    
    return np.array(averagedPsds).T


timeWindows = [1, 5, 60, 1800]
freqs = [i for i in range(0, 100)]
statParams = ['min', 'max', 'mean', 'stdev']
# Dictionary with all the features. The keys are strings with format 
# '<ch_name> <timeWindow>s <freq>Hz <statParam>'
features = {}
# Array of features. Elements are the features in the form of dictionaries with 
# keys being the feature value, ch_name, timeWindow, freq, statParam
featuresArr = []

for i in range(len(ch_names)) :
    print(i)
    for j in range(len(timeWindows)) :
        print('---', j)
        averagedPsds = averagePsds(clips[i], sfreq, timeWindows[j])
        for k in range(len(freqs)) :
            for l in range(len(statParams)) :
                param = statParams[l]
                featureName = ch_names[i] + ' ' + str(timeWindows[j]) + 's ' + str(freqs[k]) + 'Hz ' + param
                if (param == 'min') :
                    feature = np.min(averagedPsds[k])
                elif (param == 'max') :
                    feature = np.max(averagedPsds[k])
                elif (param == 'mean') :
                    feature = np.mean(averagedPsds[k])
                elif (param == 'stdev') :
                    feature = np.std(averagedPsds[k])
                features[featureName] = feature 
                featureDict = {'value': feature, 'ch_name': ch_names[i], 'timeWindow (s)': timeWindows[j], 'freq (Hz)': freqs[k], 'statParam': param}
                featuresArr.append(featureDict)

print(len(features))
print(len(featuresArr))

with open('features.json', 'w') as f :
    json.dump(featuresArr, f)

# for i in range(1, 14) :
#     run = 1
#     # print(i)
#     while (True) :
#         # print('----', run)
#         bids_path = BIDSPath(subject='0001', session='preimplant'+str(i).zfill(2), run=run, 
#             datatype='eeg', root='./bids_dataset', task='positive')
#         try :
#             # raw = mne_bids.read_raw_bids(bids_path)
#             print(run)
#             raw = turn_off_verbosity(bids_path)
#             # print(run)
#             clip = raw.get_data()
#             ind = 0
#             sfreq = raw.info['sfreq']
#             duration = 5
#             while (ind + duration * sfreq <= clip.shape[1]) :
#                 clips.append(clip[:][ind:int(ind + duration * sfreq)])
#                 ind += 60 * sfreq
#             run += 1
#         except Exception as e:
#             print(e)
#             break


# print('\n\n\n')
# print(clips.shape)
