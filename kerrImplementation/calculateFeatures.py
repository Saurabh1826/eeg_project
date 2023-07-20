import sys
import os
from mne_bids import BIDSPath, write_raw_bids, print_dir_tree, get_entity_vals
import mne_bids
import logging
import numpy as np
import subprocess
import json 

sys.path.append(os.path.join('.'))
from eeg_funcs import *

def turn_off_verbosity(bids_path):
    # Save the current standard output file descriptor
    original_stdout = os.dup(1)
    original_stderr = os.dup(2)
    # Open a null device for output
    null_device = open(os.devnull, 'w')
    # err_file = open(os.devnull, 'w')
    # Redirect the standard output to the null device
    os.dup2(null_device.fileno(), 1)
    os.dup2(null_device.fileno(), 2)
    # Call the mne_bids.read_raw_bids function
    raw = mne_bids.read_raw_bids(bids_path)
    # Restore the original standard output
    os.dup2(original_stdout, 1)
    os.dup2(original_stderr, 2)
    # Close the null device
    null_device.close()
    # err_file.close()
    return raw

# Open config file
cwd = os.getcwd()
config = json.load(open(os.path.join(cwd, 'config.json'), 'r'))
    

ch_names = []
sfreq = 0


bids_root = os.path.join('.', config['bidsRoot'])

bids_path = BIDSPath(subject='sub01', task='rest', root=bids_root)
subjects = get_entity_vals(bids_root, 'subject')


# Dict of signals, one per subject. Keys are subject names, and values are the 
# signals for the subject. Each subject's signal is the concatenation
# of all of their runs in all of their sessions. The elements of signals are 
# numpy arrays of shape (numChannels, numSamples)
signals = {}
# Set of the electrodes to use. If a channel is not in this set, it is 
# discarded
channelsRef = {'fp1', 'fp2', 'f3', 'f4', 'f7', 'f8', 'fz', 'c3', 'c4', 'cz', 't3', 't4', 't5', 'p3', 'p4', 't6', 'o1', 'o2'}
# Array of the final channels that are being used. It will be a subset of 
# channelsRef, but might be smaller than channelsRef if the eeg does not have 
# all the channels that channelsRef does
channels = []

sfreq = 0
ch_names = []

# We are assuming that all tasks are named 'task'
for subject in subjects :
    signal = []
    path = os.path.join(bids_root, 'sub-'+subject)
    sessions = get_entity_vals(path, 'session')
    for session in sessions :
        path = os.path.join(path, 'ses-'+session, 'eeg')
        runs = get_entity_vals(path, 'run')
        for run in runs :
            bids_path.update(subject=subject, session=session, run=run, 
                datatype='eeg', root=bids_root, task='task')
            raw = turn_off_verbosity(bids_path)
            # Set sfreq and ch_names, to the appropriate value, if they are not
            # already set
            if (sfreq == 0) :
                sfreq = raw.info['sfreq']
            ch_names = raw.info['ch_names']
            clip = raw.get_data()
            # Keep only the channels in the set channelsRef
            ch_names = [ch_names[i].lower() for i in range(len(ch_names))]
            newClip = []
            channels = []
            for i in range(len(clip)) :
                if (ch_names[i] in channelsRef) :
                    newClip.append(clip[i])
                    channels.append(ch_names[i])
            
            # Run clip through 0.5 Hz high pass filter
            high_pass = create_high_pass_filter(4, 0.5, sfreq)
            newClip = apply_filter(newClip, high_pass)
            
            # Add clip to signal
            signal.append(newClip)
    # Concatenate the clips in signal into a single numpy array and append to 
    # signals
    signal = np.concatenate(signal, axis=1)
    signals[subject] = signal

# Assertion to check that all channels are present in the eeg file
assert len(channels) == len(channelsRef)
            
# print(len(signals))
# print(signals[0].shape)
# print(ch_names)
# print(channels)

# Variable for number of frequency bands 
numBands = 100

timeWindows = [5, 60]
freqs = [i for i in range(0, numBands)]
statParams = ['min', 'max', 'mean', 'stdev']

# Dict of features. Keys are subject names, and values are arrays of features. 
# The elements of the values are the features in the form of dictionaries with 
# keys being the feature value, ch_name, timeWindow, freq, statParam
featuresDict = {}

for subject in subjects :
    print(subject)
    print()
    clips = signals[subject]
    for i in range(len(channels)) :
        print(i)
        for j in range(len(timeWindows)) :
            # print('---', j)
            averagedPsds = averagePsds(clips[i], sfreq, timeWindows[j], numBands)
            for k in range(len(freqs)) :
                for l in range(len(statParams)) :
                    param = statParams[l]
                    if (param == 'min') :
                        feature = np.min(averagedPsds[k])
                    elif (param == 'max') :
                        feature = np.max(averagedPsds[k])
                    elif (param == 'mean') :
                        feature = np.mean(averagedPsds[k])
                    elif (param == 'stdev') :
                        feature = np.std(averagedPsds[k])
                    
                    featureDict = {'value': feature, 'ch_name': channels[i], 'timeWindow (s)': timeWindows[j], 'freq (Hz)': freqs[k], 'statParam': param}
                    if (subject in featuresDict) :
                        featuresDict[subject].append(featureDict)
                    else :
                        featuresDict[subject] = [featureDict]

# print(len(features))
# print(len(featuresArr))

with open('features.json', 'w') as f :
    json.dump(featuresDict, f)


