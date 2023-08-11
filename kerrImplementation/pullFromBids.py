import sys
import os
from mne_bids import BIDSPath, write_raw_bids, print_dir_tree, get_entity_vals
import mne_bids
import logging
import numpy as np
import subprocess
import json 
import matplotlib.pyplot as plt 
import matplotlib
import seaborn as sns
import pandas as pd

sys.path.append(os.path.join('.'))
from eeg_funcs import *

# Function to use functions to pull from bids dataset without the verbosity
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

# Set up bids settings 
bids_root = os.path.join('.', config['bidsRoot'])
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
# Loop through all subjects, sessions, and runs to get the data for each subject
# and store it in signals
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