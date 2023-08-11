# Script to download data from ieeg.org and save only the cutoff amount of data 
# (ie: a certain number of samples from the averaged psd of the clip) for each 
# patient

from ieeg.auth import Session
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
import os
import pathlib
from scipy.fft import fft, fftfreq
import math
import sys
from scipy.integrate import simps
from joblib import Parallel, delayed
import pandas as pd
from scipy.stats import zscore
import mne
import json

sys.path.append('.')

from eeg_funcs import *

from mne_bids import BIDSPath, write_raw_bids, print_dir_tree, update_sidecar_json
import mne_bids
from pyedflib import highlevel

# Variable for number of 1Hz frequency bands we take from the psd
numBands = 100
# Size of the sample we take from the full averaged psd
sampleSize = 100
# Time windows to be used
# timeWindows = [1, 5, 60]
timeWindows = [5, 60]

# Read in admissionToSubject.json
admissionToSubject = json.load(open('admissionToSubject.json', 'r'))
# Read in ieegFilenames.json, an array of all ieeg filenames
ieegFilenames = json.load(open('ieegFilenames.json', 'r'))

# Set of the electrodes to use. If a channel is not in this set, it is 
# discarded
channelsRef = {'fp1', 'fp2', 'f3', 'f4', 'f7', 'f8', 'fz', 'c3', 'c4', 'cz', 't3', 't4', 't5', 'p3', 'p4', 't6', 'o1', 'o2'}

# Create a mapping from subject to admission(s) from admissionToSubject
subjectToAdmissions = {}
for key, value in admissionToSubject.items():
    if value in subjectToAdmissions:
        subjectToAdmissions[value].append(key)
    else:
        subjectToAdmissions[value] = [key]

# Create a mapping from admission to ieeg filenames
admissionToFilenames = {}
for admission, _ in admissionToSubject.items() :
    for filename in ieegFilenames :
        if (admission in filename) :
            if (admission in admissionToFilenames) : 
                admissionToFilenames[admission].append(filename)
            else :
                admissionToFilenames[admission] = [filename]

# Initialize variables necessary to download file from ieeg.org
cwd = os.getcwd()
config = json.load(open(os.path.join(cwd, 'config.json'), 'r'))
f = open(config['pwd_fpath'], 'rb')
pwd = f.read().decode()
session = Session(config['username'], pwd)
fs = 500

###############
# subjectSet = {'0352', '0332', '0379', '0246', '0265', '0387', '0366', '0390', '0388', '0344', '0283', '0078', '0075', '0245', '0280', '0358', '0370', '0318', '0354', '0331', '0321', '0386', '0350', '0278', '0285', '0345', '0039', '0303', '0308', '0262', '0275', '0252', '0047', '0385', '0384', '0347', '0336', '0295', '0229', '0205', '0239', '0228', '0261', '0328', '0302', '0179', '0045', '0235', '0312', '0190', '0230', '0299', '0238', '0240', '0277', '0310', '0355', '0264', '0383', '0368', '0250', '0269', '0293', '0360', '0076', '0367', '0357', '0129', '0373', '0392', '0314', '0291', '0169', '0243', '0048', '0351'}
###############

# Loop through each subject, download their data from ieeg.org, and save to BIDS
for subject, admissions in subjectToAdmissions.items() :
    ###############
    # if (not subject == '0045') : continue
    # if (not subject in subjectSet) : continue
    ###############
    numSamples = 0
    print()
    print(subject)
    # Array of arrays that are the individual files for this subject
    fileClips = []
    # Loop through all admissions for this subject
    for admission in admissions :
        # Check if we have 100min of data
        if (numSamples / fs >= 6000) : break
        print(admission)
        # Loop through all ieeg files in this admission
        for file in admissionToFilenames[admission] :
            # Check if we have 100min of data
            if (numSamples / fs >= 6000) : break
            print(file)
            try :
                # List of all the runs of this file
                dataClips, channels, sfreq = getRuns(session, file, channelsRef, printProgress=True)
                if (dataClips == None) : 
                    print('Error on file', file) 
                    continue
            except : 
                continue
             
            # Concatenate the arrays in dataClips
            if (len(dataClips) == 0) : continue
            dataClip = np.concatenate(dataClips, axis=1)
            # Add dataClip to fileClips
            fileClips.append(dataClip)
            # Increment numSamples
            numSamples += len(dataClip[0])
    
    # Concatenate all arrays in fileClips 
    if (len(fileClips) == 0) : continue
    clip = np.concatenate(fileClips, axis=1)
    
    # Check if all channels are present
    if (channels == None or len(channels) < len(channelsRef)) : continue
    
    # Save clip to BIDS format
    fname = os.path.join(cwd, 'temp.edf')
    signal_headers = highlevel.make_signal_headers(channels, sample_frequency=sfreq, physical_max=2000000, physical_min=-2000000)
    highlevel.write_edf(fname, clip, signal_headers)

    raw = mne.io.read_raw_edf(fname, verbose=False, preload=False)

    sessionNum = 1
    runNumber = 1
    
    bids_path = BIDSPath(subject=subject, session='preimplant'+str(sessionNum).zfill(4), 
        run=runNumber, datatype='eeg', root='./bidsData', task='task')

    write_raw_bids(raw, bids_path=bids_path, verbose=0)


      







