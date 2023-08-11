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
# Dict that will contain the final data. Keys are of the form 
# subject,timeWindow,channelName (eg: 'sub-0001,5s,fp1'), and values are arrays of 
# shape (numBands, sampleSize)
cutoffData = {}

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

# Loop through each subject, download their data from ieeg.org, get the averaged
# psd of the data, and save the sampled data to cutoffData
for subject, admissions in subjectToAdmissions.items() :
    ####
    # if (int(subject) < 189) : continue
    if (not subject == '0045') : continue
    ####
    print()
    print(subject)
    # Array of arrays that are the individual files for this subject
    fileClips = []
    # Loop through all admissions for this subject
    for admission in admissions :
        print(admission)
        # Loop through all ieeg files in this admission
        for file in admissionToFilenames[admission] :
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
    
    # Concatenate all arrays in fileClips 
    if (len(fileClips) == 0) : continue
    clip = np.concatenate(fileClips, axis=1)

    # Check if all channels are present
    if (channels == None or len(channels) < len(channelsRef)) : continue
    
    # Iterate through channels and timeWindows, get the averaged psd and sample it,
    # and then save it to cutoffData
    for timeWindow in timeWindows :
        for i in range(len(channels)) :
            try :
                averagedPsds = averagePsds(clip[i], sfreq, timeWindow, numBands)
                # Sample averagedPsds
                inds = [i for i in range(len(averagedPsds[0]))]
                # Check if inds is large enough to sample
                if (len(inds) < sampleSize) : continue
                sampledInds = np.random.choice(inds, size=sampleSize, replace=False)
                averagedPsds = averagedPsds[:, sampledInds]

                # Add averagedPsds to cutoffData
                key = ['sub-' + subject, str(timeWindow) + 's', channels[i]]
                key = ','.join(key)
                print(key)
                cutoffData[key] = averagedPsds.tolist()

                # Save what we have of cutoffData so far
                with open('cutoffData3.json', "w") as json_file:
                    json.dump(cutoffData, json_file)
            except :
                continue


      







