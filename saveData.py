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

from eeg_funcs import *

from mne_bids import BIDSPath, write_raw_bids, print_dir_tree, update_sidecar_json
import mne_bids
from pyedflib import highlevel

# np.set_printoptions(threshold=sys.maxsize)


# Download clips from ieeg
cwd = os.getcwd()
config = json.load(open(os.path.join(cwd, 'config.json'), 'r'))
f = open(config['pwd_fpath'], 'rb')
pwd = f.read().decode()
session = Session(config['username'], pwd)
cnt = 0
powers = []
sessionNum = 0

# Load in admissionToSubject.json
with open('admissionToSubject.json', 'r') as json_file:
    admissionToSubject = json.load(json_file)

# Dict to keep track of which session (ie: admission id) is on which run
runsDict = {}

fileNames = ['EMU0190_Day01_1', 'EMU0190_Day02_1']

# for file in config['ieeg_filenames'] :
for file in fileNames :

    print(cnt)
    cnt += 1
    # if (cnt != 6) :
    #     continue

    sessionNum += 1

    ## LOAD FILE ##
    try :
        dataset = session.open_dataset(file)
    except :
        print('Error on file ', cnt)
        continue
    ch_names = dataset.get_channel_labels() # Get channel labels and put them in ch_names array 
    details = dataset.get_time_series_details(ch_names[0]) # Assign time_series_details object to details variable
    sfreq = details.sample_rate 
    ch_indices = [i for i in range(len(ch_names))] # ch_indices array (array of indices for each channel)
    pairs = get_annotation_times(dataset, 'EEG clip times')
    data_clips = []

    for i in range(len(pairs)) :
        print('-----', i)
        start, end = pairs[i][0], pairs[i][1]
        while (True) :
            try :
                data = load_full_channels(dataset, end-start, sfreq, ch_indices, offset_time=start)
                break
            except :
                continue
        data = data.T
        data_clips.append(data)
    
    ## Chop off NaNs from downloaded clips ##
    for i in range(len(data_clips)) :
        clip = data_clips[i]
        max_first_num = 0
        min_last_num = len(clip[0])-1
        for j in range(len(clip)) :
            first_num = 0
            last_num = len(clip[0])-1
            while (np.isnan(clip[j][first_num])) :
                first_num += 1
            while (np.isnan(clip[j][last_num])) :
                last_num -= 1
            max_first_num, min_last_num = max(max_first_num, first_num), min(min_last_num, last_num)
        # Adjust start/end times of the clip appropriately
        pairs[i][0] += max_first_num / sfreq
        pairs[i][1] -= (len(clip[0]) - 1 - min_last_num) / sfreq
        data_clips[i] = data_clips[i][:, max_first_num:min_last_num+1] # Cut out nan segments of data
    
    ## Run clips through preprocessing ##
    # for i in range(len(data_clips)) :
    #     data_clips[i] = preprocess(data_clips[i], sfreq)
    
    labels = [(('ekg' in ch_names[i]) or ('ecg' in ch_names[i])) for i in range(len(ch_names))]
    ch_types = ['ecg' if (labels[i]==True) else 'eeg' for i in range(len(labels))]

    # Create edf file from data clips and export to bids
    info = mne.create_info(ch_names, sfreq, ch_types, verbose=False)
    clip_number = 1
    for clip in data_clips :
        fname = os.path.join(cwd, 'temp.edf')
        signal_headers = highlevel.make_signal_headers(ch_names, sample_frequency=sfreq, physical_max=2000000, physical_min=-2000000)
        highlevel.write_edf(fname, clip, signal_headers)

        raw = mne.io.read_raw_edf(fname, verbose=False, preload=False)

        admission = file[:7]
        admissionID = file[3:7]
        runNumber = 0
        if (admissionID in runsDict) :
            runNumber = runsDict[admissionID] + 1
            runsDict[admissionID] += 1
        else :
            runNumber = 1
            runsDict[admissionID] = 1
        
        bids_path = BIDSPath(subject=admissionToSubject[admission], session='preimplant'+admissionID, 
            run=runNumber, datatype='eeg', root='./bids', task='task')

        write_raw_bids(raw, bids_path=bids_path, verbose=0)

        # Update json sidecar file with ieeg file name for the run
        bids_path = BIDSPath(subject=admissionToSubject[admission], session='preimplant'+admissionID, 
            run=runNumber, datatype='eeg', root='./bids', task='task', extension='.json')
        update_sidecar_json(bids_path=bids_path, entries={"fileName": file})

        clip_number += 1




