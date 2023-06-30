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

from mne_bids import BIDSPath, write_raw_bids, print_dir_tree
import mne_bids
from pyedflib import highlevel

# np.set_printoptions(threshold=sys.maxsize)

# bids_path = BIDSPath(subject='0001', session='preimplant'+str(6).zfill(2), run=3, 
#     datatype='eeg', root='./bids_dataset', task='positive')
# raw = mne_bids.read_raw_bids(bids_path)
# newClip = raw.get_data()
# print(clip[0][:20])
# plt.plot(clip[3])
# plt.show()
# exit(0)


# Download clips from ieeg
cwd = os.getcwd()
config = json.load(open(os.path.join(cwd, 'config.json'), 'r'))
f = open(config['pwd_fpath'], 'rb')
pwd = f.read().decode()
session = Session(config['username'], pwd)
cnt = 0
powers = []
sessionNum = 0

for file in config['ieeg_filenames'] :
    if ('EMU0002' in file) : 
        break

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

    # signals, signal_headers, header = highlevel.read_edf('temp.edf', ch_names=ch_names)
    # print(signals)
    # break

    for i in range(len(pairs)) :
        start, end = pairs[i][0], pairs[i][1]
        # DOWNLOAD ONLY 15 SECONDS OF EACH CLIP
        data = load_full_channels(dataset, min(end-start, 15), sfreq, ch_indices, offset_time=start)
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
    for i in range(len(data_clips)) :
        data_clips[i] = preprocess(data_clips[i], sfreq)
    
    labels = [(('ekg' in ch_names[i]) or ('ecg' in ch_names[i])) for i in range(len(ch_names))]
    ch_types = ['ecg' if (labels[i]==True) else 'eeg' for i in range(len(labels))]

    # print(data_clips[0][0][:50])
    # plt.plot(data_clips[2][3])
    # plt.show()
    # exit(0)

    # Create edf file from data clips and export to bids
    info = mne.create_info(ch_names, sfreq, ch_types, verbose=False)
    clip_number = 1
    for clip in data_clips :
        # raw = mne.io.RawArray(clip, info, verbose=False)
        fname = os.path.join(cwd, 'temp.edf')
        # mne.export.export_raw(fname, raw, fmt='edf', add_ch_type=False, overwrite=True, verbose=False)
        signal_headers = highlevel.make_signal_headers(ch_names, sample_frequency=sfreq, physical_max=20000, physical_min=-20000)
        highlevel.write_edf(fname, clip, signal_headers)
        # fname = os.path.join(cwd, 'temp.fif')
        # raw.save(fname, overwrite=True)

        raw = mne.io.read_raw_edf(fname, verbose=False, preload=False)
        
        # raw = mne.io.read_raw_fif(fname, verbose=False, preload=False)

        subject = file[3:7]
        bids_path = BIDSPath(subject=subject, session='preimplant'+str(sessionNum).zfill(2), run=clip_number, 
            datatype='eeg', root='./bids_dataset', task='positive')
        # mne_bids.read_raw_bids(bids_path)
        # break
        write_raw_bids(raw, bids_path=bids_path, verbose=0)

        clip_number += 1
    
    # Get powers for verifying ekg channels
    # clip_number = 1
    # for clip in data_clips :
    #     max_freq = 10
    #     # ind = (int) (max_freq * len(clip[0]) / sfreq)
    #     ind = 100
    #     power_arrs = []
    #     for i in range(len(ch_names)) :
    #         power_arrs.append(psd(clip[i], sfreq)[1][:ind])
    #     powers += power_arrs
    #     clip_number += 1




