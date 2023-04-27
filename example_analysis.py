## Imports and helpful functions for importing data from IEEG.org

# Uncomment if using colab
# !pip install git+https://github.com/aestrivex/bctpy.git
# !pip install git+https://github.com/ieeg-portal/ieegpy.git
# from google.colab import files
# from google.colab import drive
# drive.mount('/content/gdrive')
# !cp gdrive/MyDrive/eeg_funcs.py ./eeg_funcs.py

from ieeg.auth import Session
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
import json
from eeg_funcs import *
# import mtspec

# Open config file
config = json.load(open('config.json', 'r'))


# ####
# # OPEN IEEG SESSION
# ####

f = open(config['pwd_fpath'], 'rb')
pwd = f.read().decode()
session = Session(config['username'], pwd)
dataset = session.open_dataset(config['dataset'])

ch_names = dataset.get_channel_labels() # Get channel labels and put them in ch_names array
details = dataset.get_time_series_details(ch_names[0]) # Assign time_series_details object to details variable
sfreq = details.sample_rate 

ch_indices = [i for i in range(len(ch_names))] # ch_indices array (array of indices for each channel)
num_samples = details.number_of_samples
duration = num_samples / sfreq # Duration of the entire recording (number of samples divided by sampling frequency)

pairs = get_annotation_times(dataset, 'EEG clip times')
data_clips = []

for i in range(len(pairs[:5])) :
  start, end = pairs[i][0], pairs[i][1]
  print(i, end-start)
  data = load_full_channels(dataset, end-start, sfreq, ch_indices, offset_time=start)
  data = data.T
  data_clips.append(data)


# Chop off NaNs from downloaded clips

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

for i in range(len(data_clips)) :
  assert np.isnan(data_clips[i]).any() == False, f"Clip {i} of data_clips contains NaNs"


# Run clip through preprocessing

clip_ind = 0
clip = np.copy(data_clips[clip_ind])

# Run filtering on the clip and common average reference it
notch = create_notch_filter(4, 55, 65, sfreq)
high_pass = create_high_pass_filter(4, 0.5, sfreq)
# low_pass = create_low_pass_filter()
b, a = sig.butter(N=4, Wn=15, btype='lowpass', fs=sfreq)

padding_right = np.copy(clip)
padding_left = np.copy(clip)
clip = np.concatenate((clip, padding_right), axis=1)
clip = np.concatenate((padding_left, clip), axis=1)

# clip = apply_filter(clip, notch)
# clip = apply_filter(clip, high_pass)
clip = apply_filter(clip, (b, a))

# Remove padding 
clip = clip[:, len(padding_left[0]):-len(padding_right[0])]

clip = common_average_reference(clip)



# Plot eeg
plot_eeg(data_clips[0], sfreq, ch_names=ch_names, figsize=(30, 18))

# Segment clip into 10 second long clips with 5 second overlap
times, clips = split_data(clip, sfreq, 10, 5)

# Extract UTMs
utm_funcs = [line_length, area, energy, zero_crossings, peak_to_peak]
utms = extract_utms(clips, utm_funcs)

# Obtain graphs from the data clips
graphs = data_to_graphs(clips, 'coherence', sfreq)

# Extract graph metric features
graph_funcs = [avg_node_strength, global_efficiency, avg_clustering_coef, char_path_len, avg_eccentricity]
graph_feats = extract_graph_metrics(graphs, graph_funcs)

# Plot evolution over time of some extracted features
names = ['Average Node Strength', 'Global Efficiency', 'Average Clustering Coefficient','Characteristic Path Length','Average Eccentricity']
plot_metrics(features=graph_feats, times=times, names=names)

# Plot median values of UTMs 

utms = extract_utms(clips, [line_length, area, energy, zero_crossings, peak_to_peak])
utms = np.median(utms, axis=1)
plot_metrics(utms, times, ['Line Length', 'Area', 'Energy', 'Zero Crossings', 'Peak to Peak'])

