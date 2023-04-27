## Imports and helpful functions for importing data from IEEG.org

# Uncomment if using colab
# !pip install git+https://github.com/ieeg-portal/ieegpy.git
# !pip install git+https://github.com/aestrivex/bctpy.git
# from google.colab import files
# from google.colab import drive
# drive.mount('/content/gdrive')
# !cp gdrive/MyDrive/eeg_funcs.py ./eeg_funcs.py


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
import json
from sklearn.ensemble import GradientBoostingClassifier
from eeg_funcs import *

config = json.load(open('config.json', 'r'))

####
# OPEN IEEG SESSION
####

num_clips = []

d = {'ieeg_name': [], 'channel_names': [], 'is_ekg': [], 'clip_number': []}

powers = []
cnt = 0

f = open(config['pwd_fpath'], 'rb')
pwd = f.read().decode()
session = Session(config['username'], pwd)

for file in config['ieeg_filenames'][:2] :
  print(cnt)
  cnt += 1

  ## LOAD FILE ##
  dataset = session.open_dataset(file)
  ch_names = dataset.get_channel_labels() # Get channel labels and put them in ch_names array 
  details = dataset.get_time_series_details(ch_names[0]) # Assign time_series_details object to details variable
  sfreq = details.sample_rate 
  ch_indices = [i for i in range(len(ch_names))] # ch_indices array (array of indices for each channel)
  pairs = get_annotation_times(dataset, 'EEG clip times')
  data_clips = []

  num_clips.append(len(pairs))

  for i in range(len(pairs)) :
    start, end = pairs[i][0], pairs[i][1]
    # print(i, end-start)
    # DOWNLOAD ONLY 15 SECONDS OF EACH CLIP
    data = load_full_channels(dataset, min(end-start, 15), sfreq, ch_indices, offset_time=start)
    data = data.T
    data_clips.append(data)
  

  # Create labels list (position i is true if channel name is ekg/ecg and false otherwise)
  for i in range(len(ch_names)) :
    ch_names[i] = ch_names[i].lower()
  labels = [(('ekg' in ch_names[i]) or ('ecg' in ch_names[i])) for i in range(len(ch_names))]

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
  
  ## Get power spectra
  clip_number = 1
  for clip in data_clips :
    ind = 100
    power_arrs = []
    for i in range(len(ch_names)) :
      power_arrs.append(psd(clip[i], sfreq, plot=False, log_psd=False)[1][:ind])
    
    file_name = [file for i in range(len(ch_names))]
    d['ieeg_name'] += file_name
    d['channel_names'] += ch_names
    powers += power_arrs
    d['is_ekg'] += labels
    num = [clip_number] * len(ch_names)
    d['clip_number'] += num
    
    clip_number += 1



X = np.array(powers)
y = d['is_ekg']
y = np.array(y).astype(int)

mask1 = [True if (y[i] == 1) else False for i in range(len(y))]
mask0 = [False if (y[i] == 1) else True for i in range(len(y))]
X1 = X[mask1]
y1 = y[mask1]
X0 = X[mask0]
y0 = y[mask0]

train = (int) (0.5 * len(X1))
y_f = np.concatenate((np.array([0 for i in range(train)]), np.array([1 for i in range(train)])))

mdl = GradientBoostingClassifier()
np.random.shuffle(X1)
np.random.shuffle(X0)
X_f = np.concatenate((X0[:train], X1[:train]))
mdl.fit(X_f, y_f)


## Create training data and labels from dictionary d

X = np.array(powers)
y = d['is_ekg']
y = np.array(y).astype(int)


# Obtain a single guess for each clip

preds = mdl.predict(X)

# Reshape preds so rows are predictions for a given channel
preds_vote = preds.reshape(len(ch_names), -1, order='F') 

num_clips_cumsum = np.cumsum(np.array(num_clips))

for i in range(len(num_clips_cumsum)) :
  start, end = 0, 0
  if (i == 0) :
    start, end = 0, num_clips_cumsum[i]
  else :
    start, end = num_clips_cumsum[i-1], num_clips_cumsum[i]

  vote = np.sum(preds_vote[:, start:end], axis=1) / (end - start)
  for i in range(len(preds_vote)) :
    preds_vote[i, start:end] = 1 if (vote[i] > 0.5) else 0

preds_vote = preds_vote.reshape(-1, order='F')

np.set_printoptions(threshold=sys.maxsize)

diff = np.where(preds != y)[0]


df = pd.DataFrame.from_dict(d)
diff_df = df.iloc[diff]
final_df = diff_df.loc[df['clip_number'] == 1]
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(final_df)

