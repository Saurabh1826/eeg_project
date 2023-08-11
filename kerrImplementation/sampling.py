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

            
# print(len(signals))
# print(signals[0].shape)
# print(ch_names)
# print(channels)

# Variable for number of frequency bands 
numBands = 100

# print(channels)

timeWindows = [5, 60]
freqs = [i for i in range(0, numBands)]
statParams = ['min', 'max', 'mean', 'stdev']

# Dict of features. Keys are subject names, and values are arrays of features. 
# The elements of the values are the features in the form of dictionaries with 
# keys being the feature value, ch_name, timeWindow, freq, statParam
featuresDict = {}

sampleSizes = [5, 10, 25, 50, 100, 500]
# sampleSizes = [5, 10, 25, 50, 100]

subject = '0188'
# channelNum = 13
timeWindowInd = 0

clips = signals[subject]

# Dicts to hold the sample features calculated below. Keys are 'channels' and 
# 'features'. Values are arrays of corresponding channel names and features. 
# Dicts are specific to a particular sample size
meanSampleFeatures = {'channels': [], 'features': []}
minSampleFeatures = {'channels': [], 'features': []}
maxSampleFeatures = {'channels': [], 'features': []}
stdSampleFeatures = {'channels': [], 'features': []}

f, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(24, 12))

for i in range(len(channels)) :
    if (i > 3) : break
    print('----', i)
    averagedPsds = averagePsds(clips[i], sfreq, timeWindows[timeWindowInd], numBands)
    numSamples = 1000
    meanVars, minVars, maxVars, stdVars = [], [], [], []
    meanVarsDict = {'sample size': [], 'variance': []}
    d = {'sample size': [], 'features': []}
    for freq in range(numBands) :
        if (freq in [58, 59, 60, 61, 62]) : continue
        print('---------', freq)
        for size in sampleSizes :
            # print(f'Sample size: {size}')
            means, mins, maxes, stds = [], [], [], []
            for _ in range(numSamples) :
                sample = np.random.choice(averagedPsds[freq], size=size, replace=False)
                means.append(np.mean(sample))
                mins.append(np.min(sample))
                maxes.append(np.max(sample))
                stds.append(np.std(sample))


            means = np.array(means)
            mins = np.array(mins)
            maxes = np.array(maxes)
            stds = np.array(stds)

            d['sample size'] += [size] * numSamples
            d['features'] += means.tolist()

            meanVarsDict['sample size'].append(size)
            meanVarsDict['variance'].append(np.std(means) ** 2)
            
            meanVars.append(np.std(means) ** 2)
            minVars.append(np.std(mins) ** 2)
            maxVars.append(np.std(maxes) ** 2)
            stdVars.append(np.std(stds) ** 2)

            # Normalize the sample features
            # means = (means - np.mean(means)) / np.std(means) 
            # mins = (mins - np.mean(mins)) / np.std(mins) 
            # maxes = (maxes - np.mean(maxes)) / np.std(maxes) 
            # stds = (stds - np.mean(stds)) / np.std(stds) 

            if (size == 50) :
                meanSampleFeatures['channels'] += [channels[i]] * numSamples
                meanSampleFeatures['features'] += means.tolist()
                minSampleFeatures['channels'] += [channels[i]] * numSamples
                minSampleFeatures['features'] += mins.tolist()
                maxSampleFeatures['channels'] += [channels[i]] * numSamples
                maxSampleFeatures['features'] += maxes.tolist()
                stdSampleFeatures['channels'] += [channels[i]] * numSamples
                stdSampleFeatures['features'] += stds.tolist()


            # print(f'Sample mean: {np.mean(means)}')
            # print(f'Sample min: {np.mean(mins)}')
            # print(f'Sample max: {np.mean(maxes)}')
            # print(f'Sample std: {np.mean(stds)}')
            # print()


        # print(averagedPsds.shape)
        # print(f'True mean: {np.mean(averagedPsds[0])}')
        # print(f'True min: {np.min(averagedPsds[0])}')
        # print(f'True max: {np.max(averagedPsds[0])}')
        # print(f'True std: {np.std(averagedPsds[0])}')
        # print(meanVars)
        # print(minVars)
        # print(maxVars)
        # print(stdVars)

        # plt.plot(sampleSizes, meanVars, marker='.')
        # plt.ylabel('Variance')
        # plt.xlabel('Sample size')
        # plt.savefig('fig.png')
        # plt.show()

        # Violinplot of features vs sample size for a given channel and time window
        # plt.figure(figsize=(12, 8))
        # plot = sns.violinplot(pd.DataFrame.from_dict(d), x='sample size', y='features')
        # plt.xlabel('Sample Size', fontsize=25)
        # plt.ylabel('Mean', fontsize=25)
        # plt.xticks(fontsize=25)
        # fig_padding = 0.1
        # plt.subplots_adjust(left=fig_padding, right=1-fig_padding, top=1-fig_padding, bottom=fig_padding)
        # fig = plot.get_figure()
        # fig.savefig("out.png") 
        # exit(0)

    # Line plot of variance vs sample size 
    # plot = sns.lineplot(pd.DataFrame.from_dict(meanVarsDict), x='sample size', 
    #     y='variance', orient='x')
    # plt.xlabel('')
    # plt.ylabel('')
    # plt.xticks(fontsize=25)
    # plt.yticks(fontsize=25)
    # axes[r][c].title.set_text(channels[i])
    # axes[r][c].set(xlabel=None, ylabel=None)
    c = i % 2
    r = int((i - (i % 2)) / 2)
    plot = sns.lineplot(pd.DataFrame.from_dict(meanVarsDict), x='sample size', 
        y='variance', orient='x', ax=axes[r][c])
    axes[r][c].title.set_text('')
    axes[r][c].title.set_fontsize(25) 
    axes[r][c].set(xlabel=None, ylabel=None)
    axes[r][c].tick_params(axis='x', labelsize=25)
    # fig = plot.get_figure()
    # fig.savefig("out.png") 
    # exit(0)
# plt.xlabel('Sample Size')
# plt.ylabel('Variance')
plt.xlabel('')
plt.ylabel('')
# plt.xticks(fontsize=25)
# for ax in axes:
#     ax.tick_params(axis='x', labelsize=25)
f.savefig('out1.png')

# plot = sns.violinplot(pd.DataFrame.from_dict(meanSampleFeatures), x='channels', y='features')
# fig = plot.get_figure()
# fig.savefig("out.png") 

