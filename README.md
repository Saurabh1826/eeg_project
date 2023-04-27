# EEG Preprocessing Package

This package provides tools for preprocessing and working with EEG files. It provides functions for plotting, filtering, calculating features from EEG
signals, and more. The file example_analysis.py shows an example pipeline that uses the functions of the package to preprocess an EEG clip downloaded 
from ieeg.org and calculate various features from it. The file ekg_channel_detector.py provides code to train and run a gradient boosting classifier to
classify a given channel in an EEG clip as an EKG or non-EKG channel.


## Config File

The file config.json contains necessary information to run the package's code on your local machine. The following is a description of the keys of the 
config file: 

username: Your ieeg.org account username
pwd_fpath: The path to the file that is your UTF-8 encoded ieeg.org password
dataset: The name of the ieeg.org dataset to be used in the example_analysis.py pipeline
ieeg_filenames: An array of the ieeg.org dataset names to be used for training and testing by the ekg_channel_detector.py code
