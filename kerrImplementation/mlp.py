import numpy as np
import torch
import json 
import os 

with open('features.json', 'r') as file:
    featuresDict = json.load(file)

print(len(featuresDict['0188']))



