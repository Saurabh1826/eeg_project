import json

# Read in admissionsToSubject.json
admissionToSubject = json.load(open('admissionToSubject.json', 'r'))

# Read in labels.json
admissionToLabel = json.load(open('labels.json', 'r'))

# Create a dictionary where keys are subjects and values are diagnosis (1 for 
# epilepsy, and 0 for PNES)
dataDict = {}
for key, val in admissionToLabel.items() :
    subject = admissionToSubject[key]
    dataDict[subject] = 1 if val == '1.0' else 0

# Read in features.json
featuresDict = json.load(open('features.json', 'r'))

statParams = ['min', 'max', 'std', 'mean']

# Dict of channels
channelsDict = {'fp1': 0, 'fp2': 1, 'f3': 2, 'f4': 3, 'f7': 4, 'f8': 5, 'fz': 6, 'c3': 7, 'c4': 8, 'cz': 9, 't3': 10, 't4': 11, 't5': 12, 'p3': 13, 'p4': 14, 't6': 15, 'o1': 16, 'o2': 17}

# Array of the features. Each element in the array is an array of length 4




