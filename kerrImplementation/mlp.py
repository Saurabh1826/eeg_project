import json
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
import sys
sys.path.append('.')
from eeg_funcs import *
from sklearn.svm import SVC
from imblearn.ensemble import EasyEnsembleClassifier, BalancedRandomForestClassifier, BalancedBaggingClassifier

# Read in admissionsToSubject.json
admissionToSubject = json.load(open('admissionToSubject.json', 'r'))

# Read in labels.json
admissionToLabel = json.load(open('labels.json', 'r'))

# Create a dictionary where keys are subjects and values are diagnosis (1 for 
# epilepsy, and 0 for PNES)
subjectToLabel = {}
for key, val in admissionToLabel.items() :
    subject = admissionToSubject[key]
    subjectToLabel[subject] = 1 if val == '1.0' else 0

# Read in features.json
# featuresDict = json.load(open('features.json', 'r'))
featuresDict = json.load(open('temp.json', 'r'))

statParams = ['min', 'max', 'std', 'mean']

# Dict of arrays of features. Keys are subjects and values are arrays of features
subjectToFeatures = {subject: [] for subject in subjectToLabel.keys()}

# subjectSet = set()

# Populate featureArr and labels
cnt = 0
for featureName, arr in featuresDict.items() :
    # Get subject from feature name
    subject = featureName.split(',')[0][4:]
    ##########
    # subjectSet.add(subject)
    # if (subject == '0045') :
    #     print(arr)
    # print(subject)
    #########
    # Check if we have a label for this subject
    if (not subject in subjectToLabel) : continue
    # Get features from arr
    arr = np.array(arr)
    means = np.mean(arr, axis=1)
    mins = np.min(arr, axis=1)
    maxes = np.max(arr, axis=1)
    stds = np.std(arr, axis=1)
    # Add features to subjectToFeatures
    newFeatures = means.tolist() + mins.tolist() + maxes.tolist() + stds.tolist()
    subjectToFeatures[subject] += newFeatures
#############
# cnt = 0
# set2 = set()
# for subject, arr in subjectToFeatures.items() :
#     if (not len(arr) > 0) : 
#         set2.add(subject)
# print(set2)
# set2 = [int(x) for x in set2]
# set2.sort()
# print(set2)
# exit(0)
# for subject, arr in subjectToFeatures.items() :
#     nan_mask = np.isnan(arr)
#     if (np.sum(nan_mask) > 0) :
#         cnt += 1
#         print(subject)
# exit(0)
#############
# Array of arrays of features, where each array is the feature array for a single
# subject
featureArr = []
# Array of labels corresponding to featureArr (1 for epilepsy, 0 for PNES)
labels = []

# Populate featureArr, labels
for subject, arr in subjectToFeatures.items() :
    featureArr.append(arr)
    labels.append(subjectToLabel[subject])

# Remove the elements of featureArr/labels that we don't have any features for
mask = [True if len(featureArr[i]) > 0 else False for i in range(len(featureArr))]
featureArr = [featureArr[i] for i in range(len(featureArr)) if mask[i]]
labels = [labels[i] for i in range(len(labels)) if mask[i]]


# Impute missing values in featureArr with -1
featureArr = impute(featureArr)

# Conver featureArr, labels to numpy arrays
featureArr = np.array(featureArr)
labels = np.array(labels)

# Remove rows of featureArr that contain NaNs
nan_rows_mask = np.isnan(featureArr).any(axis=1)
featureArr = featureArr[~nan_rows_mask]
labels = labels[~nan_rows_mask]

####################################
# nan_mask = np.isnan(featureArr)
# print(np.sum(nan_mask))
# cnt = 0
# for i in range(len(featureArr)) :
#     nan_mask = np.isnan(featureArr[i])
#     if (np.sum(nan_mask) > 0) :
#         cnt += 1
#         print(featureArr[i])
#         break
# exit(0)
####################################

mlpClf = MLPClassifier(hidden_layer_sizes=(10,5), random_state=42, max_iter=50)
gradBoostingCLf = GradientBoostingClassifier(n_estimators=10, random_state=42)
eec = EasyEnsembleClassifier(random_state=42, n_estimators=20)
balancedRandomForestClf = BalancedRandomForestClassifier(random_state=42, n_estimators=50)
balancedBaggingClf = BalancedBaggingClassifier(random_state=42, n_estimators=10)
# svmClf = SVC(kernel='linear', class_weight={0: 2, 1: 1}) 
# mlpClf.fit(featureArr, labels)
# print(mlpClf.score(featureArr, labels))

# X, y = balancedSubsample(featureArr, labels)
print(losoCrossVal(balancedBaggingClf, featureArr, labels))
# print(losoCrossVal(gradBoostingCLf, featureArr, labels))
# print(losoCrossVal(gradBoostingCLf, featureArr, labels))
# print(losoCrossVal(mlpClf, X, y))
# print(losoCrossVal(mlpClf, featureArr, labels))
# print(losoCrossVal(mlpClf, X, y))

# print(losoCrossVal(svmClf, featureArr, labels))
# plotPcaScatter(featureArr, labels)

# svmClf = SVC(kernel='linear', class_weight='balanced') 
# print(losoCrossVal(svmClf, featureArr, labels))
# svmClf = SVC(kernel='poly', class_weight='balanced') 
# print(losoCrossVal(svmClf, featureArr, labels))
# svmClf = SVC(kernel='rbf', class_weight='balanced') 
# print(losoCrossVal(svmClf, featureArr, labels))
# svmClf = SVC(kernel='sigmoid', class_weight='balanced') 
# print(losoCrossVal(svmClf, featureArr, labels))
