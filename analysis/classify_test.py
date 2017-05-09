import cPickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, scale
import sys

sys.path.append('..')

from engine.classifier import *

if len(sys.argv) != 3:
    print "Usage: python classify_test.py <features file (Pickle)> <labels file (Pickle)>"
    sys.exit(1)

# Read in features and labels
features_filename = sys.argv[1]
labels_filename = sys.argv[2]
with open(features_filename, "rb") as fid:
    features = cPickle.load(fid)
with open(labels_filename, "rb") as fid:
    labels = cPickle.load(fid)

# Normalize rows
normalized_features = np.asarray(normalize(features, axis=0, norm="l1"))

# Hand tuned constants
kSilenceLFE = 0.0
kHihatZC = 0.0375
kHighFreqStartIdx = kFFTBins / 4
kSnareHighFreq = 0.114

# See how good our classification is!
total_events = labels.shape[0]
correct = 0
for i in xrange(total_events):
    label = labels[i]

    high_freq_sum = sum(normalized_features[i, kHighFreqStartIdx:(kFFTBins / 2) + 1])
    lfe = normalized_features[i, (kFFTBins / 2) + 1]
    zc = normalized_features[i, (kFFTBins / 2) + 2]

    classification = 254
    if lfe >= kSilenceLFE:
        # It's NOT silence!
        if zc >= kHihatZC:
            classification = 1
        elif high_freq_sum >= kSnareHighFreq:
            classification = 2
        else:
            classification = 0

    if classification == label:
        # print "Correctly classified %d" % label
        correct += 1
    else:
        # print "Misclassified %d as %d" % (label, classification)
        pass

print "Manual thresholds accuracy: %.3f" % (correct / float(total_events))


# Whiten the data
features_scaled = scale(features)

# Classify using SVMs
from sklearn import svm
clf = svm.SVC()
clf.fit(features_scaled, labels)
correct = 0
for i in xrange(total_events):
    label = labels[i]
    classification = clf.predict([features_scaled[i, :]])[0]
    if classification == label:
        correct += 1
print "SVM accuracy: %.3f" % (correct / float(total_events))

# Classify using decision trees
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(features_scaled, labels)
correct = 0
for i in xrange(total_events):
    label = labels[i]
    classification = clf.predict([features_scaled[i, :]])[0]
    if classification == label:
        correct += 1
print "Decision Tree accuracy: %.3f" % (correct / float(total_events))

# Classify using random forests of decision trees
from sklearn import ensemble
clf = ensemble.RandomForestClassifier()
clf.fit(features_scaled, labels)
correct = 0
for i in xrange(total_events):
    label = labels[i]
    classification = clf.predict([features_scaled[i, :]])[0]
    if classification == label:
        correct += 1
print "Random Forest accuracy: %.3f" % (correct / float(total_events))

# Classify using extremely randomized trees
from sklearn import ensemble
clf = ensemble.ExtraTreesClassifier()
clf.fit(features_scaled, labels)
correct = 0
for i in xrange(total_events):
    label = labels[i]
    classification = clf.predict([features_scaled[i, :]])[0]
    if classification == label:
        correct += 1
print "Extremely Randomized Trees accuracy: %.3f" % (correct / float(total_events))

# Classify using nearest neighbors
from sklearn import neighbors
for neighbor_count in xrange(1, 10):
    clf = neighbors.KNeighborsClassifier(n_neighbors=neighbor_count)
    clf.fit(features_scaled, labels)
    correct = 0
    for i in xrange(total_events):
        label = labels[i]
        classification = clf.predict([features_scaled[i, :]])[0]
        if classification == label:
            correct += 1
    print "%d-Nearest Neighbors accuracy: %.3f" % (neighbor_count, (correct / float(total_events)))
