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

'''
# Normalize rows
normalized_features = np.asarray(normalize(features, axis=0, norm="l1"))
'''

# See how good our classification is!
total_events = labels.shape[0]
correct = 0
for i in xrange(total_events):
    label = labels[i]

    lfe = features[i, 0]
    zc = features[i, 1]

    classification = kSilence
    if lfe >= kSilenceLFE:
        # It's NOT silence!
        if zc <= kKickZC:
            # It's a kick!
            classification = kKick
        elif zc <= kHihatZC:
            # It's a snare!
            classification = kSnare
        else:
            # It must be a hi-hat then
            classification = kHihat

        '''
        if zc >= kHihatZC:
            # It's a hi-hat!
            classification = kHihat
        elif decay >= kDecay:
            # It's a kick!
            classification = kKick
        else:
            # It must be a snare then
            classification = kSnare
        '''

    if classification == label:
        print "Correctly classified %d" % label
        correct += 1
    else:
        print "Misclassified %d as %d" % (label, classification)
        pass

print "Manual thresholds accuracy: %.3f" % (correct / float(total_events))

'''
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

# Classify using gradient boosted regression trees
from sklearn import ensemble
clf = ensemble.GradientBoostingClassifier()
clf.fit(features_scaled, labels)
correct = 0
for i in xrange(total_events):
    label = labels[i]
    classification = clf.predict([features_scaled[i, :]])[0]
    if classification == label:
        correct += 1
print "GBRT accuracy: %.3f" % (correct / float(total_events))
'''


