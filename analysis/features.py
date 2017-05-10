import cPickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import sys

sys.path.append('..')

from engine.classifier import *

if len(sys.argv) != 3:
    print "Usage: python features.py <features file (Pickle)> <labels file (Pickle)>"
    sys.exit(1)

# Read in features and labels so we can plot them
features_filename = sys.argv[1]
labels_filename = sys.argv[2]
with open(features_filename, "rb") as fid:
    features = cPickle.load(fid)
with open(labels_filename, "rb") as fid:
    labels = cPickle.load(fid)

'''
# Normalize rows
normalized_features = normalize(features, axis=0, norm="l1")
'''

# Order spectrogram by classification type
kick_features = []
hihat_features = []
snare_features = []
silence_features = []
total_events = labels.shape[0]

kFeatureStart = 1
kFeatureEnd = kFeatureStart + 1
# kFeatureEnd = normalized_features.shape[1]
# kFeatureEnd = features.shape[1]

for i in xrange(total_events):
    label = labels[i]
    if label == kKick:
        kick_features.append(features[i, kFeatureStart:kFeatureEnd])
    elif label == kSnare:
        snare_features.append(features[i, kFeatureStart:kFeatureEnd])
    elif label == kHihat:
        hihat_features.append(features[i, kFeatureStart:kFeatureEnd])
    elif label == kSilence:
        silence_features.append(features[i, kFeatureStart:kFeatureEnd])

kick_features = np.asarray(kick_features)
hihat_features = np.asarray(hihat_features)
snare_features = np.asarray(snare_features)
silence_features = np.asarray(silence_features)
padding = np.zeros((1, kFeatureEnd - kFeatureStart))

ordered_features = np.concatenate((snare_features, padding,
                                   hihat_features))
'''
ordered_features = np.concatenate((kick_features, padding,
                                   snare_features, padding,
                                   hihat_features, padding,
                                   silence_features))
'''
print ordered_features

y_labels = ["" for i in xrange(total_events)]

y_labels[0] = "Snare"
y_labels[snare_features.shape[0] + 1] = "Hihat"
'''
y_labels[0] = "Kick"
y_labels[kick_features.shape[0] + 1] = "Hihat"
y_labels[kick_features.shape[0] + 1 + hihat_features.shape[0] + 1] = "Snare"
# y_labels[kick_features.shape[0] + 1 + hihat_features.shape[0] + 1 + snare_features.shape[0] + 1] = "Silence"
'''

plt.pcolor(ordered_features, cmap="gnuplot2")
plt.yticks(np.arange(0, len(y_labels), 1))
plt.gca().set_yticklabels(y_labels)
plt.show()
