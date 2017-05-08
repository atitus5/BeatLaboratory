import cPickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import sys

sys.path.append('..')

from engine.classifier import *

if len(sys.argv) != 3:
    print "Usage: python spectrogram.py <features file (Pickle)> <labels file (Pickle)>"
    sys.exit(1)

# Read in features and labels so we can plot them
features_filename = sys.argv[1]
labels_filename = sys.argv[2]
with open(features_filename, "rb") as fid:
    features = cPickle.load(fid)
with open(labels_filename, "rb") as fid:
    labels = cPickle.load(fid)

# Normalize rows
normalized_features = normalize(features, axis=0, norm="l1")

f, axes = plt.subplots(1, 4)

# Plot feature spectrograms for kick
kick_features = []
for i in xrange(labels.shape[0]):
    label = labels[i]
    if label == kKick:
        kick_features.append(normalized_features[i, :])
axes[0].pcolor(kick_features, cmap="gnuplot2")
axes[0].set_title("Kick")

# Plot feature spectrograms for hihat
hihat_features = []
for i in xrange(labels.shape[0]):
    label = labels[i]
    if label == kHihat:
        hihat_features.append(normalized_features[i, :])
axes[1].pcolor(hihat_features, cmap="gnuplot2")
axes[1].set_title("Hihat")

# Plot feature spectrograms for snare
snare_features = []
for i in xrange(labels.shape[0]):
    label = labels[i]
    if label == kSnare:
        snare_features.append(normalized_features[i, :])
axes[2].pcolor(snare_features, cmap="gnuplot2")
axes[2].set_title("Snare")

# Plot feature spectrograms for silence
silence_features = []
for i in xrange(labels.shape[0]):
    label = labels[i]
    if label == kSilence:
        silence_features.append(normalized_features[i, :])
axes[3].pcolor(silence_features, cmap="gnuplot2")
axes[3].set_title("Silence")

plt.show()
