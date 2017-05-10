#classifier.py

import math

import cPickle
import numpy as np
import time

import sys
sys.path.append('..')

from common.audio import *

# Map event classes to labels
kKick = 0
kHihat = 1
kSnare = 2
kSilence = 254
kNoEvent = 255
kEventToLabel = {
    kKick: "kick",
    kHihat: "hihat",
    kSnare: "snare",
    kSilence: "silence",
    kNoEvent: ""
}
kLabelToEvent = {
    "kick": kKick,
    "hihat": kHihat,
    "snare": kSnare,
    "silence": kSilence,
    "": kNoEvent
}


# Computes features for a given window of audio data
class FeatureManager(object):
    def __init__(self):
        super(FeatureManager, self).__init__()

    def compute_features(self, audio_data):
        scaling = len(audio_data) / float(np.count_nonzero(audio_data))

        # Log frame energy of DC-filtered (NOT pre-emphasis) signal
        dc_comp = np.mean(audio_data)
        dc_filtered = audio_data - dc_comp
        fe = sum([dc_filtered[i] ** 2 for i in xrange(len(dc_filtered))])
        lfe = max(-50.0, np.log(fe))
        scaled_lfe = lfe * scaling

        # From Eran's input demo
        zero_crossings = np.count_nonzero(audio_data[1:] * audio_data[:-1] < 0)
        scaled_zc = int(zero_crossings * scaling)

        feature_vec = np.array([scaled_lfe, scaled_zc])
        
        return feature_vec


# Base class for various beatbox event classifiers
class BeatboxClassifier(object):
    def __init__(self):
        super(BeatboxClassifier, self).__init__()

        # No more setup should be necessary?

    # Fit our classifier to labels
    def fit(self, feature_arrays, labels):
        # Defaults to nothing
        return

    def supports_partial_fit(self):
        # Defaults to false
        return False

    # Update our model with a new feature array. Not possible for all 
    def partial_fit(self, feature_arrays, labels):
        # Defaults to an exception
        raise Exception("partial_fit not supported by this classifier")

    # Takes a feature vector and returns a string label for it
    def predict(self, feature_array):
        # Defaults to returning silence
        return kEventToLabel[kSilence]


# Hand-tuned constants that work pretty well
kSilenceLFE = 0.0
kKickZC = 700
kHihatZC = 2100

# Use handtuned constants to classify beatbox events
class ManualClassifier(BeatboxClassifier) :
    def __init__(self):
        super(ManualClassifier, self).__init__()

        # Hand-tuned to start
        # self.decay = kDecay
        self.silence_lfe = kSilenceLFE
        self.kick_zc = kKickZC
        self.hihat_zc = kHihatZC

    # Fit our classifier to labels
    def fit(self, feature_arrays, labels):
        # Recompute constants based on what we just heard as training data
        kick_features = []
        hihat_features = []
        snare_features = []
        silence_features = []
        total_events = labels.shape[0]

        # Format our data
        for i in xrange(total_events):
            label = labels[i]
            if label == kKick:
                kick_features.append(feature_arrays[i, :])
            elif label == kSnare:
                snare_features.append(feature_arrays[i, :])
            elif label == kHihat:
                hihat_features.append(feature_arrays[i, :])
            elif label == kSilence:
                silence_features.append(feature_arrays[i, :])
        kick_features = np.asarray(kick_features)
        hihat_features = np.asarray(hihat_features)
        snare_features = np.asarray(snare_features)
        silence_features = np.asarray(silence_features)

        # Determine a good LFE threshold for classifying out silence
        silence_avg_lfe = np.mean(silence_features[:, 0])
        nonsilence_avg_lfe = np.mean(np.concatenate((kick_features[:, 0], hihat_features[:, 0], snare_features[:, 0])))
        self.silence_lfe = (silence_avg_lfe + nonsilence_avg_lfe) / 2.0

        # Determine a good zero-crossing threshold for classifying between kick and non-kick
        kick_avg_zc = np.mean(kick_features[:, 1])
        snare_avg_zc = np.mean(snare_features[:, 1])
        hihat_avg_zc = np.mean(hihat_features[:, 1])
        nonkick_avg_zc = min(snare_avg_zc, hihat_avg_zc)
        self.kick_zc = int((kick_avg_zc + nonkick_avg_zc) / 2.0)

        # Determine a good zero-crossing threshold for classifying between snare and hi-hat
        self.hihat_zc = int((snare_avg_zc + hihat_avg_zc) / 2.0)

        print "Using silence LFE threshold %.3f, kick ZC threshold %d, hihat ZC threshold %d" % (self.silence_lfe,
                                                                                                 self.kick_zc,
                                                                                                 self.hihat_zc)

        '''
        # Log features and labels so we can load them in analysis scripts
        with open("features.pkl", "wb") as fid:
            cPickle.dump(feature_arrays, fid)
        with open("labels.pkl", "wb") as fid:
            cPickle.dump(labels, fid)
        '''

    def supports_partial_fit(self):
        # Nah
        return False
        '''
        # We do! Look at dat
        return True
        '''

    '''
    def partial_fit(self, feature_arrays, label):
        features_scaled = scale(feature_arrays)
        classes = (kKick, kHihat, kSnare, kSilence)
        self.clf.partial_fit([features_scaled], [label], classes)
    '''

    # Takes a feature vector and returns a string label for it
    def predict(self, feature_array):
        lfe = feature_array[0]
        zc = feature_array[1]

        classification = kSilence
        if lfe >= self.silence_lfe:
            # It's NOT silence!
            if zc <= self.kick_zc:
                # It's a kick!
                classification = kKick
            elif zc <= self.hihat_zc:
                # It's a snare!
                classification = kSnare
            else:
                # It must be a snare then
                classification = kHihat

        return classification
