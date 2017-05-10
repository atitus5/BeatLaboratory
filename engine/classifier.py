#classifier.py


import math

import cPickle
import numpy as np
from numpy.fft import rfft
import numpy.linalg as LA
import pandas as pd
from scipy.signal import lfilter
from scipy.stats import kurtosis
from sklearn.preprocessing import normalize, scale
from sklearn.ensemble import *
from sklearn.neural_network import *
from sklearn.naive_bayes import *
import time

import sys
sys.path.append('..')

from common.audio import *

kNumMFCCs = 32
kFFTBins = 512
kEnergyBands = 23

# Read in the Mel frequency filter bank at initialization
kMelFilterBank = pd.read_csv("../engine/mel_filters.csv", sep=",", header=None).as_matrix().T

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

# Simple container class with defined features
class FeatureVector(object):
    def __init__(self, **kwargs):
        self.raw_features = {}
        self.feature_count = 0
        for key, val in kwargs.iteritems():
            # Vals must be numbers or array-like
            self.raw_features[key] = val
            if hasattr(val, '__len__'):
                self.feature_count += len(val)
            else:
                self.feature_count += 1

    def get_features(self):
        return self.raw_features

    # Return the entries as a serial numpy array
    def asarray(self):
        feature_array = np.empty(self.feature_count)
        current_idx = 0
        for key in sorted(self.raw_features.keys()):
            val = self.raw_features[key]
            if hasattr(val, '__len__'):
                feature_array[current_idx:current_idx + len(val)] = val
                current_idx += len(val)
            else:
                feature_array[current_idx] = val
                current_idx += 1
        return feature_array

# Computes features for a given window of audio data
class FeatureManager(object):
    def __init__(self):
        super(FeatureManager, self).__init__()

        # Set up pre-emphasis filter coefficients
        self.s_pe = 1.0      # Output; s_pe[n]
        self.s_of = [1.0, -0.97]     # Input;  s_of[n] - 0.97s_of[n - 1]

        # Set up DCT matrix to convert from MSFCs to MFCCs
        # C_{ij} = cos(pi * i / 23.0 * (j + 0.5)) 
        i_vec = np.asarray(np.multiply(np.pi, range(kNumMFCCs)) / float(kMelFilterBank.shape[0]))
        j_vec = np.asarray(map(lambda j: j + 0.5, range(kMelFilterBank.shape[0])))
        self.dct = np.cos(np.dot(i_vec.reshape((i_vec.size, 1)), j_vec.reshape((1, j_vec.size))))

    def compute_features(self, audio_data):
        # Take real-optimized FFT of audio signal
        spectrum = np.fft.rfft(audio_data, n=kFFTBins)

        # Log frame energy of DC-filtered (NOT pre-emphasis) signal
        dc_comp = np.mean(audio_data)
        dc_filtered = audio_data - dc_comp
        fe = sum([dc_filtered[i] ** 2 for i in xrange(len(dc_filtered))])
        lfe = max(-50.0, np.log(fe))

        abs_spectrum = np.asarray(map(abs, spectrum)).T

        '''
        # Compute Mel-frequency Spectral Coefficients (MFSCs)
        mel_energies = np.dot(kMelFilterBank, abs_spectrum)
            
        mfsc = np.maximum(np.multiply(-50.0, np.ones(kMelFilterBank.shape[0])),
                          np.log(mel_energies))

        # Compute Mel-frequency Cepstral Coefficients (MFCCs)
        # mfcc[i] = sum_{j=1}^{23.0} (mfsc[j] * cos(pi * i / 23.0 * (j - 0.5))
        #         = C * msfc
        mfcc = np.dot(self.dct, mfsc.reshape((mfsc.shape[0], 1)))
        mfcc = mfcc.reshape((mfcc.size))
        '''

        # From Eran's input demo
        zero_crossings = np.count_nonzero(audio_data[1:] * audio_data[:-1] < 0)

        '''
        # Compute normalized energies in subsets of Mel bands
        energies = []
        band_size = len(mel_energies) / float(kEnergyBands)
        for i in xrange(kEnergyBands):
            band_start = int(i * band_size)
            band_end = int((i + 1) * band_size)
            energies.append(sum(mel_energies[band_start:band_end]))
        # normalized_energies = np.divide(energies, LA.norm(energies))
        normalized_energies = normalize(np.reshape(energies, (1, -1)), axis=1, norm="l1")

        # normalize spectrum
        normalized_spectrum = normalize(abs_spectrum.reshape(1, -1), axis=1, norm="l1")

        # Compute rolloff and brightness
        rolloff_pct = 0.85
        rolloff_idx = 0         # Index in spectrum below which 85% of frame energy is
        brightness_idx = 34     # Approx. 1500 Hz
        brightness_energy = 0.0 # Energy above frequency corresponding to brightness_idx

        total_spectral_energy = sum([abs_spectrum[i] ** 2 for i in xrange(len(abs_spectrum))])
        energy_so_far = 0.0
        for i in xrange(len(abs_spectrum)):
            current_energy = abs_spectrum[i] ** 2
            energy_so_far += current_energy
            if energy_so_far <= (rolloff_pct * total_spectral_energy):
                rolloff_idx += 1
            if i >= brightness_idx:
                brightness_energy += current_energy

        # Percentage of energy above frequency corresponding to brightness_idx
        brightness = brightness_energy / total_spectral_energy

        # Compute pitch as spectral peak (ignoring DC and low-freq info)
        pitch_idx = np.argmax(abs_spectrum[2:])
        '''

        # Compute ratio of peak to average of rectified signal in order to get an idea of decay
        rectified_audio = abs(audio_data)
        rectified_peak = max(rectified_audio)
        rectified_avg = np.mean(rectified_audio)
        decay = rectified_peak / rectified_avg

        # Compose feature vector
        feature_vec = FeatureVector(decay=decay,
                                    lfe=lfe,
                                    zc=zero_crossings)
        
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



kDecay = 16.0
kSilenceLFE = 0.0
kHihatZC = 2100

# Use handtuned constants to classify beatbox events
class ManualClassifier(BeatboxClassifier) :
    def __init__(self):
        super(ManualClassifier, self).__init__()

        # Hand-tuned to start
        self.cutoff_idx = kFFTBins / 4

        # Uninitialized to start
        self.decay = kDecay
        self.silence_lfe = kSilenceLFE
        self.hihat_zc = kHihatZC

    # Fit our classifier to labels
    def fit(self, feature_arrays, labels):
        # Nothing to be done now - leave as pre-trained constants
        pass
        '''
        # Log features and labels so we can load them in analysis scripts
        with open("features.pkl", "wb") as fid:
            cPickle.dump(feature_arrays, fid)
        with open("labels.pkl", "wb") as fid:
            cPickle.dump(labels, fid)
        '''

    # Takes a feature vector and returns a string label for it
    def predict(self, feature_array):
        decay = feature_array[0]
        lfe = feature_array[1]
        zc = feature_array[2]

        classification = kSilence
        if lfe >= self.silence_lfe:
            # It's NOT silence!
            if zc >= self.hihat_zc:
                # It's a hi-hat!
                classification = kHihat
            elif decay >= self.decay:
                # It's a kick!
                classification = kKick
            else:
                # It must be a snare then
                classification = kSnare

        return classification


# Uses Gradient Boosted Regression Trees to classify beatbox events
class GBRTClassifier(BeatboxClassifier) :
    def __init__(self):
        super(GBRTClassifier, self).__init__()

        self.clf = GradientBoostingClassifier()

    # Fit our classifier to labels
    def fit(self, feature_arrays, labels):
        features_scaled = scale(feature_arrays)
        self.clf.fit(features_scaled, labels)
        print "Trained GBRT with score %.3f" % self.clf.score(features_scaled, labels)

    # Takes a feature vector and returns a string label for it
    def predict(self, feature_array):
        return self.clf.predict([feature_array])[0]



# Uses Extra Trees Classifier to classify beatbox events
class ETClassifier(BeatboxClassifier) :
    def __init__(self):
        super(ETClassifier, self).__init__()

        self.clf = ExtraTreesClassifier()

    # Fit our classifier to labels
    def fit(self, feature_arrays, labels):
        features_scaled = scale(feature_arrays)
        self.clf.fit(features_scaled, labels)
        print "Trained ET with score %.3f" % self.clf.score(features_scaled, labels)

    # Takes a feature vector and returns a string label for it
    def predict(self, feature_array):
        return self.clf.predict([feature_array])[0]



# Uses Multilayer Perceptions (MLPs)/Feedforward Neural Networks to classify beatbox events
class NNClassifier(BeatboxClassifier) :
    def __init__(self):
        super(NNClassifier, self).__init__()

        hidden_layer_sizes = (128, 128, 32, 128)
        self.clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes)

    # Fit our classifier to labels
    def fit(self, feature_arrays, labels):
        features_scaled = scale(feature_arrays)
        self.clf.fit(features_scaled, labels)
        print "Trained NN with score %.3f" % self.clf.score(features_scaled, labels)

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
        return self.clf.predict([feature_array])[0]

# Uses Gaussian Naive Bayesian classification to classify beatbox events
class GNBClassifier(BeatboxClassifier) :
    def __init__(self):
        super(GNBClassifier, self).__init__()

        self.clf = GaussianNB()

    # Fit our classifier to labels
    def fit(self, feature_arrays, labels):
        features_scaled = scale(feature_arrays)
        self.clf.fit(features_scaled, labels)
        print "Trained GNB with score %.3f" % self.clf.score(features_scaled, labels)

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
        return self.clf.predict([feature_array])[0]
