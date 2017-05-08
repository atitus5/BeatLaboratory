#classifier.py


import math

import cPickle
import numpy as np
from numpy.fft import rfft
import numpy.linalg as LA
import pandas as pd
from scipy.signal import lfilter
from scipy.stats import kurtosis
import time

import sys
sys.path.append('..')

from common.audio import *

kChunkSize = int(kSampleRate * .050)    # 50 ms
kNumMFCCs = 13
kFFTBins = 512
kEnergyBands = 16

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

        # TODO
        # Set up pre-emphasis filter coefficients
        self.s_pe = 1.0      # Output; s_pe[n]
        self.s_of = [1.0, -0.97]     # Input;  s_of[n] - 0.97s_of[n - 1]

        '''
        # Set up DCT matrix to convert from MSFCs to MFCCs
        # C_{ij} = cos(pi * i / 23.0 * (j + 0.5)) 
        i_vec = np.asarray(np.multiply(np.pi, range(kNumMFCCs)) / float(kMelFilterBank.shape[0]))
        j_vec = np.asarray(map(lambda j: j + 0.5, range(kMelFilterBank.shape[0])))
        self.dct = np.cos(np.dot(i_vec.reshape((i_vec.size, 1)), j_vec.reshape((1, j_vec.size))))
        '''


    def compute_features(self, audio_data):
        # Pre-emphasize signal (we need to recognize those snare/hi-hat fricatives!!)
        emphasized_audio = lfilter(self.s_of, self.s_pe, audio_data)

        # Take real-optimized FFT of emphasized audio signal
        spectrum = np.fft.rfft(emphasized_audio, n=kFFTBins)

        # Compute Mel-frequency Spectral Coefficients (MFSCs)
        abs_spectrum = np.asarray(map(abs, spectrum)).T
        mel_energies = np.dot(kMelFilterBank, abs_spectrum)
        if np.count_nonzero(mel_energies) < len(mel_energies):
            # We're going to divide by zero... add a tiny epsilon to the energies
            eps = 1e-9
            mel_energies = np.add(np.multiply(eps, np.ones(len(mel_energies))), mel_energies)
            
        '''
        mfsc = np.maximum(np.multiply(-50.0, np.ones(kMelFilterBank.shape[0])),
                          np.log(mel_energies))

        # Compute Mel-frequency Cepstral Coefficients (MFCCs)
        # mfcc[i] = sum_{j=1}^{23.0} (mfsc[j] * cos(pi * i / 23.0 * (j - 0.5))
        #         = C * msfc
        mfcc = np.dot(self.dct, mfsc.reshape((mfsc.shape[0], 1)))
        mfcc = mfcc.reshape((mfcc.size))
        '''

        # From Eran's input demo
        zero_crossings = np.count_nonzero(emphasized_audio[1:] * emphasized_audio[:-1] < 0)

        # Compute spectral kurtosis (spectra with distinct peaks have larger values than scattered ones)
        kurt = abs(kurtosis(spectrum))

        # Compute normalized energies in subsets of Mel bands
        energies = []
        band_size = len(mel_energies) / float(kEnergyBands)
        for i in xrange(kEnergyBands):
            band_start = int(i * band_size)
            band_end = int((i + 1) * band_size)
            energies.append(sum(mel_energies[band_start:band_end]))
        normalized_energies = np.divide(energies, LA.norm(energies))

        # Compose feature vector
        #feature_vec = FeatureVector(mfcc=mfcc, energies=normalized_energies, zc=zero_crossings, kurt=kurt)
        feature_vec = FeatureVector(energies=normalized_energies, zc=zero_crossings, kurt=kurt)

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

    # Takes a feature vector and returns a string label for it
    def predict(self, feature_array):
        # Defaults to returning silence
        return kEventToLabel[kSilence]


# Use handtuned constants to classify beatbox events
class ManualClassifier(BeatboxClassifier) :
    def __init__(self):
        super(ManualClassifier, self).__init__()

        # TODO: set up parameters

    # Fit our classifier to labels
    def fit(self, feature_arrays, labels):
        # Log features and labels so we can load them in analysis scripts
        with open("features.pkl", "wb") as fid:
            cPickle.dump(feature_arrays, fid)
        with open("labels.pkl", "wb") as fid:
            cPickle.dump(labels, fid)

        # Defaults to nothing
        return

    # Override so that prediction is, ya know, actually done
    def predict(self, feature_array):
        #TODO: actually classify features
        label = super(ManualClassifier, self).predict(feature_array)
        return label
