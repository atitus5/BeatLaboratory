#mic.py


import math

import numpy as np
from numpy.fft import rfft
import numpy.linalg as LA
import pandas as pd
from scipy.signal import hamming, lfilter
from scipy.stats import kurtosis
from sklearn.tree import DecisionTreeClassifier
import time

import sys
sys.path.append('..')

from common.audio import *

kNumMFCCs = 13
kFFTBins = 512
kEnergyBands = 8

# Read in the Mel frequency filter bank at initialization
kMelFilterBank = pd.read_csv("mel_filters.csv", sep=",", header=None).as_matrix().T


# Used to handle streaming audio input data, quantized to beats in a song, and
# return events corresponding to beatbox events on each beat
# NOTE: currently only accepts mono audio (i.e. num_channels = 1)
class MicrophoneHandler(object) :
    def __init__(self, num_channels, slop_frames, mic_buf_size):
        super(MicrophoneHandler, self).__init__()

        assert(num_channels == 1)

        # Set up audio buffer
        self.slop_frames = slop_frames
        self.mic_buf_size = mic_buf_size
        self.buf_size = 2 * self.slop_frames
        print "Buf size is %d frames (%.3f seconds)" % (self.buf_size, self.buf_size / float(kSampleRate))
        self.buf = np.zeros(self.buf_size, dtype=np.float32)
        self.buf_idx = 0

        self.processing_audio = False

        # Set up pre-emphasis filter coefficients
        self.s_pe = 1.0      # Output; s_pe[n]
        self.s_of = [1.0, -0.97]     # Input;  s_of[n] - 0.97s_of[n - 1]

        # Set up and cache window for signal
        self.window = hamming(self.buf_size)

        # Set up DCT matrix to convert from MSFCs to MFCCs
        # C_{ij} = cos(pi * i / 23.0 * (j + 0.5)) 
        i_vec = np.asarray(np.multiply(np.pi, range(kNumMFCCs)) / float(kMelFilterBank.shape[0]))
        j_vec = np.asarray(map(lambda j: j + 0.5, range(kMelFilterBank.shape[0])))
        self.dct = np.cos(np.dot(i_vec.reshape((i_vec.size, 1)), j_vec.reshape((1, j_vec.size))))

        # Set up a persistant buffer for MFCCs so we don't waste memory
        self.mfcc_buffer = np.empty(kNumMFCCs, dtype=np.float64)

        # Store data for training the classifier
        self.feature_count = kNumMFCCs + kEnergyBands + 2   # MFCCs, energies, zero-crossings, spectral Kurtosis
        self.training_data = []     # 2D array of (features, label) rows
        self.classifier = DecisionTreeClassifier()     # For now, use SVM classifier with no special parameters
        self.event_to_label = {
            0: "kick",
            1: "hihat",
            2: "snare",
            254: ""
        }

    # Receive data and send back a feature vector for the current window, if buffer fills
    def add_training_data(self, data, label):
        # Start processing audio again, if we aren't already
        self.processing_audio = True

        # Check if we will need to return a feature vector
        buffer_full = (self.buf_size - self.buf_idx) < len(data)
        if buffer_full:
            # Fill as much as we can, then reset index
            self.buf[self.buf_idx:] = np.multiply(data[:self.buf_size - self.buf_idx], self.window[self.buf_idx:])
            self.buf_idx = 0

            # Convert the buffer to features
            feature_vec = self._get_feature_vec()
            if feature_vec is not None:
                self.training_data.append([feature_vec, label])

            # Clear buffer out
            # NOTE: not strictly necessary, since it is overwritten later --- feel free
            # to delete if performance issues arise
            self.buf[:] = 0

            # Wait until we are told again to start processing audio
            self.processing_audio = False
        else:
            # Fill 'er up!
            self.buf[self.buf_idx:self.buf_idx + len(data)] = data
            self.buf_idx += len(data)

    def train_classifier(self):
        # Train up our classifier using the data we've collected so far!
        start_t = time.time()

        features = []
        labels = []
        for sample in xrange(len(self.training_data)):
            features.append(self.training_data[sample][0])
            labels.append(self.training_data[sample][1])
        features = np.asarray(features)
        labels = np.asarray(labels)

        self.classifier.fit(features, labels)

        end_t = time.time()
        elapsed_t = end_t - start_t
        print "Trained classifier in %.6f seconds" % elapsed_t

    # Receive data and send back a string indicating the event that occurred, if requested.
    # Returns empty string if no event occurred
    def add_data(self, data):
        event = ""

        # Start processing audio again, if we aren't already
        self.processing_audio = True

        # Check if we will need to classify an event
        buffer_full = (self.buf_size - self.buf_idx) < len(data)
        if buffer_full:
            # Fill as much as we can, then reset index
            self.buf[self.buf_idx:] = np.multiply(data[:self.buf_size - self.buf_idx], self.window[self.buf_idx:])
            self.buf_idx = 0

            # Classify the event now that we have a full buffer
            event = self._classify_event()

            # Clear buffer out
            # NOTE: not strictly necessary, since it is overwritten later --- feel free
            # to delete if performance issues arise
            self.buf[:] = 0

            # Wait until we are told again to start processing audio
            self.processing_audio = False
        else:
            # Fill 'er up!
            self.buf[self.buf_idx:self.buf_idx + len(data)] = data
            self.buf_idx += len(data)

        return event

    def _get_feature_vec(self):
        # Pre-emphasize signal (we need to recognize those snare/hi-hat fricatives!!)
        emphasized_audio = lfilter(self.s_of, self.s_pe, self.buf)

        # Take real-optimized FFT of emphasized audio signal
        spectrum = np.fft.rfft(emphasized_audio, n=kFFTBins)

        # Compute Mel-frequency Spectral Coefficients (MFSCs)
        abs_spectrum = np.asarray(map(abs, spectrum)).T
        mel_energies = np.dot(kMelFilterBank, abs_spectrum)
        if np.count_nonzero(mel_energies) < len(mel_energies):
            # We're going to divide by zero ABORT ABORT ABORT
            return None
        mfsc = np.maximum(np.multiply(-50.0, np.ones(kMelFilterBank.shape[0])),
                          np.log(mel_energies))

        # Compute Mel-frequency Cepstral Coefficients (MFCCs)
        # mfcc[i] = sum_{j=1}^{23.0} (mfsc[j] * cos(pi * i / 23.0 * (j - 0.5))
        #         = C * msfc
        mfcc = np.dot(self.dct, mfsc.reshape((mfsc.shape[0], 1)))
        mfcc = mfcc.reshape((mfcc.size))

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
        feature_vec = np.empty(self.feature_count)
        feature_vec[:kNumMFCCs] = mfcc
        feature_vec[kNumMFCCs:kNumMFCCs + kEnergyBands] = normalized_energies
        feature_vec[kNumMFCCs + kEnergyBands] = zero_crossings
        feature_vec[kNumMFCCs + kEnergyBands + 1] = kurt

        print ",".join(map(str, list(feature_vec)))

        return feature_vec

    # Takes our full buffer of windowed data and classifies it as an appropriate beatbox sound
    def _classify_event(self):
        # Get features (in the format scikit-learn expects)
        feature_vec = self._get_feature_vec()
        print ",".join(map(str, list(feature_vec)))
        feature_vec = np.reshape(feature_vec, (1, len(feature_vec)))

        # Classify it! (Woah, that's what this line of code does?!)
        event = self.classifier.predict(feature_vec)[0]
        classification = self.event_to_label[event]

        '''
        print "Event: %s" % str(event)
        print "Label: %s" % classification
        '''

        return classification