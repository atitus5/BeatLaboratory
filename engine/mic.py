#mic.py


import math

import numpy as np
from scipy.signal import hamming
import time

import sys
sys.path.append('..')

from common.audio import *

from classifier import *

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

        # Set up and cache window for signal
        self.window = hamming(self.buf_size)

        # Set up our feature manager for creating feature vectors from audio
        self.feature_manager = FeatureManager()

        # Store data for training the classifier
        self.training_data = []     # 2D array of (features, label) rows
        self.classifier = ManualClassifier()

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
            feature_vector = self.feature_manager.compute_features(self.buf)
            self.training_data.append([feature_vector, label])

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
            features.append(self.training_data[sample][0].asarray())
            labels.append(self.training_data[sample][1])
        features = np.asarray(features)
        labels = np.asarray(labels)

        self.classifier.fit(features, labels)
        self.training_data = None

        end_t = time.time()
        elapsed_t = end_t - start_t
        print "Trained classifier in %.6f seconds" % elapsed_t

    # Receive data and send back a string indicating the event that occurred, if requested.
    # Returns empty string if no event occurred
    def add_data(self, data, label):
        event = ""

        # Start processing audio again, if we aren't already
        self.processing_audio = True

        # Check if we will need to classify an event
        buffer_full = (self.buf_size - self.buf_idx) < len(data)
        if buffer_full:
            # Fill as much as we can, then reset index
            self.buf[self.buf_idx:] = np.multiply(data[:self.buf_size - self.buf_idx], self.window[self.buf_idx:])
            self.buf_idx = 0

            # Get features (in the format our classifier expects)
            feature_vec = self.feature_manager.compute_features(self.buf).asarray()

            # Update our model as we go, if the classifier supports it
            if self.classifier.supports_partial_fit():
                self.classifier.partial_fit(feature_vec, label)

            # Classify the event now that we have a full buffer
            event = self._classify_event(feature_vec)

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

    # Takes our full buffer of windowed data and classifies it as an appropriate beatbox sound
    def _classify_event(self, feature_vec):
        # Classify it! (Woah, that's what this line of code does?!)
        classification = self.classifier.predict(feature_vec)
        label = kEventToLabel[classification]
        return label