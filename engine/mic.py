#mic.py


import math

import numpy as np
from numpy.fft import rfft
from scipy.signal import hamming
import time

import sys
sys.path.append('..')

from common.audio import *


# Buffering concept: maintain N buffers, where (N-1) are active
# and one is "free" for overflow when a buffer fills. The "real-time"
# buffer is updated with the newest data while the delayed buffers
# are updated with delayed data. The "real-time" buffer rotates regularly.
kBufferTime = 0.100     # 100 ms
kBufferShift = 0.010    # 10 ms
kBufferSize = int(kSampleRate * kBufferTime)
kBufferSpacing = int(kSampleRate * kBufferShift)
kBufferCount = int(math.ceil(kBufferSize / float(kBufferSpacing))) + 1

from extract import *
from classifier import *

# Used to handle streaming audio input data and return events corresponding
# to beatbox events
# NOTE: currently only accepts stereo audio (i.e. num_channels = 2)
class MicrophoneHandler(object) :
    def __init__(self, num_channels):
        super(MicrophoneHandler, self).__init__()

        assert(num_channels == 2)

        # Set up audio buffers
        self.buffers = np.zeros((kBufferCount, kBufferSize), dtype=np.float32)
        self.buffer_indices = np.zeros(kBufferCount, dtype=np.int)
        self.current_buffer = kBufferCount - 1  # Track which buffer is currently "real-time"; starts at end
        for i in xrange(kBufferCount - 1):
            # Fill in previous buffers, except the first one which is still unused
            self.buffer_indices[i + 1] = i * kBufferSpacing

        # Set up and cache window for signal
        self.window = hamming(kBufferSize)

        # Set up our manager for extracting features from audio buffers for classification
        self.feature_extractor = FeatureExtractor(num_channels)
        self.training = False
        self.training_start_t = 0
        self.mfccs_buffer = []
        self.labels = []

        # Set up our classifier for determining events from MFCCs
        self.classifier = SVMClassifier()
        self.is_trained = False

    # Receive data and send back a string indicating the event that occurred.
    # Returns empty string if no event occurred
    def add_data(self, data):
        add_time = time.time()
        event = ""
        buffer_filled = self._update_buffers(data)

        if buffer_filled:
            # Extract events
            # TODO: get this actually working
            mfccs = self.feature_extractor.extract_mfccs(self.buffers[self.current_buffer])
            if self.training:
                self.mfccs_buffer.append(mfccs)
                training_t = add_time - self.training_start_t
                t_to_gem = abs(training_t - self.gems[self.gem_idx][0])
                if t_to_gem < kBufferTime:
                    # Add label for this current time
                    self.labels.append(self.gems[self.gem_idx][1])
                    self.gem_idx += 1
                else:
                    self.labels.append(-1)  # No label

            if self.is_trained:
                event = self.classifier.classify(mfccs)

            # Move to next active buffer. Don't worry about clearing buffer,
            # as it will be fully overwritten before being used again
            self.current_buffer = (self.current_buffer - 1) % kBufferCount


        return event

    def start_training(self, gems):
        self.mfccs_buffer = []
        self.labels = []
        self.gems = gems
        self.gem_idx = 0
        self.training_start_t = time.time()
        self.training = True

    def stop_training(self):
        self.training = False
        self.training_start_t = 0

    def train_classifier(self):
        self.classifier.train(self.mfccs_buffer, self.labels)
        self.is_trained = True


    # Update buffers in streaming fashion, windowing them in the process.
    # Returns True if our current buffer fills and False otherwise
    def _update_buffers(self, data):
        completed_buffer = False

        # First handle "real-time" buffer, as it may fill up here
        buf_current = self.current_buffer   # Declared here locally, as it can change below
        buf_idx = self.buffer_indices[buf_current]
        buffer_full = (kBufferSize - buf_idx) < len(data)
        if buffer_full:
            # Fill as much as we can, then reset its index
            self.buffers[buf_current][buf_idx:] = np.multiply(data[:kBufferSize - buf_idx], self.window[buf_idx:])
            self.buffer_indices[buf_current] = 0
            completed_buffer = True

            # Fill up free one with tail of data
            overlap = len(data) - (kBufferSize - buf_idx)
            free_idx = buf_current - (kBufferCount - 1)
            self.buffers[free_idx][0:overlap] = np.multiply(data[kBufferSize - buf_idx:], self.window[:overlap])
            self.buffer_indices[free_idx] = overlap
        else:
            self.buffers[buf_current][buf_idx:buf_idx + len(data)] = data
            self.buffer_indices[buf_current] += len(data)

        # Now handle the delayed buffers (except the free one)
        for i in xrange(kBufferCount - 2): 
            # Note: these indices can be negative, but this is okay, as
            # Python arrays can use negative indices to wrap around end
            buf = buf_current - i - 1
            buf_idx = self.buffer_indices[buf]

            # Fill as much as we can
            data_len = min(len(data), kBufferSize - buf_idx)
            self.buffers[buf][buf_idx:buf_idx + data_len] = np.multiply(data[:data_len], self.window[buf_idx:buf_idx + data_len])
            self.buffer_indices[buf] += data_len

        return completed_buffer
