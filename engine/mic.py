#mic.py


import math

import numpy as np
from numpy.fft import rfft
from scipy.signal import hamming

import sys
sys.path.append('..')

from common.audio import *


# Buffering concept: maintain N buffers, where (N-1) are active
# and one is "free" for overflow when a buffer fills. The "real-time"
# buffer is updated with the newest data while the delayed buffers
# are updated with delayed data. The "real-time" buffer rotates regularly.
kBufferSize = int(kSampleRate * 0.100)      # 25 ms
kBufferSpacing = int(kSampleRate * 0.010)   # 10 ms
kBufferCount = int(math.ceil(kBufferSize / float(kBufferSpacing))) + 1
kBufferFFTBins = 2 ** int(math.ceil(math.log(kBufferSize, 2)))

# Experimentally determined constants
kDCThreshold = 6.0
kBassIndexThreshold = kBufferFFTBins / 64
kTrebleBassRatio = 3.0
kDebounceBackoff = 0.925  # Used to prevent multiple classifications in a row

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

        # Track last time we classified, so we can debounce the signal
        self.last_classified = 1024     # Arbitrarily high number so that we start off with no debouncing

    # Receive data and send back a string indicating the event that occurred.
    # Returns empty string if no event occurred
    def add_data(self, data):
        event = ""
        buffer_filled = self._update_buffers(data)

        if buffer_filled:
            # Figure out what event just occurred!
            event = self._classify_event(self.buffers[self.current_buffer])

            # Move to next active buffer. Don't worry about clearing buffer,
            # as it will be fully overwritten before being used again
            self.current_buffer = (self.current_buffer - 1) % kBufferCount

        return event

    # Takes a full buffer of windowed data and classifies it into 
    def _classify_event(self, data):
        classification = ""

        # Take real-optimized FFT and convert to power
        freq_powers = abs(rfft(data, n=kBufferFFTBins))

        # Debounce the power calculation so we don't classify unnecessarily often
        dc_debounced = freq_powers[0] * (1.0 - (kDebounceBackoff ** self.last_classified))

        # Really simple kick/snare classification
        # Only bother checking if we should classify if there is a beat box attempt happening
        if dc_debounced > kDCThreshold:
            bass_power = sum(freq_powers[1:kBassIndexThreshold])
            treble_power = sum(freq_powers[kBassIndexThreshold:])
            treble_bass_ratio = treble_power / float(bass_power)

            if treble_bass_ratio > kTrebleBassRatio:
                classification = "snare"
            else:
                classification = "kick"

            # Reset our last classified counter
            self.last_classified = 0
        else:
            self.last_classified += 1

        return classification

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
