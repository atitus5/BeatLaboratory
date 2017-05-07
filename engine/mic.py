#mic.py


import math

import numpy as np
from numpy.fft import rfft
from scipy.signal import hamming, lfilter

import sys
sys.path.append('..')

from common.audio import *


# Used to handle streaming audio input data, quantized to beats in a song, and
# return events corresponding to beatbox events on each beat
# NOTE: currently only accepts stereo audio (i.e. num_channels = 2)
class MicrophoneHandler(object) :
    def __init__(self, num_channels, max_frames, slop_frames):
        super(MicrophoneHandler, self).__init__()

        assert(num_channels == 2)

        # Set up audio buffers
        self.slop_frames = slop_frames
        self.max_frames = max_frames
        self.bufsize = self.max_frames + (2 * self.slop_frames)

        # "Last" buffer: buffer storing audio whose end slop window overlaps with the "current" buffer
        self.last_buffer = np.zeros(self.bufsize, dtype=np.float32)
        self.last_buffer_idx = self.slop_frames + self.max_frames   # Only end slop window

        # "Current" buffer: buffer storing audio whose start slop window overlaps with the "last" buffer
        self.current_buffer = np.zeros(self.bufsize, dtype=np.float32)
        self.current_buffer_idx = self.slop_frames      # Ignore start slop window

        # Set up pre-emphasis filter coefficients
        self.s_pe = 1.0      # Output; s_pe[n]
        self.s_of = [1.0, -0.97]     # Input;  s_of[n] - 0.97s_of[n - 1] 

        # Set up and cache window for signal
        self.window = hamming(self.bufsize)

    # Receive data and send back a string indicating the event that occurred.
    # Returns empty string if no event occurred
    def add_data(self, data, record):
        return ""
        '''
        add_time = time.time()
        event = ""
        buffer_filled = self._update_buffers(data)

        if buffer_filled:
            # Extract event
            event = self._classify_event(self.buffers[self.current_buffer], record)

            # Move to next active buffer. Don't worry about clearing buffer,
            # as it will be fully overwritten before being used again
            self.current_buffer = (self.current_buffer - 1) % kBufferCount

        return event
        '''

    # Takes a full buffer of windowed data and classifies it into 
    def _classify_event(self, data, record):
        classification = ""

        '''
        # Pre-emphasize signal (we need to recognize those snare/hi-hat fricatives!!)
        # emphasized_audio = lfilter(self.s_of, self.s_pe, data)

        # Take real-optimized FFT and convert to power
        # freq_powers = abs(rfft(emphasized_audio, n=kBufferFFTBins))
        freq_powers = abs(rfft(data, n=kBufferFFTBins))

        # Debounce the frequencies so we don't classify unnecessarily often
        freq_powers_debounced = np.multiply(1.0 - (kDebounceBackoff ** self.last_classified), freq_powers)

        if record:
            print ",".join(map(str, freq_powers))
        '''

        return classification

    # Update buffers in streaming fashion, windowing them in the process.
    # Returns True if our current buffer fills and False otherwise
    def _update_buffers(self, data):
        return False
        '''
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
        '''
