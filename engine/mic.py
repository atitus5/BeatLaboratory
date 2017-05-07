#mic.py


import math

import numpy as np
from numpy.fft import rfft
from scipy.signal import hamming, lfilter

import sys
sys.path.append('..')

from common.audio import *


kChunkSize = int(kSampleRate * 0.025)   # 25 ms
kFFTBins = 256


# Used to handle streaming audio input data, quantized to beats in a song, and
# return events corresponding to beatbox events on each beat
# NOTE: currently only accepts stereo audio (i.e. num_channels = 2)
class MicrophoneHandler(object) :
    def __init__(self, num_channels, slop_frames, mic_buf_size):
        super(MicrophoneHandler, self).__init__()

        assert(num_channels == 2)

        # Set up audio buffer
        self.slop_frames = slop_frames
        self.mic_buf_size = mic_buf_size
        self.buf_size = (2 * self.slop_frames) + (2 * mic_buf_size)   # Just in case it overlaps
        self.buf = np.zeros(self.buf_size, dtype=np.float32)
        self.buf_idx = 0

        self.processing_audio = False

        # Set up pre-emphasis filter coefficients
        self.s_pe = 1.0      # Output; s_pe[n]
        self.s_of = [1.0, -0.97]     # Input;  s_of[n] - 0.97s_of[n - 1] 

        # Set up and cache window for signal
        self.window = hamming(self.buf_size)

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

    # Takes our full buffer of windowed data and classifies it as an appropriate beatbox sound
    def _classify_event(self):
        classification = ""

        # Pre-emphasize signal (we need to recognize those snare/hi-hat fricatives!!)
        emphasized_audio = lfilter(self.s_of, self.s_pe, self.buf)

        # Write sentinel value to our output for beginning
        print ",".join(map(str, [0.5 for x in xrange((kFFTBins / 2) + 1)]))

        # Take real-optimized FFT of each chunk and convert to power
        for chunk in xrange(int(math.ceil(len(emphasized_audio) / float(kChunkSize)))):
            audio_chunk = emphasized_audio[chunk * kChunkSize:(chunk + 1) * kChunkSize]
            freq_powers = abs(rfft(audio_chunk, n=kFFTBins))
            print ",".join(map(str, freq_powers))

        # Write sentinel value to our output for end
        print ",".join(map(str, [0.5 for x in xrange((kFFTBins / 2) + 1)]))

        return classification