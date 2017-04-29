#extract.py

import numpy as np
import scipy as sp
from ctypes import *

from mic import *


kNumMFCCs = 12

# Used to extract features from audio input buffers
# NOTE: currently only accepts stereo audio (i.e. num_channels = 2)
class FeatureExtractor(object) :
    def __init__(self, num_channels):
        super(FeatureExtractor, self).__init__()

        assert(num_channels == 2)

        # Set up function pointer
        self.mfcc_extractor = cdll.LoadLibrary("mfcc.so")

        # 

        # Set up a persistant buffer for MFCCs so we don't waste memory!
        self.mfcc_buffer = np.zeros(kNumMFCCs, dtype=np.float64)

    # Receive data and send back a Numpy array of Mel-Frequency Cepstral Coefficients
    def extract_mfccs(self, audio_buffer):
        self.mfcc_extractor.extract_mfcc(audio_buffer.ctypes.data, self.mfcc_buffer.ctypes.data,
                                         c_int(kBufferSize), c_int(kNumMFCCs))
        return self.mfcc_buffer
