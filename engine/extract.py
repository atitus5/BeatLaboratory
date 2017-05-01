#extract.py

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from mic import *


kNumMFCCs = 12
kBufferFFTBins = 512

# Read in the Mel frequency filter bank at initialization
kMelFilterBank = pd.read_csv("mel_filters.csv", sep=",", header=None).as_matrix().T

# Used to extract features from audio input buffers
# NOTE: currently only accepts stereo audio (i.e. num_channels = 2)
class FeatureExtractor(object) :
    def __init__(self, num_channels):
        super(FeatureExtractor, self).__init__()

        assert(num_channels == 2)

        # Set up pre-emphasis filter coefficients
        self.s_pe = 1.0      # Output; s_pe[n]
        self.s_of = [1.0, -0.97]     # Input;  s_of[n] - 0.97s_of[n - 1] 

        # Set up DCT matrix to convert from MSFCs to MFCCs
        # C_{ij} = cos(pi * i / 23.0 * (j + 0.5)) 
        i_vec = np.asarray(np.multiply(np.pi, range(kNumMFCCs)) / float(kMelFilterBank.shape[0]))
        j_vec = np.asarray(map(lambda j: j + 0.5, range(kMelFilterBank.shape[0])))
        self.dct = np.cos(np.dot(i_vec.reshape((i_vec.size, 1)), j_vec.reshape((1, j_vec.size))))

        # Set up a persistant buffer for MFCCs so we don't waste memory
        self.mfcc_buffer = np.empty(kNumMFCCs, dtype=np.float64)

    # Receive audio data and send back a Numpy array of Mel-Frequency Cepstral Coefficients
    def extract_mfccs(self, audio_buffer):
        # Pre-emphasize signal
        emphasized_audio = lfilter(self.s_of, self.s_pe, audio_buffer)

        # Take real-optimized FFT of emphasized audio signal
        spectrum = np.fft.rfft(emphasized_audio, n=kBufferFFTBins)

        # Compute Mel-frequency Spectral Coefficients (MFSCs)
        abs_spectrum = np.asarray(map(abs, spectrum)).T
        energies = np.dot(kMelFilterBank, abs_spectrum)
        mfsc = np.maximum(np.multiply(-50.0, np.ones(kMelFilterBank.shape[0])),
                          np.log(energies))

        # Compute Mel-frequency Cepstral Coefficients (MFCCs)
        # mfcc[i] = sum_{j=1}^{23.0} (mfsc[j] * cos(pi * i / 23.0 * (j - 0.5))
        #         = C * msfc
        mfcc = np.dot(self.dct, mfsc.reshape((mfsc.shape[0], 1)))
        mfcc = mfcc.reshape((mfcc.size))

        return mfcc
