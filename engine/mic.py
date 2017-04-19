#mic.py


import math

import numpy as np
import scipy as sp

import sys
sys.path.append('..')

from common.audio import *


# Speech recognition standard sizes, as suggested by literature on human vocal tract traits
kBufferSize = kSampleRate * 0.025       # 25 ms
kBufferSpacing = kSampleRate * 0.010    # 10 ms
kBufferCount = int(math.ceil(kBufferSize / kBufferSpacing))

# Used to handle streaming audio input data and return events corresponding
# to beatbox events
# NOTE: currently only accepts stereo audio (i.e. num_channels = 2)
class MicrophoneHandler(object) :
    def __init__(self, num_channels):
        super(MicrophoneHandler, self).__init__()

        assert(num_channels == 2, "MicrophoneHandler currently only supports stereo audio")

        self.buffers = np.zeros((kBufferCount, kBufferSize), dtype=np.float32)
        self.buffer_indices = np.zeros(kBufferCount, dtype=np.int)

    # Receive data and send back a string indicating the event that occurred or None if none occurred
    def add_data(self, data):
        self.update_buffers(data)

        # TODO: check for a completed buffer and process it for events

    # Update buffers in streaming fashion
    def update_buffers(self, data):
        # TODO 
