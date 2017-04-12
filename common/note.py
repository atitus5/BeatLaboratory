#####################################################################
#
# note.py
#
# Copyright (c) 2015, Eran Egozy
#
# Released under the MIT License (http://opensource.org/licenses/MIT)
#
#####################################################################

import numpy as np
from audio import Audio


class NoteGenerator(object):
    def __init__(self, pitch, gain, duration, attack = 0.01, harmonics = (1.0,) ):
        super(NoteGenerator, self).__init__()

    def generate(self, num_frames, num_channels) :
        output = np.zeros(num_frames * num_channels)
        return (output, True)

