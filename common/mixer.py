#####################################################################
#
# mixer.py
#
# Copyright (c) 2015, Eran Egozy
#
# Released under the MIT License (http://opensource.org/licenses/MIT)
#
#####################################################################

import numpy as np


# TODO - complete this class
class Mixer(object):
    def __init__(self):
        super(Mixer, self).__init__()

    def add(self, gen) :
        pass

    def remove(self, gen) :
        pass

    def set_gain(self, gain) :
        pass

    def get_gain(self) :
        return 0.5

    def get_num_generators(self) :
        return 0

    def generate(self, num_frames, num_channels) :
        output = np.zeros(num_frames * num_channels)
        return (output, True)
