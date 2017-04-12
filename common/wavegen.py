#####################################################################
#
# wavegen.py
#
# Copyright (c) 2016, Eran Egozy
#
# Released under the MIT License (http://opensource.org/licenses/MIT)
#
#####################################################################


import numpy as np

# generates audio data by asking an audio-source (ie, WaveFile) for that data.
class WaveGenerator(object):
    def __init__(self, wave_source, loop=False):
        super(WaveGenerator, self).__init__()
        self.source = wave_source
        self.frame = 0

    def reset(self):
        pass

    def play_toggle(self):
        pass

    def play(self):
        pass

    def pause(self):
        pass

    def release(self):
        pass

    def generate(self, num_frames, num_channels) :
        assert(num_channels == self.source.get_num_channels())

        # get data based on our position and requested # of frames
        output = self.source.get_frames(self.frame, self.frame + num_frames)

        # advance current-frame
        self.frame += num_frames

        # check for end-of-buffer condition:
        shortfall = num_frames * num_channels - len(output)
        continue_flag = shortfall == 0
        if shortfall > 0:
            output = np.append(output, np.zeros(shortfall))

        # return
        return (output, continue_flag)



# Create a generator that can modulate the speed of another generator
class SpeedModulator(object):
    def __init__(self, generator, speed = 1.0):
        super(SpeedModulator, self).__init__()

    def set_speed(self, speed) :
        pass

    def generate(self, num_frames, num_channels) :
        output = np.zeros(num_channels * num_frames)
        continue_flag = True
        return (output, continue_flag)

