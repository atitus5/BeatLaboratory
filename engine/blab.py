#blab.py


import sys
sys.path.append('..')

from common.audio import *
from common.core import *
from common.gfxutil import *
from common.mixer import *
from common.wavegen import *
from common.wavesrc import *
from common.writer import *

import numpy as np

class MainWidget(BaseWidget) :
    def __init__(self):
        super(MainWidget, self).__init__()

        # Track mic input
        self.mic = Audio(2, input_func=self.process_mic)
        self.volume_circle = CEllipse(cpos=(Window.size[0] / 2, Window.size[0] / 2),
                                      csize=(0, 0))
        self.canvas.add(self.volume_circle)

    def process_mic(self, data, num_channels):
        volume_avg = sum(data) / float(len(data))
        new_radius = volume_avg * Window.size[1] / 2.0
        self.volume_circle.csize = (2 * new_radius, 2 * new_radius)


    def on_key_down(self, keycode, modifiers):
        pass

    def on_key_up(self, keycode):
        pass

    def on_update(self) :
        self.mic.on_update()



run(MainWidget)
