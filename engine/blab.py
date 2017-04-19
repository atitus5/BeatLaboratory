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

kNumChannels = 2

class MainWidget(BaseWidget) :
    def __init__(self):
        super(MainWidget, self).__init__()

        # Set up audio input and output
        self.writer = AudioWriter('data') # for debugging audio output
        self.audio = Audio(kNumChannels, listen_func=self.writer.add_audio, input_func=self.process_mic_input)

        # Set up microphone input handling
        self.mic_handler = MicrophoneHandler(kNumChannels)

    def process_mic_input(self, data, num_channels):
        # Send mic input to our handler
        event = self.mic_handler.add_data(data)
        if event is not None:
            print event
            # TODO: process event as input

    def on_key_down(self, keycode, modifiers):
        pass

    def on_key_up(self, keycode):
        pass

    def on_update(self) :
        self.audio.on_update()



run(MainWidget)
