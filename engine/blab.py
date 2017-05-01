#gui.py

# IMPORTS
import sys
sys.path.append('..')

# command line args
seek = 0.0
if len(sys.argv) >= 2:
    seek = float(sys.argv[1])
usemic = True
record = False
if len(sys.argv) >= 3:
    usemic = not sys.argv[2] == 'nomic'
    record = sys.argv[2] == 'record'

# other game files
from graphics import *
if usemic:
    from mic import *

# common
from common.core import *
from common.audio import *
from common.clock import *
from common.mixer import *
from common.wavegen import *
from common.wavesrc import *
from common.writer import *

# other
import numpy as np

# CONSTANTS
# gameplay
kSlopWindow = .20 # amount of time gem can be hit early/late (more generous for mic)
kSnapFrac = kNumGems**-1 # if snap is true, tells to snap to nearest fraction of barline
snapGems = True # snap gems to fraction of a barline.

# audio
kNumChannels = 2
song_path = '../data/BeatItNoDrums' # could make command argument in future


# MAIN WIDGET
class MainWidget(BaseWidget) :
    def __init__(self):
        super(MainWidget, self).__init__()
        # Set up audio input and output
        self.writer = AudioWriter('data') # for debugging audio output
        self.music_audio = MusicAudio(kNumChannels)

        if usemic:
            self.mic_audio = MicAudio(kNumChannels, self.writer.add_audio, self.process_mic_input)

        # game audio output
        self.mixer = Mixer()
        self.music_audio.set_generator(self.mixer)
        self.bg = WaveGenerator(WaveFile(song_path + '_bg.wav'))
        self.bg.pause()
        self.bg.frame = int(seek * Audio.sample_rate)
        self.mixer.add(self.bg)
        self.song_data = SongData()
        self.song_data.read_data(song_path+'_gems.txt', song_path+'_barlines.txt')

        # Set up microphone input handling
        if usemic:
            self.mic_handler = MicrophoneHandler(kNumChannels)

        # game text
        self.score_label = botleft_label()
        self.add_widget(self.score_label)
        self.streak_label = botright_label()
        self.add_widget(self.streak_label)
        self.multiplier_label = botmid_label()
        self.add_widget(self.multiplier_label)
        # static title label
        self.add_widget(title_label())

        # graphics
        self.bmd = BeatMatchDisplay(self.song_data, seek)
        self.canvas.add(self.bmd)

        # timekeeping
        self.clock = Clock()
        self.clock.stop()
        self.now = seek
        self.clock.set_time(seek)

        # gameplay
        self.player = Player(self.song_data.gems, self.bmd)

    def on_key_down(self, keycode, modifiers):
        # play / pause toggle
        if keycode[1] == 'p':
            self.clock.toggle()
            self.bg.play_toggle()
            if record:
                self.writer.toggle()

        # button down
        button_idx = lookup(keycode[1], '123', (0,1,2))
        if button_idx != None:
            self.player.on_button_down(button_idx)

    def process_mic_input(self, data, num_channels):
        if not usemic:
            return
        # Send mic input to our handler
        event = self.mic_handler.add_data(data)
        if event == 'kick':
            self.player.on_button_down(0)
        elif event == 'hihat':
            self.player.on_button_down(1)
        elif event == 'snare':
            self.player.on_button_down(2)

    def on_update(self) :
        dt = self.clock.get_time() - self.now
        self.now += dt
        self.player.on_update(dt)
        self.music_audio.on_update()
        if usemic:
            self.mic_audio.on_update()
        self.score_label.text = 'score: ' + str(self.player.get_score())
        self.streak_label.text = str(self.player.get_streak()) + ' in a row'
        self.multiplier_label.text = 'x' + str(self.player.get_multiplier())


# PARSE DATA (gems & barlines)
# holds data for gems and barlines.
class SongData(object):
    def __init__(self):
        super(SongData, self).__init__()
        self.gems = []
        self.barlines = []

    # read the gems and song data. You may want to add a secondary filepath
    # argument if your barline data is stored in a different txt file.
    def read_data(self, gems_filepath, barlines_filepath):
        # read gem file
        with open(gems_filepath) as f:
            lines = f.readlines()
            gems_data = map(lambda l: l.strip().split('\t'), lines)
            gems_data.sort()
            # handle multiple button gems (make a gem for each one)
            for g in gems_data:
                self.gems.append((float(g[0]), (int(g[1][-1])-1) % len(kImages)))
            self.gems.sort()

        # read barline file
        with open(barlines_filepath) as f:
            lines = f.readlines()
            barlines_data = map(lambda l: l.strip().split('\t'), lines)
            self.barlines = map(lambda b: float(b[0]), barlines_data)
            self.barlines = [0.0] + self.barlines
            self.barlines.sort()

        # optionally snap gems to fraction of bar
        if snapGems:
            real_gems = []
            next_bar = 1
            for gem in self.gems:
                while self.barlines[next_bar] < gem[0]:
                    next_bar += 1
                dt = self.barlines[next_bar] - self.barlines[next_bar-1]
                t = round_to_multiple(gem[0], dt * kSnapFrac, self.barlines[next_bar-1])
                real_gems.append((t, gem[1]))
            self.gems = real_gems

        # no duplicates
        self.gems = sorted(list(set(self.gems)))
        self.barlines = sorted(list(set(self.barlines)))


# GAMEPLAY
# Handles game logic and keeps score.
# Controls the display and the audio
class Player(object):
    def __init__(self, gem_data, display):
        super(Player, self).__init__()
        self.gem_data = gem_data
        self.display = display
        self.next_gem = 0
        self.now = seek
        self.score = 0
        self.streak = 0

        # skip ahead in case of seeks
        while self.next_gem < len(self.gem_data) and self.gem_data[self.next_gem][0] < self.now - kSlopWindow:
            self.next_gem += 1

    # called by MainWidget
    def on_button_down(self, lane):
        if self.next_gem < len(self.gem_data):
            # check for hit
            for i in range(self.next_gem, len(self.gem_data)):
                if abs(self.gem_data[i][0] - self.now) < kSlopWindow:
                    if self.gem_data[i][1] == lane:
                        self.display.gem_hit(i)
                        print 'hit', i
                        self.next_gem += 1
                        self.streak += 1
                        self.score += 1 * min(4, 1 + self.streak/5)
                        return
                else:
                    break

            # check for lane (wrong beat) miss
            if abs(self.gem_data[self.next_gem][0] - self.now) < kSlopWindow:
                self.display.gem_miss(self.next_gem)
                self.next_gem += 1

        # else temporal miss

        # on miss
        self.streak = 0

    # needed to check if for pass gems (ie, went past the slop window)
    def on_update(self, dt):
        self.now += dt
        # check for temporal miss
        while self.next_gem < len(self.gem_data) and self.gem_data[self.next_gem][0] < self.now - kSlopWindow:
            self.display.gem_miss(self.next_gem)
            self.next_gem += 1
            self.streak = 0
        self.display.on_update(dt)

    def get_score(self):
        return self.score

    def get_streak(self):
        return self.streak

    def get_multiplier(self):
        return min(4, 1 + self.streak/5)


# HELPER FUNCTIONS

# rounds number to the nearest number composable by n * multiple + offset for some integer n
def round_to_multiple(number, multiple, offset):
    n = number - offset
    quotient = int(n / multiple)
    remainder = n - (multiple * quotient)
    if remainder > (.5 * multiple):
        return (multiple * (quotient + 1)) + offset
    return (multiple * quotient) + offset


# LET'S RUN THIS CODE!
run(MainWidget)
