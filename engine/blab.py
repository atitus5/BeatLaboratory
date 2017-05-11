#blab.py

# IMPORTS
import sys
sys.path.append('..')

# command line args
kSeek = 0.0
if len(sys.argv) >= 2:
    kSeek = float(sys.argv[1])
kUseMic = True
kRecord = False
kDefaultModel = False
if len(sys.argv) >= 3:
    kUseMic = not sys.argv[2] == 'nomic'
    kRecord = sys.argv[2] == 'record'
    kDefaultModel = sys.argv[2] == 'defaultmodel'

# other game files
from graphics import *
if kUseMic:
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
import time

# CONSTANTS
# gameplay
kSlopWindow = .100  # amount of time gem can be hit early/late
kSnapFrac = kNumGems**-1 # if snap is true, tells to snap to nearest fraction of barline
snapGems = True # snap gems to fraction of a barline.

# audio
#song_path = '../data/BeatItNoDrums' # could make command argument in future
#song_path = '../data/24KMagicNoDrums' # could make command argument in future
song_path = '../data/funk' # could make command argument in future
train_song_path = '../data/training' # could make command argument in future


# MAIN WIDGET
class MainWidget(BaseWidget) :
    def __init__(self):
        super(MainWidget, self).__init__()
        # Set up audio input and output
        self.writer = AudioWriter('data') # for debugging audio output
        self.music_audio = MusicAudio(2)

        if kUseMic:
            self.mic_audio = MicAudio(1, self.writer.add_audio, self.process_mic_input)

        # game audio output
        self.mixer = Mixer()
        self.music_audio.set_generator(self.mixer)
        self.bg = WaveGenerator(WaveFile(song_path + '_bg.wav'))
        self.bg.pause()
        self.bg.frame = int(kSeek * Audio.sample_rate)
        self.mixer.add(self.bg)
        self.song_data = SongData()
        self.song_data.read_data(song_path+'_gems.txt', song_path+'_barlines.txt')

        # training song data
        self.train_song_data = SongData()
        self.train_song_data.read_data(train_song_path+'_gems.txt', train_song_path+'_barlines.txt')
        self.training = True

        # game text
        self.score_label = topleft_label()
        self.add_widget(self.score_label)
        self.multiplier_streak_label = topright_label()
        self.add_widget(self.multiplier_streak_label)
        # static title label
        self.add_widget(title_label())

        self.player = None
        self.bmd = None
        self.training = True

        # timekeeping
        self.clock = Clock()
        self.clock.stop()
        self.now = 0

        # Set up microphone input handling
        if kUseMic:
            # Set up microphone input handling and training
            slop_frames = int(kSlopWindow * kSampleRate)
            self.mic_handler = MicrophoneHandler(1, slop_frames)

            if not kDefaultModel:
                # Need to train a classifier first
                self.training = True
                self.current_label = None   # Will be set before it is used, no problemos
                self.train_popup = Popup(title="Welcome to BeatLaboratory!",
                                        content=Label(text="Welcome to BeatLaboratory! Because everyone has\n" +
                                                      "their own beatboxing style, we need to learn more\n" +
                                                      "about yours by hearing you! Make sure your microphone\n" +
                                                      "is set up, and play along with this simple track!\n" +
                                                      "\nSimply press anywhere outside of this box to begin."),
                                        size_hint=(None, None),
                                        size=(400,400)) 
                self.train_popup.bind(on_dismiss=self.start_training)
                self.train_popup.open()
            else:
                # Pre-trained classifier already loaded - let's roll!
                self.stop_training(None)
        else:
            # No classifier to train - just go!
            self.stop_training(None)

    def start_training(self, instance):
        self.train_popup = None 

        if self.bmd is not None:
            self.canvas.remove(self.bmd)

        # graphics
        self.bmd = BeatMatchDisplay(self.train_song_data, 0)
        self.canvas.add(self.bmd)
        self.bmd.install_particle_systems(self)

        # gameplay
        self.player = Player(self.train_song_data.gems, self.bmd, 0)

        # Start the training track
        self.clock.start()

    def stop_training(self, instance):
        self.train_popup = None 

        # Fix clock
        self.clock.stop()
        self.now = kSeek
        self.clock.set_time(kSeek)

        if self.bmd is not None:
            self.canvas.remove(self.bmd)

        self.training = False

        # graphics
        self.bmd = BeatMatchDisplay(self.song_data, kSeek)
        self.canvas.add(self.bmd)
        self.bmd.install_particle_systems(self)

        # gameplay
        self.player = Player(self.song_data.gems, self.bmd, kSeek)


    def on_key_down(self, keycode, modifiers):
        if not self.training:
            # play / pause toggle
            if keycode[1] == 'p':
                self.clock.toggle()
                self.bg.play_toggle()
                if kRecord:
                    self.writer.toggle()

        if not kUseMic:
            # button down
            button_idx = lookup(keycode[1], '123', (0,1,2))
            if button_idx != None:
                self.player.on_button_down(button_idx)

    def process_mic_input(self, data, num_channels):
        if self.mic_handler.processing_audio:
            # Send mic input to our handler
            if self.training:
                self.mic_handler.add_training_data(data, self.current_label)

                if not self.mic_handler.processing_audio:
                    # Send a "no-event" event so that next_gem is updated properly
                    self.player.on_event(kNoEvent)
            else:
                start_t = time.time()
                label = self.mic_handler.add_data(data, self.current_label)
                event = kLabelToEvent[label]
                if event is not kNoEvent:
                    # print label
                    self.player.on_event(event)

    def on_update(self) :
        if self.bg.paused:
            # Run our own clock
            t = self.clock.get_time()
        else:
            # Sync with the music
            self.music_audio.on_update()
            t = self.bg.frame / float(kSampleRate)
            self.clock.set_time(t)
        dt = t - self.now
        self.now += dt

        if self.player is not None:
            self.player.on_update(dt)
            self.score_label.text = 'score: ' + str(self.player.get_score())
            self.multiplier_streak_label.text = 'x' + str(self.player.get_multiplier()) + ' (' + str(self.player.get_streak()) + ' in a row)'
        else:
            self.score_label.text = ''
            self.multiplier_streak_label.text = ''

        if kUseMic and self.player is not None:
            if self.training:
                if self.player.next_gem >= len(self.player.gem_data):
                    if self.train_popup is None:
                        # We're done! Train the classifier, then display another popup
                        self.mic_handler.train_classifier()

                        self.train_popup = Popup(title="You're ready to go!",
                                                content=Label(text="You're all ready to play!\n" + 
                                                              "\nSimply press anywhere outside of this box to begin."),
                                                size_hint=(None, None),
                                                size=(400,400)) 
                        self.train_popup.bind(on_dismiss=self.stop_training)
                        self.train_popup.open()
                    return

            if self.training or not self.bg.paused:
                process_audio = self.mic_handler.processing_audio

                # Only start processing audio if we have a gem within its slop window
                # NOTE: this assumes our slop window is small enough to not spill into
                # the slop windows of neighboring gems
                if not process_audio:
                    gems_active = self.player.next_gem < len(self.player.gem_data)
                    if gems_active:
                        time_gap = self.player.gem_data[self.player.next_gem][0] - self.player.now
                        gem_in_window = abs(time_gap) <= kSlopWindow
                        if gem_in_window:
                            # We're ready to gooooo
                            self.mic_handler.processing_audio = True
                            self.current_label = self.player.gem_data[self.player.next_gem][1]
                            frame_start = int((self.player.now - self.player.gem_data[self.player.next_gem][0] + kSlopWindow) * kSampleRate)
                            self.mic_handler.buf_idx = frame_start

                self.mic_audio.on_update()


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
                self.gems.append((float(g[0]), (int(g[1])-1)))
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
    def __init__(self, gem_data, display, seek):
        super(Player, self).__init__()
        self.gem_data = gem_data
        self.display = display
        self.next_gem = 0
        self.now = seek
        self.score = 0
        self.streak = 0
        self.display.update_ps(self.get_multiplier())

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
                        self.next_gem += 1
                        self.streak += 1
                        self.score += 1 * min(kMaxMultiplier, 1 + self.streak/5)
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

    # called by MainWidget
    def on_event(self, lane):
        if self.next_gem < len(self.gem_data):
            # check for hit
            if self.gem_data[self.next_gem][1] == lane:
                self.display.gem_hit(self.next_gem)
                self.streak += 1
                self.score += 1 * min(kMaxMultiplier, 1 + self.streak/5)
            else:
                self.display.gem_miss(self.next_gem)
                self.streak = 0
            self.next_gem += 1

    # needed to check if for pass gems (ie, went past the slop window)
    def on_update(self, dt):
        self.now += dt

        # There is no concept of a temporal miss with mic input, since we always
        # check the gem when it's done being processed!
        if not kUseMic:
            # check for temporal miss
            while self.next_gem < len(self.gem_data) and self.gem_data[self.next_gem][0] < self.now - kSlopWindow:
                self.display.gem_miss(self.next_gem)
                self.next_gem += 1
                self.streak = 0

        self.display.on_update(dt)
        self.display.update_ps(self.get_multiplier())

    def get_score(self):
        return self.score

    def get_streak(self):
        return self.streak

    def get_multiplier(self):
        return min(kMaxMultiplier, 1 + self.streak/5)


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
