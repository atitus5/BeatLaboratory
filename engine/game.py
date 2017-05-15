# game.py

from common.core import *
from common.audio import *
from common.clock import *
from common.mixer import *
from common.wavegen import *
from common.wavesrc import *
from common.writer import *

from graphics import *
from mic import *

# other
import numpy as np
import time

# gameplay
kSlopWindow = .100  # amount of time gem can be hit early/late
kSnapFrac = kNumGems**-1 # if snap is true, tells to snap to nearest fraction of barline
snapGems = True # snap gems to fraction of a barline.


# mic input (same object every game)
slop_frames = int(kSlopWindow * kSampleRate)
mic_handler = MicrophoneHandler(1, slop_frames)

class GameWidget(Widget):
    def __init__(self, song_path, train=False, seek=0, record=False):
        super(GameWidget, self).__init__()
        self.train = train
        self.record = record

        # Set up audio input and output
        self.writer = AudioWriter('data') # for debugging audio output
        self.music_audio = MusicAudio(2)
        self.mic_audio = MicAudio(1, self.writer.add_audio, self.process_mic_input)

        # game audio output
        self.mixer = Mixer()
        self.music_audio.set_generator(self.mixer)
        self.bg = WaveGenerator(WaveFile(song_path + '_bg.wav'))
        self.bg.pause()
        self.bg.frame = int(seek * Audio.sample_rate)
        self.mixer.add(self.bg)
        self.song_data = SongData(song_path+'_gems.txt', song_path+'_barlines.txt')

        self.mic_handler = mic_handler

        # game text
        if not self.train:
            self.score_label = topleft_label()
            self.add_widget(self.score_label)
            self.multiplier_streak_label = topright_label()
            self.add_widget(self.multiplier_streak_label)
        self.add_widget(title_label())

        # graphics
        self.bmd = BeatMatchDisplay(self.song_data, seek)
        self.canvas.add(self.bmd)
        if not self.train:
            self.bmd.install_particle_systems(self)

        # gameplay
        self.player = Player(self.song_data, self.bmd, seek)

        # timekeeping
        self.clock = Clock()
        self.clock.stop()
        self.now = seek
        self.clock.set_time(seek)

    def on_key_down(self, keycode, modifiers):
        if keycode[1] == 'p':
            self.clock.toggle()
            self.bg.play_toggle()
            if self.record:
                self.writer.toggle()

        '''
        if keycode[1] == 'e':
            self.player.display.explode()
        '''

    def process_mic_input(self, data, num_channels):
        if self.mic_handler.processing_audio:
            # Send mic input to our handler
            if self.train:
                self.mic_handler.add_training_data(data, self.current_label)
                if not self.mic_handler.processing_audio:
                    # Send a "no-event" event so that next_gem is updated properly
                    self.player.on_event(kNoEvent)
            else:
                start_t = time.time()
                label = self.mic_handler.add_data(data, self.current_label)
                event = kLabelToEvent[label]
                if event is not kNoEvent:
                    self.player.on_event(event)

    def get_score(self):
        return self.player.get_score()

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

        self.player.on_update(dt)
        if not self.train:
            self.score_label.text = 'score: ' + str(self.player.get_score())
            self.multiplier_streak_label.text = 'x' + str(self.player.get_multiplier()) + ' (' + str(self.player.get_streak()) + ' in a row)'

        # end of song
        if self.now >= self.song_data.barlines[-1]:
            if self.train:
                self.mic_handler.train_classifier()
            return False

        elif not self.bg.paused:
            process_audio = self.mic_handler.processing_audio

            # Only start processing audio if we have a gem within its slop window
            # this assumes our slop window is small enough to not spill into
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

        return True


# PARSE DATA (gems & barlines)
# holds data for gems and barlines.
class SongData(object):
    def __init__(self, gems_filepath, barlines_filepath):
        super(SongData, self).__init__()

        # read gem file
        gems = []
        with open(gems_filepath) as f:
            lines = f.readlines()
            gems_data = map(lambda l: l.strip().split('\t'), lines)
            gems_data.sort()
            # handle multiple button gems (make a gem for each one)
            for g in gems_data:
                gems.append((float(g[0]), (int(g[1])-1)))
            gems.sort()

        # read barline file
        with open(barlines_filepath) as f:
            lines = f.readlines()
            barlines_data = map(lambda l: l.strip().split('\t'), lines)
            barlines = map(lambda b: float(b[0]), barlines_data)
            barlines = [0.0] + barlines
            barlines.sort()

        # optionally snap gems to fraction of bar
        if snapGems:
            real_gems = []
            next_bar = 1
            for gem in gems:
                while barlines[next_bar] < gem[0]:
                    next_bar += 1
                dt = barlines[next_bar] - barlines[next_bar-1]
                t = round_to_multiple(gem[0], dt * kSnapFrac, barlines[next_bar-1])
                real_gems.append((t, gem[1]))
            gems = real_gems

        # no duplicates
        self.gems = sorted(list(set(gems)))
        self.barlines = sorted(list(set(barlines)))


# GAMEPLAY
# Handles game logic and keeps score.
# Controls the display and the audio
class Player(object):
    def __init__(self, song_data, display, seek):
        super(Player, self).__init__()
        self.gem_data = song_data.gems
        self.bar_data = song_data.barlines
        self.display = display
        self.next_gem = 0
        self.next_bar = 0
        self.now = seek
        self.score = 0
        self.streak = 0
        self.display.update_ps(self.get_multiplier())
        self.bonus = False

        # skip ahead in case of seeks
        while self.next_gem < len(self.gem_data)-1 and self.gem_data[self.next_gem][0] < self.now - kSlopWindow:
            self.next_gem += 1
        while self.next_bar < len(self.bar_data)-1 and self.bar_data[self.next_bar] <= self.gem_data[self.next_gem][0]:
            self.next_bar += 1

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
                self.bonus = False # no bonus if you miss before end of bar
                self.streak = 0
            if self.next_gem < len(self.gem_data)-1:
                self.next_gem += 1

    # clears the next n bars
    def __use_bonus(self):
        self.next_bar = min(len(self.bar_data)-1, self.next_bar+1)
        self.display.explode()
        while self.next_gem < len(self.gem_data)-1 and self.gem_data[self.next_gem][0] < self.bar_data[self.next_bar]:
            self.display.gem_hit(self.next_gem)
            self.score += 1 * min(kMaxMultiplier, 1 + self.streak/5)
            self.next_gem += 1
        self.bonus = False

    # needed to check if for pass gems (ie, went past the slop window)
    def on_update(self, dt):
        self.now += dt
        self.display.on_update(dt)
        self.display.update_ps(self.get_multiplier())
        condition = self.streak > 0 and (self.streak % 25) == 0
        self.bonus = (self.bonus or condition)
        if self.next_bar < len(self.bar_data)-1 and self.bar_data[self.next_bar] <= self.gem_data[self.next_gem][0]:
            self.next_bar += 1
            if self.bonus:
                self.__use_bonus()

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