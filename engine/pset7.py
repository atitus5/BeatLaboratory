#pset7.py

# constants
kLeftX = 20 # left screen padding
kRightX = kLeftX # right screen padding
kBottomY = 40 # bottom padding
kTopY = 40 # top padding

kNumGems = 8 # numbber of gems allowed per bar
kNumPreviews = 1

kGemSize = 60 # height of gem
kThickness = 2 # thickness of bar
kMeasureSpacing = 20 # vertical space between measures

kWindowWidth = (kNumGems * kGemSize) + (2 * kThickness) + kLeftX + kRightX
kWindowHeight = (kNumPreviews+1)*(kGemSize + 2*kThickness) + (kNumPreviews)*kMeasureSpacing + kBottomY + kTopY

kSlopWindow = .20 # amount of time gem can be hit early/late
kSnapFrac = kNumGems**-1 # if snap is true, tells to snap to nearest fraction of barline
# could make kSnapFrac an argument in hypothetical future version

import sys
sys.path.append('..')

from kivy.config import Config
Config.set('graphics', 'width', str(kWindowWidth))
Config.set('graphics', 'height', str(kWindowHeight))

from common.core import *
from common.audio import *
from common.clock import *
from common.mixer import *
from common.wavegen import *
from common.wavesrc import *
from common.gfxutil import *

from kivy.graphics.instructions import InstructionGroup
from kivy.graphics import Color, Ellipse, Line, Rectangle
from kivy.graphics import PushMatrix, PopMatrix, Translate, Scale, Rotate
from kivy.clock import Clock as kivyClock

import random
import numpy as np
import bisect

# colors used for buttons + gems
kColors = [
Color(0,1,0,mode='rgb'),
Color(1,0,0,mode='rgb'),
Color(.5,.5,0,mode='rgb'),
Color(0,0,1,mode='rgb'),
Color(.5,0,.5,mode='rgb')
]

song_path = '../data/IWannaBeSedated' # could make command argument in future
snapGems = True # Snap gems to fraction of a barline.
seek = 0.0

# returns a linear function f(x) given two points (x0, y0) and (x1, y1)
def linear(x0, y0, x1, y1):
    m = (y1-y0)*(x1-x0)**-1
    def f(x):
        return m*(x-x1)+y1
    return f

# rounds number to the nearest number composable by n * multiple + offset for some integer n
def round_to_multiple(number, multiple, offset):
    n = number - offset
    quotient = int(n / multiple)
    remainder = n - (multiple * quotient)
    if remainder > (.5 * multiple):
        return (multiple * (quotient + 1)) + offset
    return (multiple * quotient) + offset


def botleft_label() :
    l = Label(text="text",
        size=(kWindowWidth, kWindowHeight),
        text_size=(kWindowWidth, kWindowHeight),
        padding=(kLeftX,kBottomY*.25)
        )
    return l
def botright_label() :
    l = Label(text="text",
        halign='right',
        size=(kWindowWidth, kWindowHeight),
        text_size=(kWindowWidth, kWindowHeight),
        padding=(kRightX,kBottomY*.25)
        )
    return l
def botmid_label() :
    l = Label(text="text",
        halign='center',
        size=(kWindowWidth, kWindowHeight),
        text_size=(kWindowWidth, kWindowHeight),
        padding=(kRightX,kBottomY*.25)
        )
    return l

class MainWidget(BaseWidget) :
    def __init__(self):
        super(MainWidget, self).__init__()

        # audio
        self.audio_ctrl = AudioController(song_path)
        self.song_data = SongData()
        self.song_data.read_data(song_path+'_gems.txt', song_path+'_barlines.txt')

        # game text
        self.score_label = botleft_label()
        self.add_widget(self.score_label)
        self.streak_label = botright_label()
        self.add_widget(self.streak_label)
        self.multiplier_label = botmid_label()
        self.add_widget(self.multiplier_label)

        # graphics
        self.bmd = BeatMatchDisplay(self.song_data)
        self.canvas.add(self.bmd)

        # timekeeping
        self.clock = Clock()
        self.clock.stop()
        self.now = seek
        self.clock.set_time(seek)

        # gameplay
        self.player = Player(self.song_data.gems, self.bmd, self.audio_ctrl)

    def on_key_down(self, keycode, modifiers):
        # play / pause toggle
        if keycode[1] == 'p':
            self.clock.toggle()
            self.audio_ctrl.toggle()

        # button down
        button_idx = lookup(keycode[1], '12345', (0,1,2,3,4))
        if button_idx != None:
            self.player.on_button_down(button_idx)

    def on_key_up(self, keycode):
        # button up
        button_idx = lookup(keycode[1], '12345', (0,1,2,3,4))
        if button_idx != None:
            self.player.on_button_up(button_idx)

    def on_update(self) :
        dt = self.clock.get_time() - self.now
        self.now += dt
        self.player.on_update(dt)
        self.audio_ctrl.on_update()
        self.score_label.text = 'score: ' + str(self.player.get_score())
        self.streak_label.text = str(self.player.get_streak()) + ' in a row'
        self.multiplier_label.text = 'x' + str(self.player.get_multiplier())


# creates the Audio driver
# creates a song and loads it with solo and bg audio tracks
# creates snippets for audio sound fx
class AudioController(object):
    def __init__(self, song_path):
        super(AudioController, self).__init__()
        self.audio = Audio(2)

        self.mixer = Mixer()
        self.audio.set_generator(self.mixer)

        self.bg = WaveGenerator(WaveFile(song_path + '_bg.wav'))
        self.solo = WaveGenerator(WaveFile(song_path + '_solo.wav'))

        self.bg.pause()
        self.solo.pause()

        self.bg.frame = int(seek * Audio.sample_rate)
        self.solo.frame = int(seek * Audio.sample_rate)

        self.mixer.add(self.bg)
        self.mixer.add(self.solo)

    # start / stop the song
    def toggle(self):
        self.bg.play_toggle()
        self.solo.play_toggle()

    # mute / unmute the solo track
    def set_mute(self, mute):
        if mute:
            self.solo.set_gain(0.0)
        else:
            self.solo.set_gain(1.0)

    # needed to update audio
    def on_update(self):
        self.audio.on_update()


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
                self.gems.append((float(g[0]), int(g[1][-1])-1))
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


# displays a circle with a color depending on beat type
class GemDisplay(InstructionGroup):
    def __init__(self, pos, size, beat):
        super(GemDisplay, self).__init__()
        color = kColors[beat]
        self.color = Color(rgba=color.rgba)
        self.gem = Ellipse(pos=pos, size=size)
        self.add(self.color)
        self.add(self.gem)

    # change to display this gem being hit
    def on_hit(self):
        self.color.rgba = (1,1,1,.5)

    # change to display a passed gem
    def on_pass(self):
        pass

    # useful if gem is to animate
    def on_update(self, dt):
        pass


# a rectangle border
class BoxDisplay(InstructionGroup):
    def __init__(self, pos, size, thickness):
        super(BoxDisplay, self).__init__()
        self.add(Color(1,1,1,1, mode='rgba'))
        self.add(Rectangle(pos=pos, size=size))
        self.add(Color(0,0,0,1, mode='rgba'))
        w = size[0] - 2*thickness
        h = size[1] - 2*thickness
        x = pos[0] + thickness
        y = pos[1] + thickness
        self.add(Rectangle(pos=(x, y), size=(w,h)))


# a vertical line which can move linearly between two x positions
class NowbarDisplay(InstructionGroup):
    def __init__(self, x_start, x_end, y0, y1):
        super(NowbarDisplay, self).__init__()
        self.color = Color(1,1,1,.85,mode='rgba')
        self.line = Line(points=(x_start, y0, x_start, y1))
        self.add(self.color)
        self.add(self.line)
        self.x_fn = linear(0, x_start, 1, x_end) # function to move line

    # progress is in [0, 1) and specifies fraction of bar completed
    def set_progress(self, progress):
        newX = self.x_fn(progress)
        y0 = self.line.points[1]
        y1 = self.line.points[3]
        self.line.points = (newX, y0, newX, y1)


# Displays one bar
class MeasureDisplay(InstructionGroup):
    def __init__(self, pos, gems):
        super(MeasureDisplay, self).__init__()
        w = kNumGems * kGemSize + 2*kThickness
        h = kGemSize + 2*kThickness
        self.add(BoxDisplay(pos=pos, size=(w, h), thickness=kThickness))

        self.gems = []
        for gem in gems:
            x = pos[0] + kThickness + gem[0]*kGemSize
            y = pos[1] + kThickness
            gd = GemDisplay(pos=(x,y), size=(kGemSize, kGemSize), beat=gem[1])
            self.gems.append(gd)
            self.add(gd)

        self.nbd = NowbarDisplay(pos[0], pos[0]+w, pos[1], pos[1]+h)
        self.add(self.nbd)

    # move nowbar
    def set_progress(self, progress):
        self.nbd.set_progress(progress)

    # hit gem (gem_idx is relative to start of measure)
    def gem_hit(self, gem_idx):
        self.gems[gem_idx].on_hit()


# Displays and controls all game elements: Nowbar, Buttons, BarLines, Gems.
class BeatMatchDisplay(InstructionGroup):
    def __init__(self, song_data):
        super(BeatMatchDisplay, self).__init__()

        self.md = None
        self.previews = []

        self.bars = []
        self.bar_durations = []
        i = 1
        j = 0
        while i < len(song_data.barlines):
            bar = []
            barlength = song_data.barlines[i] - song_data.barlines[i-1]
            self.bar_durations.append(barlength)
            while j < len(song_data.gems) and song_data.gems[j][0] < song_data.barlines[i]:
                frac = (song_data.gems[j][0] - song_data.barlines[i-1])
                new_idx = np.round(kNumGems * frac * barlength**-1)
                bar.append((new_idx, song_data.gems[j][1]))
                j += 1
            self.bars.append(bar)
            i += 1
        self.current_bar = 0
        self.gem_offset = 0

        temp_l = filter(lambda b: b < seek, song_data.barlines)
        self.bar_dur = self.now - max(temp_l) if temp_l else 0.0

        self.__update_display()

    def __update_display(self):
        if self.md:
            self.remove(self.md)
        for pd in self.previews:
            self.remove(pd)
        self.previews = []
        y = kBottomY + kNumPreviews * (kGemSize + 2*kThickness + kMeasureSpacing)
        self.md = MeasureDisplay(pos=(kLeftX, + 150), gems=self.bars[self.current_bar])
        self.add(self.md)

        for i in range(min(kNumPreviews, len(self.bars)-self.current_bar-1)):
            y = kBottomY + (kNumPreviews - i - 1) * (kGemSize + 2*kThickness + kMeasureSpacing)
            pd = MeasureDisplay(pos=(kLeftX,y), gems=self.bars[self.current_bar + 1])
            self.previews.append(pd)
            self.add(pd)

    # called by Player. Causes the right thing to happen
    def gem_hit(self, gem_idx):
        if gem_idx - self.gem_offset < len(self.bars[self.current_bar]):
            self.md.gem_hit(gem_idx - self.gem_offset)

    # called by Player. Causes the right thing to happen
    def gem_pass(self, gem_idx):
        pass

    # call every frame to make gems and barlines flow down the screen
    def on_update(self, dt):
        self.bar_dur += dt
        if self.bar_dur >= self.bar_durations[self.current_bar]:
            self.bar_dur -= self.bar_durations[self.current_bar]
            self.gem_offset += len(self.bars[self.current_bar])
            self.current_bar += 1
            self.__update_display()
        self.md.set_progress(self.bar_dur * self.bar_durations[self.current_bar]**-1)


# Handles game logic and keeps score.
# Controls the display and the audio
class Player(object):
    def __init__(self, gem_data, display, audio_ctrl):
        super(Player, self).__init__()
        self.gem_data = gem_data
        self.display = display
        self.audio_ctrl = audio_ctrl
        self.next_gem = 0
        self.now = seek
        self.score = 0
        self.streak = 0
        self.mute = False

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
                        self.score += 1 * min(4, 1 + self.streak/5)
                        self.mute = False
                        return
                else:
                    break

            # check for lane miss
            if abs(self.gem_data[self.next_gem][0] - self.now) < kSlopWindow:   
                    self.display.gem_pass(self.next_gem)
                    self.next_gem += 1

        # else temporal miss

        # on miss
        self.streak = 0
        self.mute = True


    # called by MainWidget
    def on_button_up(self, lane):
        pass

    # needed to check if for pass gems (ie, went past the slop window)
    def on_update(self, dt):
        self.now += dt
        # check for temporal miss
        while self.next_gem < len(self.gem_data) and self.gem_data[self.next_gem][0] < self.now - kSlopWindow:
            self.display.gem_pass(self.next_gem)
            self.next_gem += 1
            self.streak = 0
            self.mute = True
        self.audio_ctrl.set_mute(self.mute)
        self.display.on_update(dt)

    def get_score(self):
        return self.score

    def get_streak(self):
        return self.streak

    def get_multiplier(self):
        return min(4, 1 + self.streak/5)

if len(sys.argv) > 2:
    seek = float(sys.argv[2])

run(MainWidget)