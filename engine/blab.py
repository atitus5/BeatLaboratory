#gui.py

# CONSTANTS
# sreen padding
kLeftX = 20 # left screen padding
kRightX = kLeftX # right screen padding
kBottomY = 40 # bottom padding
kTopY = 60 # top padding

# graphics only
kGemWidth = 100
kGemHeight = kGemWidth # (leave square since we are using images)
kThickness = 5 # thickness of bar (>= 2)
kMeasureSpacing = 20 # vertical space between measures
# gem image filepaths
kImages = [
    '../data/kick.png',
    '../data/snare.png'
]

# graphics that indirectly affect gameplay
kNumGems = 8 # numbber of gems allowed per bar
kNumPreviews = 1 # number of measures ahead shown

# gameplay only
# kSlopWindow = .20 # amount of time gem can be hit early/late
kSlopWindow = .30 # amount of time gem can be hit early/late (more generous for mic)
kSnapFrac = kNumGems**-1 # if snap is true, tells to snap to nearest fraction of barline

# these are just convenient
kWindowWidth = (kNumGems * kGemWidth) + (2 * kThickness) + kLeftX + kRightX
kWindowHeight = (kNumPreviews+1)*(kGemHeight + 2*kThickness) + (kNumPreviews)*kMeasureSpacing + kBottomY + kTopY

# technical audio stuff
kNumChannels = 2

# IMPORTS
import sys
sys.path.append('..')

# kivy config (needs to be done first)
from kivy.config import Config
# disable multi-touch emulation
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
# set size of window
Config.set('graphics', 'width', str(kWindowWidth))
Config.set('graphics', 'height', str(kWindowHeight))

# common
from common.core import *
from common.audio import *
from common.clock import *
from common.mixer import *
from common.wavegen import *
from common.wavesrc import *
from common.gfxutil import *
from common.writer import *

# kivy
from kivy.graphics.instructions import InstructionGroup
from kivy.graphics import Color, Ellipse, Line, Rectangle
from kivy.graphics import PushMatrix, PopMatrix, Translate, Scale, Rotate
from kivy.clock import Clock as kivyClock
from kivy.core.image import Image
from kivy.core.window import Window
# set background of screen to white
Window.clearcolor = (1, 1, 1, 1)

# other
import numpy as np
from mic import *



# ARGUMENTS
# song_path = '../data/IWannaBeSedated' # could make command argument in future
song_path = '../data/BeatItNoDrums' # could make command argument in future
snapGems = True # snap gems to fraction of a barline.
seek = 0.0 # start x seconds into song (pass in via command line)


# MAIN WIDGET
class MainWidget(BaseWidget) :
    def __init__(self):
        super(MainWidget, self).__init__()
        # Set up audio input and output
        self.writer = AudioWriter('data') # for debugging audio output
        self.audio = Audio(kNumChannels, listen_func=self.writer.add_audio, input_func=self.process_mic_input)

        # game audio output
        self.mixer = Mixer()
        self.audio.set_generator(self.mixer)
        self.audio_ctrl = AudioController(song_path, self.mixer)
        self.song_data = SongData()
        self.song_data.read_data(song_path+'_gems.txt', song_path+'_barlines.txt')

        # Set up microphone input handling
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
        button_idx = lookup(keycode[1], '12', (0,1))
        if button_idx != None:
            self.player.on_button_down(button_idx)

    def process_mic_input(self, data, num_channels):
        # Send mic input to our handler
        event = self.mic_handler.add_data(data)
        if event:
            print event
        if event == 'kick':
            self.player.on_button_down(0)
        elif event == 'snare':
            self.player.on_button_down(1)

    def on_update(self) :
        dt = self.clock.get_time() - self.now
        self.now += dt
        self.player.on_update(dt)
        self.audio.on_update()
        self.score_label.text = 'score: ' + str(self.player.get_score())
        self.streak_label.text = str(self.player.get_streak()) + ' in a row'
        self.multiplier_label.text = 'x' + str(self.player.get_multiplier())


# AUDIO
# creates the Audio driver
# creates a song and loads it with solo and bg audio tracks
# creates snippets for audio sound fx
# TODO: Get rid of this since we wont need to mute/unmute solo track
class AudioController(object):
    def __init__(self, song_path, mixer):
        super(AudioController, self).__init__()

        self.bg = WaveGenerator(WaveFile(song_path + '_bg.wav'))
        # self.solo = WaveGenerator(WaveFile(song_path + '_solo.wav'))

        self.bg.pause()
        # self.solo.pause()

        self.bg.frame = int(seek * Audio.sample_rate)
        # self.solo.frame = int(seek * Audio.sample_rate)

        mixer.add(self.bg)
        # mixer.add(self.solo)

    # start / stop the song
    def toggle(self):
        self.bg.play_toggle()
        # self.solo.play_toggle()

    # mute / unmute the solo track
    def set_mute(self, mute):
        pass    # No solo track for B-Lab, yo
        '''
        if mute:
            self.solo.set_gain(0.0)
        else:
            self.solo.set_gain(1.0)
        '''


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
                self.gems.append((float(g[0]), (int(g[1][-1])-1) % 2))
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


# GRAPHICS
# LABEL FUNCTIONS
def title_label() :
    l = Label(text="Beat Laboratory",
        color=(0,0,0,1),
        halign='center',
        size=(kWindowWidth, kWindowHeight),
        text_size=(kWindowWidth, kWindowHeight),
        padding=(kRightX,kWindowHeight - kTopY*.75),
        font_size= .5*kTopY
        )
    return l
def botleft_label() :
    l = Label(text="text",
        color=(0,0,0,1),
        size=(kWindowWidth, kWindowHeight),
        text_size=(kWindowWidth, kWindowHeight),
        padding=(kLeftX,kBottomY*.25),
        font_size= .5*kBottomY
        )
    return l
def botright_label() :
    l = Label(text="text",
        color=(0,0,0,1),
        halign='right',
        size=(kWindowWidth, kWindowHeight),
        text_size=(kWindowWidth, kWindowHeight),
        padding=(kRightX,kBottomY*.25),
        font_size= .5*kBottomY
        )
    return l
def botmid_label() :
    l = Label(text="text",
        color=(0,0,0,1),
        halign='center',
        size=(kWindowWidth, kWindowHeight),
        text_size=(kWindowWidth, kWindowHeight),
        padding=(kRightX,kBottomY*.25),
        font_size= .5*kBottomY
        )
    return l


#KIVY INSTRUCTION GROUPS
# displays a circle with a color depending on beat type
class GemDisplay(InstructionGroup):
    def __init__(self, pos, size, beat):
        super(GemDisplay, self).__init__()
        self.color = Color(1,1,1,mode='rgb')
        #self.gem = Ellipse(pos=pos, size=size)
        path = kImages[beat]
        self.gem = Rectangle(pos=pos, size=size, texture=Image(path).texture)
        self.add(self.color)
        self.add(self.gem)

    # change the gem's position
    def set_pos(self, pos):
        self.gem.pos = pos

    # change to display this gem being hit
    def on_hit(self):
        self.color.rgba = (1,1,1,.25)


# a rectangle border
class BoxDisplay(InstructionGroup):
    def __init__(self, pos, size, thickness):
        super(BoxDisplay, self).__init__()
        self.add(Color(0,0,0,1, mode='rgba'))
        self.outer = Rectangle(pos=pos, size=size)
        self.add(self.outer)
        self.add(Color(1,1,1,1, mode='rgba'))
        w = size[0] - 2*thickness
        h = size[1] - 2*thickness
        x = pos[0] + thickness
        y = pos[1] + thickness
        self.thickness = thickness
        self.inner = Rectangle(pos=(x, y), size=(w,h))
        self.add(self.inner)

    # change the box's position
    def set_pos(self, pos):
        self.outer.pos = pos
        self.inner.pos = [pos[0] + self.thickness, pos[1] + self.thickness]


# a vertical line which can move linearly between two x positions
class NowbarDisplay(InstructionGroup):
    def __init__(self, x_start, x_end, y0, y1):
        super(NowbarDisplay, self).__init__()
        self.color = Color(0,0,0,.85,mode='rgba')
        self.line = Line(points=(x_start, y0, x_start, y1), width=kThickness/2, cap='none')
        self.trans = Translate(0,0)
        self.trans_i = Translate(0,0)
        self.add(self.trans)
        self.add(self.color)
        self.add(self.line)
        self.add(self.trans_i)
        self.x_fn = linear(0, x_start, 1, x_end) # function to move line

    # progress is in [0, 1) and specifies fraction of bar completed
    def set_progress(self, progress):
        if progress < 0:
            progress = 0
        if progress > 1:
            progress = 1
        newX = self.x_fn(progress)
        y0 = self.line.points[1]
        y1 = self.line.points[3]
        self.line.points = (newX, y0, newX, y1)

    # translate the bar from its original specified points
    def set_translate(self, xy):
        self.trans.xy = xy
        self.trans_i.xy = (-xy[0], -xy[1])


# Displays one bar
# pos is iterable (x, y)
# gems is iterable ((beat_num, beat_type), ...)
#   where 0 <= beat_num < kNumBeats specifies position
#   and beat_type specifies type of hit needed
class MeasureDisplay(InstructionGroup):
    def __init__(self, pos, gems):
        super(MeasureDisplay, self).__init__()
        w = kNumGems * kGemWidth + 2*kThickness
        h = kGemHeight + 2*kThickness
        self.box = BoxDisplay(pos=pos, size=(w, h), thickness=kThickness)
        self.add(self.box)

        self.gems = []
        for gem in gems:
            x = pos[0] + kThickness + gem[0]*kGemWidth
            y = pos[1] + kThickness
            gd = GemDisplay(pos=(x,y), size=(kGemWidth, kGemHeight), beat=gem[1])
            self.gems.append(gd)
            self.add(gd)

        self.nbd = NowbarDisplay(pos[0]+kThickness/2, pos[0]+w-kThickness/2, pos[1], pos[1]+h)
        self.add(self.nbd)

    # move nowbar
    def set_progress(self, progress):
        self.nbd.set_progress(progress)

    # hit gem (gem_idx is relative to start of measure)
    def gem_hit(self, gem_idx):
        self.gems[gem_idx].on_hit()

    # update measure position on screen
    def set_pos(self, pos):
        xy = (pos[0] - self.box.outer.pos[0], pos[1] - self.box.outer.pos[1])
        self.box.set_pos(pos)
        self.nbd.set_translate(xy)
        for gem in self.gems:
            x = gem.gem.pos[0] + xy[0]
            y = gem.gem.pos[1] + xy[1]
            gem.set_pos((x,y))


# Displays and controls all game elements: Nowbar, Buttons, BarLines, Gems.
class BeatMatchDisplay(InstructionGroup):
    def __init__(self, song_data):
        super(BeatMatchDisplay, self).__init__()

        # process song data to pre-generate bar graphics
        self.bars = []
        self.bar_durations = []
        self.bar_num_gems = []
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
            self.bar_num_gems.append(len(bar))
            self.bars.append(MeasureDisplay(pos=(kLeftX, kBottomY), gems=bar))
            i += 1

        # current bar (probably 0 unless seek is set)
        self.current_bar = 0
        while self.current_bar < len(self.bars) and seek > song_data.barlines[self.current_bar + 1]:
            self.current_bar += 1

        self.gem_offset = 0

        # time into bar (probably 0s unless seek is set)
        temp_l = filter(lambda b: b < seek, song_data.barlines)
        self.bar_dur = self.now - max(temp_l) if temp_l else 0.0

        # show intial graphics
        for i in range(min(kNumPreviews + 1, len(self.bars) - self.current_bar - 1)):
            y = kBottomY + (kNumPreviews - i) * (kGemHeight + 2*kThickness + kMeasureSpacing)
            self.bars[self.current_bar + i].set_pos((kLeftX, y))
            self.add(self.bars[self.current_bar + i])

    # stop displaying current measure, move preivews up, add next preview measure
    # updates gem_offset and current bar
    def __update_display(self):
        # remove finished measure
        self.remove(self.bars[self.current_bar])
        # update current measure
        self.gem_offset += self.bar_num_gems[self.current_bar]
        self.current_bar += 1
        # move preview measures up
        for i in range(min(kNumPreviews, len(self.bars) - self.current_bar - 1)):
            y = kBottomY + (kNumPreviews - i) * (kGemHeight + 2*kThickness + kMeasureSpacing)
            self.bars[self.current_bar + i].set_pos((kLeftX, y))
        # add new preview measure
        if self.current_bar + kNumPreviews < len(self.bars):
            self.add(self.bars[self.current_bar + kNumPreviews])

    # called by player, sends gem hit to correct measure
    def gem_hit(self, gem_idx):
        # this logic assumes gem hit will never be given more than +/- 1 bar early/late
        if 0 <= (gem_idx - self.gem_offset) < self.bar_num_gems[self.current_bar]:
            self.bars[self.current_bar].gem_hit(gem_idx - self.gem_offset)
        elif (gem_idx - self.gem_offset) < 0:
            self.bars[self.current_bar-1].gem_hit(gem_idx - (self.gem_offset - self.bar_num_gems[self.current_bar-1]))
        else:
            self.bars[self.current_bar+1].gem_hit(gem_idx - (self.gem_offset + self.bar_num_gems[self.current_bar]))

    # call every frame to move nowbar and check if measures need updating
    def on_update(self, dt):
        if self.current_bar >= len(self.bar_durations):
            return
        self.bar_dur += dt
        # we update measure half a beat early
        if self.bar_dur >= (2*kNumGems-1)*(2*kNumGems)**-1 * self.bar_durations[self.current_bar]:
            self.bar_dur -= self.bar_durations[self.current_bar]
            self.__update_display()

        progress = self.bar_dur * self.bar_durations[self.current_bar]**-1
        progress += (2*kNumGems)**-1 # nowbar goes in middle of gem on exact hit, not front
        # we changed progress from [0,1] to [(2*kNumGems)**-1, (2*kNumGems)**-1], so adjust
        if progress >= 1:
            progress -= 1
        # don't move nowbar during first measure since we add a dead measure at the beginning
        if self.current_bar == 0:
            progress = 0
        self.bars[self.current_bar].set_progress(progress)


# GAMEPLAY (not graphics)
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

            # check for lane (wrong beat) miss
            if abs(self.gem_data[self.next_gem][0] - self.now) < kSlopWindow:   
                    self.next_gem += 1

        # else temporal miss

        # on miss
        self.streak = 0
        self.mute = True

    # needed to check if for pass gems (ie, went past the slop window)
    def on_update(self, dt):
        self.now += dt
        # check for temporal miss
        while self.next_gem < len(self.gem_data) and self.gem_data[self.next_gem][0] < self.now - kSlopWindow:
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


# HELPER FUNCTIONS
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


# LET'S RUN THIS CODE!
if len(sys.argv) > 2:
    seek = float(sys.argv[2])

run(MainWidget)