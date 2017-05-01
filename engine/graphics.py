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

# these are just convenient
kWindowWidth = (kNumGems * kGemWidth) + (2 * kThickness) + kLeftX + kRightX
kWindowHeight = (kNumPreviews+1)*(kGemHeight + 2*kThickness) + (kNumPreviews)*kMeasureSpacing + kBottomY + kTopY


# IMPORTS
# kivy config (needs to be done first)
from kivy.config import Config
# disable multi-touch emulation
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
# set size of window
Config.set('graphics', 'width', str(kWindowWidth))
Config.set('graphics', 'height', str(kWindowHeight))

from kivy.graphics.instructions import InstructionGroup
from kivy.graphics import Color, Ellipse, Line, Rectangle
from kivy.graphics import PushMatrix, PopMatrix, Translate, Scale, Rotate
from kivy.clock import Clock as kivyClock
from kivy.core.image import Image
from kivy.core.window import Window

# set background of screen to white
Window.clearcolor = (1, 1, 1, 1)

from common.gfxutil import *
import numpy as np


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
    def __init__(self, pos, gems, width, height):
        super(MeasureDisplay, self).__init__()

        #passed-in values of measure width and height
        self.width = width
        self.height = height

        #position variables for the measure
        self.moving = False
        self.current_pos = pos
        self.final_pos = pos
        self.speed = np.array([0, 0])

        self.gems = []
        for i in range(len(gems)):

            if gems[i] != None:
                x = self.current_pos[0] + float(i * self.width)/kNumGems
                y = self.current_pos[1] + kThickness

                gd = GemDisplay(pos=np.array([x, y]), size=(kGemWidth, kGemHeight), beat=gem[1])
                self.gems.append(gd)
                self.add(gd)
            else:
                self.gems.append(None)

        #measure no longer has a nowbar
        # self.nbd = NowbarDisplay(pos[0]+kThickness/2, pos[0]+w-kThickness/2, pos[1], pos[1]+h)
        # self.add(self.nbd)

    # no longer set_progress
    # def set_progress(self, progress):
    #     self.nbd.set_progress(progress)

    # hit gem (gem_idx is relative to start of measure)
    def gem_hit(self, gem_idx):
        self.gems[gem_idx].on_hit()

    # update measure position on screen
    def set_pos(self, pos):
        initial = self.current_pos
        final = pos

        self.speed = final - initial
        self.final_pos = final

        self.moving = True

    def set_width(self, width):
        self.width = width

    def set_height(self, height):
        self.height = height

    def on_update(self, dt):
        if self.current_pos == self.final_pos:
            self.moving = False
            
        if self.moving:
            self.current_pos += self.speed * dt

            if self.current_pos[1] > self.final_pos[1]:
                self.current_pos = self.final_pos

            for i in range(len(self.gems)):
                if self.gems[i] != None:
                    x = self.current_pos[0] + float(i * self.width)/kNumGems
                    y = self.current_pos[1] + kThickness
                    gem.set_pos(np.array([x,y]))





# Displays and controls all game elements: Nowbar, Buttons, BarLines, Gems.
class BeatMatchDisplay(InstructionGroup):
    def __init__(self, song_data, seek):
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


# HELPER FUNCTIONS
# returns a linear function f(x) given two points (x0, y0) and (x1, y1)
def linear(x0, y0, x1, y1):
    m = (y1-y0)*(x1-x0)**-1
    def f(x):
        return m*(x-x1)+y1
    return f
