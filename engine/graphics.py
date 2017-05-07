# CONSTANTS
# sreen padding
kLeftX = 20 # left screen padding
kRightX = kLeftX # right screen padding
kBottomY = 40 # bottom padding
kTopY = 60 # top padding

# graphics only
# kGemWindowWidth = 150
kGemWindowWidth = 150
kGemWindowHeight = kGemWindowWidth  # (leave square since we are using images)
kGemWidth = 100
kGemHeight = kGemWidth
kBoxThickness = 5 # thickness of bar (>= 2)
kMeasureSpacing = 20 # vertical space between measures
# gem image filepaths
kImages = [
    '../data/kick.png',
    '../data/hihat.png',
    '../data/snare.png'
]
kBeatlineThickness = 2
kNowbarThickness = 3

# graphics that indirectly affect gameplay
kNumGems = 8 # numbber of gems allowed per bar
kNumPreviews = 2 # number of measures ahead shown

# these are just convenient
kWindowWidth = (kNumGems * kGemWindowWidth) + (2 * kBoxThickness) + kLeftX + kRightX
kWindowHeight = (kNumPreviews+1)*(kGemWindowHeight + 2*kBoxThickness) + (kNumPreviews)*kMeasureSpacing + kBottomY + kTopY


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

# set background of screen to dark dark blue
kBgColor = (13 / 256.0, 26 / 256.0, 38 / 256.0, 0.8)
Window.clearcolor = kBgColor

kTitleColor = (51 / 256.0, 153 / 256.0, 51 / 256.0, 0.9)  # Green
kTextColor = (242 / 256.0, 242 / 256.0, 242 / 256.0, 0.9)  # Slightly off-white
kFontPath = "../data/CevicheOne-Regular.ttf"

kBeatlineColor = (1,1,1,.5)

kTitleFontSize = 0.9 * kTopY
kTopFontSize = 0.6 * kTopY

from common.gfxutil import *
import numpy as np


# GRAPHICS
# LABEL FUNCTIONS
def title_label() :
    l = Label(text="Beat Laboratory",
        # color=(0,0,0,1),
        color=kTitleColor,
        halign='center',
        size=(kWindowWidth, kWindowHeight),
        text_size=(kWindowWidth, kWindowHeight),
        padding=(kRightX,kWindowHeight - (kTopY + kTitleFontSize) / 2),
        font_size= kTitleFontSize,
        font_name=kFontPath
        )
    return l
def topleft_label() :
    l = Label(text="text",
        #color=(0,0,0,1),
        color=kTextColor,
        size=(kWindowWidth, kWindowHeight),
        text_size=(kWindowWidth, kWindowHeight),
        padding=(kLeftX,kWindowHeight - (kTopY + kTopFontSize) / 2),
        font_size= kTopFontSize,
        font_name=kFontPath
        )
    return l
def topright_label() :
    l = Label(text="text",
        #color=(0,0,0,1),
        color=kTextColor,
        halign='right',
        size=(kWindowWidth, kWindowHeight),
        text_size=(kWindowWidth, kWindowHeight),
        padding=(kRightX,kWindowHeight - (kTopY + kTopFontSize) / 2),
        font_size= kTopFontSize,
        font_name=kFontPath
        )
    return l


#KIVY INSTRUCTION GROUPS
# displays a circle with a color depending on beat type
class GemDisplay(InstructionGroup):
    def __init__(self, pos, size, beat):
        super(GemDisplay, self).__init__()
        #self.color = Color(1,1,1,mode='rgb')
        self.color = Color(kTextColor[0], kTextColor[1], kTextColor[2], kTextColor[3], mode='rgba')
        #self.gem = Ellipse(pos=pos, size=size)
        path = kImages[beat]
        self.gem = Rectangle(pos=pos, size=size, texture=Image(path).texture)
        self.add(self.color)
        self.add(self.gem)

    # change the gem's position
    def set_pos(self, pos):
        self.gem.pos = pos

    # change to display this gem being hit
    def on_miss(self):
        #self.color.rgba = (1,1,1,.25)
        self.color.rgba = (kTextColor[0], kTextColor[1], kTextColor[2], 0.25)

    # change to display this gem being hit
    def on_hit(self):
        #self.color.rgba = (1,1,1,0)
        self.color.rgba = (kTextColor[0], kTextColor[1], kTextColor[2], 0)


# a rectangle border
class BoxDisplay(InstructionGroup):
    def __init__(self, pos, size, thickness):
        super(BoxDisplay, self).__init__()
        self.add(Color(kTextColor[0], kTextColor[1], kTextColor[2], kTextColor[3], mode='rgba'))
        self.outer = Rectangle(pos=pos, size=size)
        self.add(self.outer)
        self.add(Color(kBgColor[0], kBgColor[1], kBgColor[2], kBgColor[3], mode='rgba'))
        w = size[0] - 2*thickness
        h = size[1] - 2*thickness
        x = pos[0] + thickness
        y = pos[1] + thickness
        self.thickness = thickness
        self.inner = Rectangle(pos=(x, y), size=(w,h))
        self.add(self.inner)
        for i in range(kNumGems):
            current_x = x + (i + 0.5) * kNumGems ** -1 * w
            self.add(BeatlineDisplay((current_x, y), h))



# displays a vertical line starting at pos and going up for length
class BeatlineDisplay(InstructionGroup):
    def __init__(self, pos, length):
        super(BeatlineDisplay, self).__init__()
        self.length = length
        self.add(Color(kBeatlineColor[0], kBeatlineColor[1], kBeatlineColor[2], kBeatlineColor[3], mode='rgba'))
        self.line = Line(points=(pos[0], pos[1], pos[0], pos[1] + self.length), width=kBeatlineThickness, cap='none')
        self.add(self.line)


# a vertical line which can move linearly between two x positions
class NowbarDisplay(InstructionGroup):
    def __init__(self, x_start, x_end, y0, y1):
        super(NowbarDisplay, self).__init__()
        self.color = Color(kTextColor[0], kTextColor[1], kTextColor[2], kTextColor[3], mode='rgba')
        self.line = Line(points=(x_start, y0, x_start, y1), width=kNowbarThickness, cap='none')
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
                '''
                x = self.current_pos[0] + float(i * self.width)/kNumGems
                y = self.current_pos[1] + kBoxThickness
                '''
                x = self.current_pos[0] + float(i * self.width)/kNumGems + (kGemWindowWidth - kGemWidth) / 2
                y = self.current_pos[1] + kBoxThickness + (kGemWindowHeight - kGemHeight) / 2

                gd = GemDisplay(pos=np.array([x, y]), size=(kGemWidth, kGemHeight), beat=gems[i][1])
                self.gems.append(gd)
                self.add(gd)
            else:
                self.gems.append(None)

    # update measure position on screen
    def move(self, pos):
        initial = self.current_pos
        final = pos

        self.speed = np.array([final[0] - initial[0], 4.0*final[1] - initial[1]])
        self.final_pos = final

        self.moving = True

    def set_pos(self, pos):
        self.current_pos = pos
        self.final_pos = pos
        self.moving = False
        for i in range(len(self.gems)):
            if self.gems[i] != None:
                '''
                x = self.current_pos[0] + float(i * self.width)/kNumGems
                y = self.current_pos[1] + kBoxThickness
                '''
                x = self.current_pos[0] + float(i * self.width)/kNumGems + (kGemWindowWidth - kGemWidth) / 2
                y = self.current_pos[1] + kBoxThickness + (kGemWindowHeight - kGemHeight) / 2
                self.gems[i].set_pos(np.array([x,y]))


    def set_width(self, width):
        self.width = width

    def set_height(self, height):
        self.height = height

    def get_gem(self, gem_idx):
        filtered_gems = filter(lambda x : x != None, self.gems)
        return filtered_gems[gem_idx]


    def on_update(self, dt):
        if self.current_pos[0] == self.final_pos[0] and self.current_pos[1] == self.final_pos[1]:
            self.moving = False
            return False

        if self.moving:
            self.current_pos += self.speed * dt

            if self.current_pos[1] > self.final_pos[1]:
                self.current_pos = self.final_pos

            for i in range(len(self.gems)):
                if self.gems[i] != None:
                    '''
                    x = self.current_pos[0] + float(i * self.width)/kNumGems
                    y = self.current_pos[1] + kBoxThickness
                    '''
                    x = self.current_pos[0] + float(i * self.width)/kNumGems + (kGemWindowWidth - kGemWidth) / 2
                    y = self.current_pos[1] + kBoxThickness + (kGemWindowHeight - kGemHeight) / 2
                    self.gems[i].set_pos(np.array([x,y]))

            return True


# Displays and controls all game elements: Nowbar, Buttons, BarLines, Gems.
class BeatMatchDisplay(InstructionGroup):
    def __init__(self, song_data, seek):
        super(BeatMatchDisplay, self).__init__()

        w = kNumGems * kGemWindowWidth + 2*kBoxThickness
        h = kGemWindowHeight + 2*kBoxThickness
        for i in range(kNumPreviews, -1, -1):
            y = kBottomY + (kNumPreviews - i) * (kGemWindowHeight + 2*kBoxThickness + kMeasureSpacing)
            self.add(BoxDisplay(pos=(kLeftX, y), size=(w,h), thickness=kBoxThickness))
        self.nbd = NowbarDisplay(kLeftX+kBoxThickness/2, kLeftX+w-kBoxThickness/2, y, y+h)
        self.add(self.nbd)

        self.measure_updates = []

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
            last_index = 0
            while j < len(song_data.gems) and song_data.gems[j][0] < song_data.barlines[i]:
                frac = (song_data.gems[j][0] - song_data.barlines[i-1])
                new_idx = np.round(kNumGems * frac * barlength**-1)

                for k in range(int(last_index + 1), int(new_idx)):
                    bar.append(None)

                bar.append((new_idx, song_data.gems[j][1]))
                j += 1
                last_index = new_idx
            self.bar_num_gems.append(len(filter(lambda x: x != None, bar)))
            self.bars.append(MeasureDisplay(pos=(kLeftX, kBottomY), gems=bar, height=h, width=w))
            i += 1

        # current bar and gem (probably 0 unless seek is set)
        self.current_bar = 0
        self.gem_offset = 0
        while self.current_bar < len(self.bars) and seek > song_data.barlines[self.current_bar + 1]:
            self.gem_offset += self.bar_num_gems[self.current_bar]
            self.current_bar += 1


        # time into bar (probably 0s unless seek is set)
        temp_l = filter(lambda b: b < seek, song_data.barlines)
        self.bar_dur = seek - max(temp_l) if temp_l else 0.0

        # show intial graphics
        for i in range(min(kNumPreviews + 1, len(self.bars) - self.current_bar - 1)):
            y = kBottomY + (kNumPreviews - i) * (kGemWindowHeight + 2*kBoxThickness + kMeasureSpacing)
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
            y = kBottomY + (kNumPreviews - i) * (kGemWindowHeight + 2*kBoxThickness + kMeasureSpacing)
            self.bars[self.current_bar + i].move((kLeftX, y))
            self.measure_updates.append(self.bars[self.current_bar + i])
        # add new preview measure
        if self.current_bar + kNumPreviews < len(self.bars):
            self.add(self.bars[self.current_bar + kNumPreviews])

    # called by player, sends gem hit to correct measure
    def gem_hit(self, gem_idx):
        if self.__find_gem(gem_idx) != None:
            self.__find_gem(gem_idx).on_hit()

    def gem_miss(self, gem_idx):
        if self.__find_gem(gem_idx) != None:
            self.__find_gem(gem_idx).on_miss()

    def __find_gem(self, gem_idx):
        # this logic assumes gem hit will never be given more than +/- 1 bar early/late
        if 0 <= (gem_idx - self.gem_offset) < self.bar_num_gems[self.current_bar]:
            return self.bars[self.current_bar].get_gem(gem_idx - self.gem_offset)
        elif (gem_idx - self.gem_offset) < 0:
            return self.bars[self.current_bar-1].get_gem(gem_idx - (self.gem_offset - self.bar_num_gems[self.current_bar-1]))
        else:
            return self.bars[self.current_bar+1].get_gem(gem_idx - (self.gem_offset + self.bar_num_gems[self.current_bar]))

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
        self.nbd.set_progress(progress)

        for measure in self.measure_updates:
            animate = measure.on_update(dt)
            if not animate:
                self.measure_updates.remove(measure)


# HELPER FUNCTIONS
# returns a linear function f(x) given two points (x0, y0) and (x1, y1)
def linear(x0, y0, x1, y1):
    m = (y1-y0)*(x1-x0)**-1
    def f(x):
        return m*(x-x1)+y1
    return f
