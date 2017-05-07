# CONSTANTS
# sreen padding
kLeftX = 20 # left screen padding
kRightX = kLeftX # right screen padding
kBottomY = 40 # bottom padding
kTopY = 60 # top padding

# graphics only
kGemWidth = 150
kGemHeight = kGemWidth
kBoxThickness = 5
kMeasureSpacing = 20 # vertical space between measures
kBeatlineThickness = 2
kNowbarThickness = 3
kTextColor = (242 / 256.0, 242 / 256.0, 242 / 256.0, 0.9)  # Slightly off-white
kBgColor = (13 / 256.0, 26 / 256.0, 38 / 256.0, 0.8) # dark blue
kBeatlineColor = (1,1,1,.5)
# gem image filepaths
kImages = [
    '../data/kick.png',
    '../data/hihat.png',
    '../data/snare.png'
]
kFontPath = "../data/CevicheOne-Regular.ttf"
kTitleFontSize = .8 * kTopY
kBottomFontSize = 0.7 * kBottomY
kReplaceMe = 100
kAnimDur = .25 # number of seconds animations take


# graphics that indirectly affect gameplay
kNumGems = 8 # numbber of gems allowed per bar
kNumPreviews = 1 # number of measures ahead shown
kPreviewScale = 2*3**-1 # previews are this fraction of full size (<= 1)

# these are just convenient
kWindowWidth = (kNumGems * kGemWidth) + (2 * kBoxThickness) + kLeftX + kRightX
kWindowHeight = int((kNumPreviews*kPreviewScale+1)*(kGemHeight + 2*kBoxThickness)) + (kNumPreviews)*kMeasureSpacing + kBottomY + kTopY


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

# set background of screen
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

        self.pos = pos
        self.size = size
        self.target_pos = pos
        self.target_size = size
        self.time = 0
        self.animDur = kAnimDur
        self.animating = False

    # immediately change the gem's position without animating
    def set_pos(self, pos):
        self.gem.pos = pos
        self.pos = pos
        self.target_pos = pos

    def set_size(self, size):
        self.gem.size = size
        self.size = size
        self.target_size = size

    def transform(self, pos, size, animDur = kAnimDur):
        self.target_pos = pos
        self.target_size = size
        self.time = 0
        self.animating = True
        self.animDur = animDur

    # change to display this gem being hit
    def on_miss(self):
        self.color.rgba = (1,1,1,.25) # mostly transparent

    # change to display this gem being hit
    def on_hit(self):
        self.color.rgba = (1,1,1,0) # invisible

    # update position and size of gem if it is animating
    def on_update(self, dt):
        if self.animating:
            self.time += dt
            if self.time > self.animDur:
                self.time = self.animDur
            progress = (self.time * self.animDur**-1)
            x = progress * (self.target_pos[0] - self.pos[0]) + self.pos[0]
            y = progress * (self.target_pos[1] - self.pos[1]) + self.pos[1]
            w = progress * (self.target_size[0] - self.size[0]) + self.size[0]
            h = progress * (self.target_size[1] - self.size[1]) + self.size[1]
            self.gem.pos = (x,y)
            self.gem.size = (w,h)
            if self.time == self.animDur:
                self.pos = self.target_pos
                self.size = self.target_size
                self.animating = False
            else:
                return True
        return False


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


# Displays one bar
# pos is iterable (x, y)
# gems is iterable (time, beat_type) for hit or None for rest
#   where 0 <= beat_num < kNumBeats specifies position
#   and beat_type specifies type of hit needed
class MeasureDisplay(InstructionGroup):
    def __init__(self, pos, size, gems):
        super(MeasureDisplay, self).__init__()

        self.size = size
        self.gems = []
        w = int((size[0])*len(gems)**-1)
        h = size[1]
        y = pos[1]
        for i in range(len(gems)):
            if gems[i] != None:
                x = pos[0] + w*i
                gd = GemDisplay(pos=np.array([x, y]), size=(w, h), beat=gems[i][1])
                self.gems.append(gd)
                self.add(gd)
            else:
                self.gems.append(None)

        self.animating = []

    # update position and size with animation
    def transform(self, pos, size, animDur=kAnimDur):
        self.size = size
        self.animating = []
        for i in range(len(self.gems)):
            if self.gems[i] != None:
                x = pos[0] + float(i * size[0])/len(self.gems)
                y = pos[1]
                w = int((size[0])*len(self.gems)**-1)
                h = size[1]
                self.gems[i].transform((x,y),(w,h), animDur)
                self.animating.append(self.gems[i])

    # immediately update position without animating
    def set_pos(self, pos):
        w = int((self.size[0])*len(self.gems)**-1)
        h = self.size[1]
        y = pos[1]
        for i in range(len(self.gems)):
            if self.gems[i] != None:
                x = pos[0] + w*i
                self.gems[i].set_pos((x,y))
                self.gems[i].set_size((w,h))

    def set_size(self, size):
        self.size = size
        w = int((size[0])*len(self.gems)**-1)
        h = size[1]
        for i in range(len(self.gems)):
            if self.gems[i] != None:
                self.gems[i].set_size((w,h))


    # get the i'th gem of the measure
    def get_gem(self, gem_idx):
        filtered_gems = filter(lambda x : x != None, self.gems)
        return filtered_gems[gem_idx]

    # let animating gems animate
    def on_update(self, dt):
        for gem in self.animating:
            if gem == None:
                continue
            cont = gem.on_update(dt)
            if not cont:
                self.animating.remove(gem)
        return len(self.animating) > 0


# Displays and controls all game elements: Nowbar, Buttons, BarLines, Gems.
class BeatMatchDisplay(InstructionGroup):
    def __init__(self, song_data, seek):
        super(BeatMatchDisplay, self).__init__()

        act_w = kNumGems * kGemWidth
        act_h = kGemHeight
        pre_w = kPreviewScale * kNumGems * kGemWidth
        pre_h = kPreviewScale * kGemHeight
        preLeftX = kLeftX + (1-kPreviewScale) * act_w * .5
        # previews
        for i in range(kNumPreviews, -1, -1):
            y = kBottomY + (kNumPreviews - i) * (pre_h + kMeasureSpacing + 2*kBoxThickness)
            x = (kLeftX if i==0 else preLeftX)
            w = (act_w if i==0 else pre_w) + 2*kBoxThickness
            h = (act_h if i==0 else pre_h) + 2*kBoxThickness
            self.add(BoxDisplay(pos=(x, y), size=(w,h), thickness=kBoxThickness))
        # nowbar for active measure (i = 0 right now)
        self.nbd = NowbarDisplay(kLeftX+kBoxThickness/2, kLeftX+act_w+kBoxThickness/2, y, y+h)
        self.add(self.nbd)

        # tracks which measures are animating
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
            last_index = -1
            while j < len(song_data.gems) and song_data.gems[j][0] < song_data.barlines[i]:
                frac = (song_data.gems[j][0] - song_data.barlines[i-1])
                new_idx = np.round(kNumGems * frac * barlength**-1)

                for k in range(int(last_index + 1), int(new_idx)):
                    bar.append(None)

                bar.append((new_idx, song_data.gems[j][1]))
                j += 1
                last_index = new_idx
            for k in range(int(last_index + 1), kNumGems):
                bar.append(None)
            assert(len(bar) == kNumGems)

            self.bar_num_gems.append(len(filter(lambda x: x != None, bar)))
            self.bars.append(MeasureDisplay(pos=(kWindowWidth/2, -kBottomY), size=(0, 0), gems=bar))
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
            x = (kLeftX if i==0 else preLeftX) + 2*kBoxThickness
            y = kBottomY + (kNumPreviews - i) * (pre_h + 2*kBoxThickness + kMeasureSpacing) + kBoxThickness
            w = act_w if i == 0 else pre_w
            h = act_h if i == 0 else pre_h
            self.bars[self.current_bar + i].set_size((w, h))
            self.bars[self.current_bar + i].set_pos((x, y))
            self.add(self.bars[self.current_bar + i])

    # stop displaying current measure, move preivews up, add next preview measure
    # updates gem_offset and current bar
    def __update_display(self):
        # remove finished measure
        self.remove(self.bars[self.current_bar])
        # update current measure
        self.gem_offset += self.bar_num_gems[self.current_bar]
        animDur = self.bar_durations[self.current_bar]*(2*kNumGems)**-1
        self.current_bar += 1
        # move preview measures up
        act_w = kNumGems * kGemWidth
        act_h = kGemHeight
        pre_w = kPreviewScale * kNumGems * kGemWidth
        pre_h = kPreviewScale * kGemHeight
        preLeftX = kLeftX + (1-kPreviewScale) * act_w * .5
        for i in range(min(kNumPreviews+1, len(self.bars) - self.current_bar - 1)):
            x = (kLeftX if i == 0 else preLeftX) + kBoxThickness
            y = kBottomY + (kNumPreviews - i) * (pre_h + 2*kBoxThickness + kMeasureSpacing) + kBoxThickness
            w = act_w if i == 0 else pre_w
            h = act_h if i == 0 else pre_h
            self.bars[self.current_bar + i].transform((x, y), (w,h), animDur)
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
