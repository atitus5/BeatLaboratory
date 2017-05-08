import numpy as np

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
kImages = (
    '../data/kick.png',
    '../data/hihat.png',
    '../data/snare.png'
)
kFontPath = "../data/CevicheOne-Regular.ttf"
kTitleFontSize = .8 * kTopY
kBottomFontSize = 0.7 * kBottomY
kReplaceMe = 100
kAnimDur = .25 # number of seconds animations take
kHitAnimDur = .5 # number of seconds hit animations take



# graphics that indirectly affect gameplay
kNumGems = 8 # numbber of gems allowed per bar
kPreviews = np.array([2**i*3**-i for i in range(1,7)])

# these are just convenient
kMeasureWidth = kNumGems * kGemWidth
kMeasureHeight = kGemHeight
kWindowWidth = kMeasureWidth + (2 * kBoxThickness) + kLeftX + kRightX
kWindowHeight =  int(sum(kMeasureHeight*kPreviews)) + len(kPreviews)*kMeasureSpacing + (len(kPreviews)+1)*2*kBoxThickness + kMeasureHeight + kBottomY + kTopY
# function to calculate y coordinate of preview i (0 for active measure)
def previewY(i):
    if i >= len(kPreviews):
        return kBottomY
    return sum(kMeasureHeight*kPreviews[i:]) + (len(kPreviews)-i)*(kMeasureSpacing + 2*kBoxThickness) + kBottomY


# IMPORTS
# kivy config (needs to be done first)
from kivy.config import Config
# disable multi-touch emulation
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
# set size of window
Config.set('graphics', 'width', str(kWindowWidth))
Config.set('graphics', 'height', str(kWindowHeight))

from kivy.graphics.instructions import InstructionGroup
from kivy.graphics import Color, Ellipse, Line, Rectangle, Bezier
from kivy.graphics import PushMatrix, PopMatrix, Translate, Scale, Rotate
from kivy.clock import Clock as kivyClock
from kivy.core.image import Image
from kivy.core.window import Window
from kivy.uix.popup import Popup

from common.kivyparticle import ParticleSystem

# set background of screen
Window.clearcolor = kBgColor

kTitleColor = (51 / 256.0, 153 / 256.0, 51 / 256.0, 0.9)  # Green
kTextColor = (242 / 256.0, 242 / 256.0, 242 / 256.0, 0.9)  # Slightly off-white
kFontPath = "../data/CevicheOne-Regular.ttf"

kBeatlineColor = (1,1,1,.5)

kTitleFontSize = 0.9 * kTopY
kTopFontSize = 0.6 * kTopY

from common.gfxutil import *


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


# MENUS AND POPUPS
# TODO


#KIVY INSTRUCTION GROUPS
# displays a circle with a color depending on beat type
class GemDisplay(InstructionGroup):
    def __init__(self, pos, size, beat):
        super(GemDisplay, self).__init__()
        #self.color = Color(1,1,1,mode='rgb')
        self.color = Color(kTextColor[0], kTextColor[1], kTextColor[2], kTextColor[3], mode='rgba')
        #self.gem = Ellipse(pos=pos, size=size)

        self.gem = None
        if beat < len(kImages):
            # We do this because we may want "silence" gems for training, with no image
            path = kImages[beat]
            self.gem = Rectangle(pos=pos, size=size, texture=Image(path).texture)
            self.add(self.color)
            self.add(self.gem)

        self.pos = pos
        self.size = size
        self.target_pos = pos
        self.target_size = size

        self.shift_time = 0
        self.hit_time = 0

        self.animDur = kAnimDur
        self.deathTime = kHitAnimDur

        self.shift_animating = False
        self.hit_animating = False

        self.size_anim = KFAnim((0, kGemWidth), (.1, 1.125*kGemWidth), (.5, 0))
        self.alpha_anim = KFAnim((0, 1), (.5, 0))

        self.prev_anim_size = kGemWidth


    # immediately change the gem's position without animating
    def set_pos(self, pos):
        if self.gem is not None:
            self.gem.pos = pos
        self.pos = pos
        self.target_pos = pos

    # immediately change the gem's size without animating
    def set_size(self, size):
        if self.gem is not None:
            self.gem.size = size
        self.size = size
        self.target_size = size

    # gradually change the gem's position and size over animDur seconds
    def transform(self, pos, size, animDur=kAnimDur):
        self.target_pos = pos
        self.target_size = size
        self.shift_time = 0
        self.shift_animating = True

        self.animDur = animDur

    # change to display this gem being hit
    def on_miss(self):
        self.color.rgba = (1,1,1,.25) # mostly transparent

    # change to display this gem being hit
    def on_hit(self):
        # self.color.rgba = (1,1,1,0) # invisible
        self.hit_time = 0
        self.hit_animating = True



    # update position and size of gem if it is animating
    def on_update(self, dt):
        if self.shift_animating:
            self.shift_time += dt
            if self.shift_time > self.animDur:
                self.shift_time = self.animDur
            progress = (self.shift_time * self.animDur**-1)
            x = progress * (self.target_pos[0] - self.pos[0]) + self.pos[0]
            y = progress * (self.target_pos[1] - self.pos[1]) + self.pos[1]
            w = progress * (self.target_size[0] - self.size[0]) + self.size[0]
            h = progress * (self.target_size[1] - self.size[1]) + self.size[1]

            if self.gem != None:
                self.gem.pos = (x,y)
                self.gem.size = (w,h)

            if self.shift_time == self.animDur:
                self.pos = self.target_pos
                self.size = self.target_size
                self.shift_animating = False

        if self.hit_animating:
            self.hit_time += dt

            if self.hit_time < self.deathTime:
                alpha = self.alpha_anim.eval(self.hit_time)
                r = self.size_anim.eval(self.hit_time)

                self.gem.size = (r, r)
                self.color.a = alpha

                size_diff = self.prev_anim_size - r
                x = self.pos[0] + .5*size_diff
                y = self.pos[1] + .5*size_diff

                self.pos = np.array([x, y])
                self.gem.pos = (x, y)

                self.prev_anim_size = r

            else:
                self.hit_animating = False


        return self.hit_animating or self.shift_animating


# a rectangle border
class BoxDisplay(InstructionGroup):
    def __init__(self, pos, size, thickness):
        super(BoxDisplay, self).__init__()

        #outer
        self.add(Color(kTextColor[0], kTextColor[1], kTextColor[2], kTextColor[3], mode='rgba'))
        self.outer_size = size
        self.outer_pos = pos
        self.outer = Rectangle(pos=pos, size=size)
        self.add(self.outer)

        #inner
        self.add(Color(kBgColor[0], kBgColor[1], kBgColor[2], kBgColor[3], mode='rgba'))
        self.w = size[0] - 2*thickness
        self.h = size[1] - 2*thickness
        self.x = pos[0] + thickness
        self.y = pos[1] + thickness
        self.thickness = thickness
        self.inner = Rectangle(pos=(self.x, self.y), size=(self.w, self.h))
        self.add(self.inner)

        self.beat_lines = []

        for i in range(kNumGems):
            current_x = self.x + (i + 0.5) * kNumGems ** -1 * self.w
            bl = BeatlineDisplay((current_x, self.y), self.h)
            self.add(bl)
            self.beat_lines.append(bl)

        self.anim_time = 0
        self.total_time = .1
        self.pulsing = False

        self.pulse_anim = KFAnim((0, 1), (0.02, 1), (0.05, 1.05), (.1, 1))
        self.prev_inner_size_anim = (self.w, self.h)
        self.prev_outer_size_anim = self.outer_size

    def on_update(self, dt):

        #Really only update anything if the box is currently pulsing
        if self.pulsing:

            #do the pulse animation for every beat line as well
            for line in self.beat_lines:
                line.on_update(dt)

            #if there is still room to animate
            if self.anim_time < self.total_time:
                self.anim_time += dt
                delta_size = self.pulse_anim.eval(self.anim_time)

                #change the inner and outer box sizes
                self.inner.size = (self.w * delta_size, self.h * delta_size)
                self.outer.size = (self.outer_size[0] * delta_size, self.outer_size[1] * delta_size)

                #differences to shift the box to keep it center
                inner_diff_x = self.prev_inner_size_anim[0] - self.inner.size[0]
                inner_diff_y = self.prev_inner_size_anim[1] - self.inner.size[1]

                outer_diff_x = self.prev_outer_size_anim[0] - self.outer.size[0]
                outer_diff_y = self.prev_outer_size_anim[1] - self.outer.size[1]

                #shift positions to keep it center
                self.inner.pos = (self.x + inner_diff_x, self.y + inner_diff_y)
                self.outer.pos = (self.outer_pos[0] + outer_diff_x, self.outer_pos[1] + outer_diff_y)

                #update previous to current
                self.prev_inner_size_anim = (self.inner.size[0], self.inner.size[1])
                self.prev_outer_size_anim = (self.outer.size[0], self.outer.size[1])

            elif self.anim_time == self.total_time:
                self.inner.pos = (self.x, self.y)
                self.inner.size = (self.w, self.h)

                self.outer.pos = self.outer_pos
                self.outer.size = self.outer_size

                self.pulsing = False
                self.anim_time = 0

            else:
                self.anim_time = self.total_time

        return True

    def pulse(self):
        self.pulsing = True
        for line in self.beat_lines:
            line.pulse()
        self.anim_time = 0



# displays a vertical line starting at pos and going up for length
class BeatlineDisplay(InstructionGroup):
    def __init__(self, pos, length):
        super(BeatlineDisplay, self).__init__()
        self.pos = pos
        self.length = length
        self.add(Color(kBeatlineColor[0], kBeatlineColor[1], kBeatlineColor[2], kBeatlineColor[3], mode='rgba'))
        self.line = Line(bezier=(self.pos[0], self.pos[1], self.pos[0], self.pos[1] + .5*self.length, self.pos[0], self.pos[1] + self.length), width=kBeatlineThickness, cap='none')
        self.add(self.line)

        self.anim_time = 0
        self.total_time = .1
        self.pulsing = False

        self.pulse_anim = KFAnim((0, 1), (0.02, 1), (0.05, 1.05), (.1, 1))

        self.frames = np.arange(1000)
        self.vib = 12 * np.sin(2 * np.pi * self.frames / 3) #sine curve for the standing wave
        self.vibrate_anim = KFAnim(*enumerate([self.vib[i] for i in self.frames])) #animation for pluck position

        self.prev_anim_length = self.length

    def on_update(self, dt):

        #only on-update if pulsing
        if self.pulsing:
            if self.anim_time < self.total_time:

                self.anim_time += dt
                delta_size = self.pulse_anim.eval(self.anim_time)

                #change the length of the beatline to match the box
                length = self.length * delta_size
                diff_y = self.prev_anim_length - length

                wave_offset = self.vibrate_anim.eval(1000 * self.anim_time)

                #update the points of the beatline, as well as the width
                self.line.bezier = [self.pos[0], self.pos[1] + diff_y, self.pos[0] + wave_offset, self.pos[1] + .5*length, self.pos[0], self.pos[1] + length]
                self.line.width = kBeatlineThickness*2*delta_size

                self.prev_anim_length = length

            elif self.anim_time == self.total_time:
                self.line.width = kBeatlineThickness
                self.line.bezier = [self.pos[0], self.pos[1], self.pos[0], self.pos[1] + .5*self.length, self.pos[0], self.pos[1] + self.length]

                self.pulsing = False
                self.anim_time = 0

            else:
                self.anim_time = self.total_time

        return True


    def pulse(self):
        self.pulsing = True
        self.anim_time = 0



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

        self.updates = []

    # update position and size with animation
    def transform(self, pos, size, animDur=kAnimDur):
        self.animating = []
        w = int((size[0])*len(self.gems)**-1)
        y = pos[1]
        h = size[1]
        for i in range(len(self.gems)):
            if self.gems[i] != None:
                x = pos[0] + float(i * size[0])/len(self.gems)
                self.gems[i].transform((x,y),(w,h), animDur)

                if self.gems[i] not in self.updates:
                    self.updates.append(self.gems[i])

    # immediately update position without animating
    def set_pos_size(self, pos, size):
        w = int((size[0])*len(self.gems)**-1)
        y = pos[1]
        h = size[1]
        for i in range(len(self.gems)):
            if self.gems[i] != None:
                x = pos[0] + float(i * size[0])/len(self.gems)
                self.gems[i].set_pos((x,y))
                self.gems[i].set_size((w,h))

    # get the i'th gem of the measure
    def get_gem(self, gem_idx):
        for i in range(len(self.gems)):
            if self.gems[i] == None:
                continue
            if gem_idx == 0:
                return self.gems[i], i
            gem_idx -= 1
        print "warning: requested gem out of measure bounds"
        return None, None

    # let animating gems animate
    def on_update(self, dt):
        for elm in self.updates:
            cont = elm.on_update(dt)
            if not cont:
                self.updates.remove(elm)
        return len(self.updates) > 0


class HitParticleDisplay(InstructionGroup):
    def __init__(self, pos, size):
        super(HitParticleDisplay, self).__init__()
        # particle systems on active measure
        self.m_psystems = []
        self.m_hit_times = [-1]
        for i in range(kNumGems):
            ps = ParticleSystem('../particle/particle1.pex')
            ps.emitter_x = pos[0] + (i + 0.5) * kNumGems ** -1 * size[0]
            ps.emitter_y = pos[1] + .5*size[1]
            self.m_psystems.append(ps)
            self.m_hit_times.append(-1)

        # particle systems on bottom corners of screen
        self.bl = ParticleSystem('../particle/particle2.pex')
        self.bl.emitter_x = 0
        self.bl.emitter_y = 0
        self.bl.emit_angle = np.pi*4**-1
        self.br = ParticleSystem('../particle/particle2.pex')
        self.br.emitter_x = kWindowWidth-1
        self.br.emitter_y = 0
        self.br.emit_angle = 3*np.pi*4**-1
        self.b_hit_time = -1


    def puff(self, i):
        if i < 0 or i > len(self.m_psystems):
            print "warning: ps puff out of bounds"
            return
        self.m_psystems[i].start()
        self.m_hit_times[i] = 0
        self.bl.start()
        self.br.start()
        self.b_hit_time = 0

    def install_particle_systems(self, widget):
        for ps in self.m_psystems:
            widget.add_widget(ps)
        widget.add_widget(self.bl)
        widget.add_widget(self.br)

    def on_update(self, dt):
        for i in range(len(self.m_hit_times)):
            if self.m_hit_times[i] >= 0:
                self.m_hit_times[i] += dt
            if self.m_hit_times[i] >= kHitAnimDur:
                self.m_hit_times[i] = -1
                self.m_psystems[i].stop()
        if self.b_hit_time >= 0:
            self.b_hit_time += dt
        if self.b_hit_time >= kHitAnimDur:
            self.b_hit_time = -1
            self.bl.stop()
            self.br.stop()


# Displays and controls all game elements: Nowbar, Buttons, BarLines, Gems.
class BeatMatchDisplay(InstructionGroup):
    def __init__(self, song_data, seek):
        super(BeatMatchDisplay, self).__init__()

        self.main_box_display = None

        # previews
        for i in range(len(kPreviews), -1, -1):
            y = previewY(i)
            x = kLeftX + (0 if i==0 else .5 * (1-kPreviews[i-1]) * (kMeasureWidth+2*kBoxThickness))
            w = kMeasureWidth * (1 if i==0 else kPreviews[i-1]) + 2*kBoxThickness
            h = kMeasureHeight * (1 if i==0 else kPreviews[i-1]) + 2*kBoxThickness
            bd = BoxDisplay(pos=(x, y), size=(w,h), thickness=kBoxThickness)
            self.add(bd)

            if i == 0:
                self.main_box_display = bd

        # nowbar for active measure (i = 0 right now)
        self.nbd = NowbarDisplay(kLeftX+kBoxThickness/2, kLeftX+kMeasureWidth+kBoxThickness/2, y, y+h)
        self.add(self.nbd)
        self.hpd = HitParticleDisplay((x+kBoxThickness,y+kBoxThickness), (w-kBoxThickness, h-kBoxThickness))

        # tracks which measures are animating
        self.updates = []

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
        for i in range(min(len(kPreviews) + 1, len(self.bars) - self.current_bar - 1)):
            y = previewY(i) + kBoxThickness
            x = kLeftX + (0 if i==0 else .5 * (1-kPreviews[i-1]) * (kMeasureWidth+2*kBoxThickness)) + kBoxThickness
            w = kMeasureWidth * (1 if i==0 else kPreviews[i-1])
            h = kMeasureHeight * (1 if i==0 else kPreviews[i-1])
            self.bars[self.current_bar + i].set_pos_size((x,y), (w, h))
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
        for i in range(min(len(kPreviews) + 1, len(self.bars) - self.current_bar - 1)):
            y = previewY(i) + kBoxThickness
            x = kLeftX + (0 if i==0 else .5 * (1-kPreviews[i-1]) * (kMeasureWidth+2*kBoxThickness)) + kBoxThickness
            w = kMeasureWidth * (1 if i==0 else kPreviews[i-1])
            h = kMeasureHeight * (1 if i==0 else kPreviews[i-1])
            self.bars[self.current_bar + i].transform((x, y), (w,h), animDur)

            if self.bars[self.current_bar + i] not in self.updates:
                self.updates.append(self.bars[self.current_bar + i])

        # add new preview measure
        if self.current_bar + len(kPreviews) < len(self.bars):
            self.add(self.bars[self.current_bar + len(kPreviews)])

    # called by player, sends gem hit to correct measure
    def gem_hit(self, gem_idx):
        if self.__find_gem(gem_idx) != None:
            bar, idx = self.__find_gem(gem_idx)
            gem, i = bar.get_gem(idx)
            gem.on_hit()

            if bar not in self.updates:
                self.updates.append(bar)

            if gem not in bar.updates:
                bar.updates.append(gem)

            self.hpd.puff(i)
            self.main_box_display.pulse()

    def gem_miss(self, gem_idx):
        bar, idx = self.__find_gem(gem_idx)
        gem, i = bar.get_gem(idx)
        if gem != None:
            gem.on_miss()

    def __find_gem(self, gem_idx):
        # this logic assumes gem hit will never be given more than +/- 1 bar early/late
        if 0 <= (gem_idx - self.gem_offset) < self.bar_num_gems[self.current_bar]:
            return self.bars[self.current_bar], gem_idx - self.gem_offset
        elif (gem_idx - self.gem_offset) < 0:
            return self.bars[self.current_bar-1], gem_idx - (self.gem_offset - self.bar_num_gems[self.current_bar-1])
        else:
            return self.bars[self.current_bar+1], gem_idx - (self.gem_offset + self.bar_num_gems[self.current_bar])

    def install_particle_systems(self, widget):
        self.hpd.install_particle_systems(widget)

    # call every frame to move nowbar and check if measures need updating
    def on_update(self, dt):
        if self.current_bar >= len(self.bar_durations):
            return
        self.bar_dur += dt
        # we update measure half a beat early
        if self.bar_dur >= (2*kNumGems-1)*(2*kNumGems)**-1 * self.bar_durations[self.current_bar]:
            self.bar_dur -= self.bar_durations[self.current_bar]
            self.__update_display()

        if self.current_bar >= len(self.bar_durations):
            return

        progress = self.bar_dur * self.bar_durations[self.current_bar]**-1
        progress += (2*kNumGems)**-1 # nowbar goes in middle of gem on exact hit, not front
        # we changed progress from [0,1] to [(2*kNumGems)**-1, (2*kNumGems)**-1], so adjust
        if progress >= 1:
            progress -= 1
        # don't move nowbar during first measure since we add a dead measure at the beginning
        if self.current_bar == 0:
            progress = 0
        self.nbd.set_progress(progress)

        for elm in self.updates:
            cont = elm.on_update(dt)
            if not cont:
                self.updates.remove(elm)

        self.hpd.on_update(dt)
        self.main_box_display.on_update(dt)

# HELPER FUNCTIONS
# returns a linear function f(x) given two points (x0, y0) and (x1, y1)
def linear(x0, y0, x1, y1):
    m = (y1-y0)*(x1-x0)**-1
    def f(x):
        return m*(x-x1)+y1
    return f
