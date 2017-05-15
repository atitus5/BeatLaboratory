# menu.py

from common.core import *

from graphics import *
from kivy.uix.button import Button

# displays the game title
class IntroWidget(Widget):
	def __init__(self):
		super(IntroWidget, self).__init__()
		self.add_widget(Label(text="Beat Laboratory",
	        color=kTitleColor,
	        halign='center',
	        size=(kWindowWidth, kWindowHeight),
	        text_size=(kWindowWidth, kWindowHeight),
	        padding=(0,3*kWindowHeight/8),
	        font_size= 0.25*kWindowHeight,
	        font_name=kFontPath
	        )
		)
		self.add_widget(Label(text="Click anywhere to continue.",
	        color=kTitleColor,
	        halign='center',
	        size=(kWindowWidth, kWindowHeight),
	        text_size=(kWindowWidth, kWindowHeight),
	        padding=(0,kWindowHeight/8),
	        font_size= 0.5*kTopY,
	        font_name=kFontPath
	        )
		)
		self.live = True

	def on_touch_down(self, touch):
		self.live = False

	def on_update(self):
		return self.live


# displays songs and lets user choose one
class SongSelectWidget(Widget):
	def __init__(self, songs):
		super(SongSelectWidget, self).__init__()
		self.num_options = len(songs)
		buttonW = kWindowWidth/self.num_options
		buttonH = kWindowHeight - (kTopY + kTitleFontSize) / 2
		self.add_widget(Label(text="Choose A Song",
			color=kTitleColor,
	        halign='center',
	        size=(kWindowWidth, kWindowHeight),
	        text_size=(kWindowWidth, kWindowHeight),
	        padding=(0, buttonH),
	        font_size= 0.9*kTopY,
	        font_name=kFontPath
	        )
		)
		for i in range(len(songs)):
			button = Button(text=songs[i]['title'],
				color=kTitleColor,
				# background_normal='',
				# background_color=kBgColor,
				halign='center',
				size=(buttonW, buttonH),
				text_size=(buttonW, buttonH),
				pos=(i*buttonW, 0),
				padding=(0, .4*buttonH),
				font_size=.2*buttonH,
				font_name=kFontPath
				)
			self.add_widget(button)
		self.chosen = None

	def on_touch_down(self, touch):
		if 0 <= touch.pos[0] < kWindowWidth and 0 <= touch.pos[1] < kWindowHeight-- (kTopY + kTitleFontSize) / 2:
			self.chosen = int(touch.pos[0] / (kWindowWidth/self.num_options))

	def reset(self):
		self.chosen = None

	def on_update(self):
		return self.chosen


# simple menu class
# takes a list of strings
# on_update returns index of option clicked
class MenuWidget(Widget):
	def __init__(self, options, pos, size):
		super(MenuWidget, self).__init__()
		self.chosen = None
		self.pos = pos
		self.size = size
		self.num_options = len(options)
		buttonY = size[1]/self.num_options
		for i in range(self.num_options):
			button = Button(text=options[i],
				color=kTitleColor,
				background_normal='',
				background_color=kBgColor,
				halign='center',
				size=(size[0], buttonY),
				text_size=(size[0], buttonY),
				pos=(0, i*buttonY),
				padding=(0, buttonY/2),
				font_size=.9*buttonY,
				font_name=kFontPath
				)
			self.add_widget(button)

	def on_touch_down(self, touch):
		if self.pos[0] <= touch.pos[0] < self.pos[0] + self.size[0] and self.pos[1] <= touch.pos[1] < self.pos[1] + self.size[1]:
			self.chosen = int(touch.pos[1] / (self.size[1]/self.num_options))

	def reset(self):
		self.chosen = None

	def on_update(self):
		return self.chosen


# displays the player's score
class ScoreWidget(Widget):
	def __init__(self, title, artist, score):
		super(ScoreWidget, self).__init__()
		self.add_widget(Label(text='"'+title+'" by '+artist,
	        color=kTitleColor,
	        halign='center',
	        size=(kWindowWidth, kWindowHeight),
	        text_size=(kWindowWidth, kWindowHeight),
	        padding=(0,7*kWindowHeight/8),
	        font_size= 0.5*kTopY,
	        font_name=kFontPath
	        )
		)
		self.add_widget(Label(text="Your score is "+str(score)+".",
	        color=kTitleColor,
	        halign='center',
	        size=(kWindowWidth, kWindowHeight),
	        text_size=(kWindowWidth, kWindowHeight),
	        padding=(0,3*kWindowHeight/4),
	        font_size= 0.5*kTopY,
	        font_name=kFontPath
	        )
		)
		self.add_widget(Label(text="Click anywhere to continue.",
	        color=kTitleColor,
	        halign='center',
	        size=(kWindowWidth, kWindowHeight),
	        text_size=(kWindowWidth, kWindowHeight),
	        padding=(0,kWindowHeight/2),
	        font_size= 0.5*kTopY,
	        font_name=kFontPath
	        )
		)
		self.live = True

	def on_touch_down(self, touch):
		self.live = False

	def on_update(self):
		return self.live