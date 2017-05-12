# menu.py

from common.core import *

from graphics import *
from kivy.uix.button import Button

class IntroWidget(Widget):
	def __init__(self):
		super(IntroWidget, self).__init__()
		self.add_widget(title_label())
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

# displays songs and lets user choose one
class SongSelectWidget(Widget):
	def __init__(self, options):
		super(SongSelectWidget, self).__init__()
		self.add_widget(Label(text="Choose A Song",
			color=kTitleColor,
	        halign='center',
	        size=(kWindowWidth, kWindowHeight),
	        text_size=(kWindowWidth, kWindowHeight),
	        padding=(0,kWindowHeight - (kTopY + kTitleFontSize) / 2),
	        font_size= 0.9*kTopY,
	        font_name=kFontPath
	        )
		)
		self.menu = MenuWidget(options, (0,0), (kWindowWidth, kWindowHeight))
		self.add_widget(self.menu)

	def on_touch_down(self, touch):
		self.menu.on_touch_down(touch)

	def on_update(self):
		return self.menu.on_update()


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
				halign='center',
				size=(size[0], buttonY),
				text_size=(size[0], buttonY),
				pos=(0, i*buttonY),
				padding=(0,0),
				font_size=.9*buttonY,
				font_name=kFontPath
				)
			self.add_widget(button)

	def on_touch_down(self, touch):
		if self.pos[0] <= touch.pos[0] < self.pos[0] + self.size[0] and self.pos[1] <= touch.pos[1] < self.pos[1] + self.size[1]:
			self.chosen = int(touch.pos[1] / (self.size[1]/self.num_options))

	def on_update(self):
		return self.chosen