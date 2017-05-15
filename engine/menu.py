# menu.py

from common.core import *

from graphics import *
from kivy.uix.image import Image as ImageWidget
from kivy.uix.button import Button

kHoverColor = (33 / 256.0, 65 / 256.0, 95 / 256.0, 0.5)

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
		self.songs = songs
		H = kWindowHeight - (kTopY + kTitleFontSize) / 2
		H2 = H/len(self.songs)

		# screen header
		self.add_widget(Label(text="Choose A Song",
			color=kTitleColor,
	        halign='center',
	        size=(kWindowWidth, kWindowHeight),
	        text_size=(kWindowWidth, kWindowHeight),
	        padding=(0, H),
	        font_size= 0.9*kTopY,
	        font_name=kFontPath
	        )
		)

		# album art, changes with hovered title
		self.image = ImageWidget(size=(kWindowWidth/2, H),
			pos=(kWindowWidth/2,0)
			)

		# buttons for titles
		self.buttons = []
		for i in range(len(songs)):
			title = Button(text=songs[i]['title'],
				color=kTitleColor,
				background_normal='',
				background_color=kBgColor,
				size=(kWindowWidth/2, H2),
				text_size=(kWindowWidth/2, H2),
				pos=(0, i*H2),
				padding=(kLeftX, .5*H2),
				font_size=.5*H2,
				font_name=kFontPath
				)
			self.add_widget(title)
			self.buttons.append(title)
			artist = Label(text=songs[i]['artist'],
				color=kTitleColor,
				size=(kWindowWidth/2, H2),
				text_size=(kWindowWidth/2, H2),
				pos=(0, i*H2),
				padding=(2*kLeftX, .20*H2),
				font_size=.25*H2,
				font_name=kFontPath
				)
			self.add_widget(artist)

		# state
		self.hovered = None
		self.chosen = None

	def set_mouse_pos(self, mouse_pos):
		H = kWindowHeight - (kTopY + kTitleFontSize) / 2
		if 0 <= mouse_pos[0] < kWindowWidth/2 and 0 <= mouse_pos[1] < H:
			hovered = int(mouse_pos[1] / (H/len(self.songs)))
			if self.hovered != hovered:
				if self.hovered != None:
					self.buttons[self.hovered].background_color = kBgColor
				self.image.source = self.songs[hovered]['image']
				self.buttons[hovered].background_color = kHoverColor
			if self.hovered == None:
				self.add_widget(self.image)
			self.hovered = hovered
		else:
			self.remove_widget(self.image)
			self.image.source = ''
			if self.hovered != None:
				self.buttons[self.hovered].background_color = kBgColor
			self.hovered = None

	def on_touch_down(self, touch):
		H = kWindowHeight - (kTopY + kTitleFontSize) / 2
		mouse_pos = touch.pos
		if 0 <= mouse_pos[0] < kWindowWidth/2 and 0 <= mouse_pos[1] < H:
			self.chosen = int(mouse_pos[1] / (H/len(self.songs)))

	def reset(self):
		self.chosen = None
		if self.hovered != None:
			self.buttons[self.hovered].background_color = kBgColor
			self.remove_widget(self.image)
			self.image.source = ''
			self.hovered = None

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
	        font_size= kTopY,
	        font_name=kFontPath
	        )
		)
		self.add_widget(Label(text="Your score is ",
	        color=kTitleColor,
	        halign='center',
	        size=(kWindowWidth, kWindowHeight),
	        text_size=(kWindowWidth, kWindowHeight),
	        padding=(0,5*kWindowHeight/8),
	        font_size= 0.5*kTopY,
	        font_name=kFontPath
	        )
		)
		self.add_widget(Label(text=str(score),
	        color=kTitleColor,
	        halign='center',
	        size=(kWindowWidth, kWindowHeight),
	        text_size=(kWindowWidth, kWindowHeight),
	        padding=(0,3*kWindowHeight/8),
	        font_size= kWindowHeight/4,
	        font_name=kFontPath
	        )
		)
		self.add_widget(Label(text="Click anywhere to continue.",
	        color=kTitleColor,
	        halign='center',
	        size=(kWindowWidth, kWindowHeight),
	        text_size=(kWindowWidth, kWindowHeight),
	        padding=(0,.1*kWindowHeight),
	        font_size= 0.5*kTopY,
	        font_name=kFontPath
	        )
		)
		self.live = True

	def on_touch_down(self, touch):
		self.live = False

	def on_update(self):
		return self.live