#blab.py

# IMPORTS
import sys
sys.path.append('..')

# command line args
kSeek = 0.0
kRecord = False
kUseDefaultModel = False
if len(sys.argv) >= 2:
    kSeek = float(sys.argv[1])
kRecord = False
if len(sys.argv) >= 3:
    kRecord = sys.argv[2] == 'record'
    kUseDefaultModel = sys.argv[2] == 'usedefault'

# other game files
from graphics import *
from game import *
from menu import *

# common
from common.core import *

# CONSTANTS
kSongs = (
    {'title' : '24K Magic',
    'artist' : 'Bruno Mars',
    'path': '../data/24KMagicNoDrums'},
    {'title' : 'Beat It',
    'artist' : 'Michael Jackson',
    'path' : '../data/BeatItNoDrums'}
    )

train_song_path = '../data/training'

kPopupTitle1 = "Welcome to BeatLaboratory!"
kPopupMessage1 = '''Welcome to BeatLaboratory! Because everyone has
their own beatboxing style, we need to learn more
about yours by hearing you! Make sure your microphone
is set up, and play along with this simple track!

Simply press anywhere outside of this box to begin.'''

kPopupTitle2 = "You're ready to go!"
kPopupMessage2 = '''You're all ready to play!
Simply press anywhere outside of this box to begin.'''

# MAIN WIDGET
class MainWidget(BaseWidget):
    def __init__(self):
        super(MainWidget, self).__init__()
        self.state = 'limbo'
        self.trained = kUseDefaultModel
        self.chosen = None # index of selected song

        self.training_widget = GameWidget(train_song_path, train=True)
        self.intro_widget = IntroWidget()
        self.select_widget = SongSelectWidget(kSongs)
        self.game_widget = None # gets populated whenever a song is chosen
        self.score_widget = None # gets populated after song

        self.add_widget(self.intro_widget)
        self.state = 'intro'

    def on_key_down(self, keycode, modifiers):
        if self.state == 'training':
            self.training_widget.on_key_down(keycode, modifiers)

        elif self.state == 'game':
            self.game_widget.on_key_down(keycode, modifiers)

    def __sc_training(self, instance=None):
        self.add_widget(self.training_widget)
        self.training_widget.on_key_down((None, 'p'), [])
        self.state = 'training'

    def __sc_game(self, instance=None):
        self.trained = True
        self.remove_widget(self.training_widget)
        self.add_widget(self.game_widget)
        self.game_widget.on_key_down((None, 'p'), [])
        self.state = 'game'

    def on_touch_down(self, touch):
        if self.state == 'intro':
            self.intro_widget.on_touch_down(touch)

        elif self.state == 'select':
            self.select_widget.on_touch_down(touch)

        elif self.state == 'score':
            self.score_widget.on_touch_down(touch)

    def on_update(self):
        if self.state == 'limbo':
            pass

        elif self.state == 'intro':
            if not self.intro_widget.on_update():
                self.remove_widget(self.intro_widget)
                self.add_widget(self.select_widget)
                self.state = 'select'

        elif self.state == 'select':
            i = self.select_widget.on_update()
            if i != None:
                self.chosen = i
                self.game_widget = GameWidget(kSongs[i]['path'], seek=kSeek)
                self.remove_widget(self.select_widget)
                if not self.trained:
                    self.state = 'limbo'
                    popup = Popup(title=kPopupTitle1,
                        content=Label(text=kPopupMessage1),
                        size_hint=(None, None),
                        size=(400,400))
                    popup.bind(on_dismiss=self.__sc_training)
                    popup.open()
                else:
                    self.add_widget(self.game_widget)
                    self.game_widget.on_key_down((None, 'p'), [])
                    self.state = 'game'

        elif self.state == 'training':
            if not self.training_widget.on_update():
                self.state = 'limbo'
                popup = Popup(title=kPopupTitle2,
                    content=Label(text=kPopupMessage2),
                    size_hint=(None, None),
                    size=(400,400))
                popup.bind(on_dismiss=self.__sc_game)
                popup.open()

        elif self.state == 'game':
            if not self.game_widget.on_update():
                self.state = 'score'
                self.remove_widget(self.game_widget)
                title = kSongs[self.chosen]['title']
                artist = kSongs[self.chosen]['artist']
                score = self.game_widget.get_score()
                self.score_widget = ScoreWidget(title, artist, score)
                self.add_widget(self.score_widget)

        elif self.state == 'score':
            if not self.score_widget.on_update():
                self.state = 'select'
                self.remove_widget(self.score_widget)
                self.select_widget.reset()
                self.add_widget(self.select_widget)

        else:
            print "Fatal error: Invalid state!"
            assert(False)


# LET'S RUN THIS CODE!
run(MainWidget)
