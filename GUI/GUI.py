from kivymd.app import MDApp
from kivymd.uix.label import MDLabel
from kivymd.app import Builder

# imports widgets
from SecondScreen import SecondScreen
from NavigationBar import NavigationBar

# local imports
from core.core import Core

class MainApp(MDApp):
    def __init__(self):
        super().__init__()
        self.RUMB = None
        self.x = None
        self.y = None
        self.z = None


    def build(self):
        return Builder.load_file("kv\\MainScreen.kv")

    def check(self):
        if self.x and self.y and self.z and self.RUMB:
            Core.get_plots('4', (self.x, self.y, self.z), self.RUMB, )

    def get_RUMB(self, text_field):
        if int(text_field.text) % 5 != 0:
            print("кратные 5")
        else:
            self.RUMB = text_field.text
        self.check()

    def get_size(self, text_field):
        self.x, self.y, self.z = text_field.text.split()


    def tests(self,text):
        print(dir(text))
        print(text.text)


MainApp().run()