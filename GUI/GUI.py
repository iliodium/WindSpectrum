from kivymd.app import MDApp
from kivymd.uix.label import MDLabel
from kivymd.app import Builder

# local imports
from SecondScreen import SecondScreen
from NavigationBar import NavigationBar

class MainApp(MDApp):
    def build(self):
        return Builder.load_file("kv\\MainScreen.kv")

    def tests(self,text):
        print(dir(text.root))
        print(text.root._get_screen_names())
        print(text.root.screens)
        print(text.root.current_screen)
        print(text.root.current)


MainApp().run()