# dev imports (optional)
import os
# os.environ["KIVY_NO_CONSOLELOG"] = "1"


from kivymd.app import MDApp
from kivymd.uix.label import MDLabel
from kivymd.app import Builder
from matplotlib import pyplot as plt
import numpy as np
from kivy.garden.matplotlib import FigureCanvasKivyAgg
from kivymd.uix.menu import MDDropdownMenu

# imports widgets
from SecondScreen import SecondScreen
from NavigationBar import NavigationBar

# local imports
from core.core import Core


class MainApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rumb = None
        self.x = None
        self.y = None
        self.z = None
        self.app = None
        self.core = Core()

    def build(self):
        self.app = Builder.load_file("kv\\MainScreen.kv")

        menu_items = [
            {
                "text": i,
                "viewclass": "OneLineListItem",
                "on_release": lambda x = i: self.check(x),
            } for i in ('mean',
                        'min',
                        'max',
                        'std',
                        )
        ]

        self.menu = MDDropdownMenu(
            caller=self.app.ids.second_screen.ids.isofield_modes,
            items=menu_items,
            width_mult=4,
        )

        return self.app

    def check(self, mode):
        self.get_rumb()
        self.get_size()

        if self.x and self.y and self.z and self.rumb:
            fig = self.core.get_isofileds('6', (self.x, self.y, self.z), self.rumb, mode, 'discrete')
            fig1 = self.core.get_isofileds('6', (self.x, self.y, self.z), self.rumb, mode, 'integral')
            if 'FigureCanvasKivyAgg' in str(self.app.ids.second_screen.ids.field_top_right.children):
                self.app.ids.second_screen.ids.field_top_right.clear_widgets()
            if 'FigureCanvasKivyAgg' in str(self.app.ids.second_screen.ids.field_top_left.children):
                self.app.ids.second_screen.ids.field_top_left.clear_widgets()

            self.app.ids.second_screen.ids.field_top_right.add_widget(FigureCanvasKivyAgg(fig))
            self.app.ids.second_screen.ids.field_top_left.add_widget(FigureCanvasKivyAgg(fig1))
            # self.app.ids.second_screen.ids.field_bottom_right.add_widget(FigureCanvasKivyAgg(plt.gcf()))
            # self.app.ids.second_screen.ids.field_bottom_left.add_widget(FigureCanvasKivyAgg(plt.gcf()))

    def get_rumb(self):
        rumb = self.app.ids.second_screen.ids.rumb.text
        if int(rumb) % 5 != 0:
            print("кратные 5")
        else:
            self.rumb = rumb

    def get_size(self):
        size = self.app.ids.second_screen.ids.model_size.text
        self.x, self.y, self.z = size.split()

    def tests(self, text):
        print(dir(text))
        print(text.text)


MainApp().run()
