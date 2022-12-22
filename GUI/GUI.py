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
        self.rumb = '0'
        self.x = '0.1'
        self.y = '0.1'
        self.z = '0.1'
        self.app = None
        self.core = Core()
        self.summary_plot_mode = 'Cx|Cy|CMz'
        self.summary_plot_scale = 'linear'

    def build(self):
        self.app = Builder.load_file("kv\\MainScreen.kv")
        self.top_left_plot()
        self.top_right_plot()
        self.bottom_left_plot()
        self.bottom_right_plot()

        return self.app

    def top_left_plot(self):
        items_dropdownmenu_top_left = [
            {
                "text": i,
                "viewclass": "OneLineListItem",
                "on_release": lambda x = i: self.plot_top_left(x),
            } for i in ('mean',
                        'min',
                        'max',
                        'std',
                        )
        ]

        self.dropdownmenu_top_left = MDDropdownMenu(
            caller=self.app.ids.second_screen.ids.button_plot_top_left,
            items=items_dropdownmenu_top_left,
            width_mult=1.5,
            max_height=200
        )

    def top_right_plot(self):
        items_dropdownmenu_top_right = [
            {
                "text": i,
                "viewclass": "OneLineListItem",
                "on_release": lambda x = i: self.plot_top_right(x),
            } for i in ('mean',
                        'min',
                        'max',
                        'std',
                        )
        ]
        self.dropdownmenu_top_right = MDDropdownMenu(
            caller=self.app.ids.second_screen.ids.button_plot_top_right,
            items=items_dropdownmenu_top_right,
            width_mult=1.5,
            max_height=200
        )

    def bottom_left_plot(self):
        items_dropdownmenu_bottom_left_1 = [
            {
                "text": i,
                "viewclass": "OneLineListItem",
                "on_release": lambda x = i: self.action_bottom_left_button_1(x),
            } for i in ('Cx',
                        'Cy',
                        'CMz',
                        'Cx|Cy',
                        'Cx|CMz',
                        'Cy|CMz',
                        'Cx|Cy|CMz',
                        )
        ]
        self.dropdownmenu_bottom_left_1 = MDDropdownMenu(
            caller=self.app.ids.second_screen.ids.button_plot_button_left_1,
            items=items_dropdownmenu_bottom_left_1,
            width_mult=2,
            max_height=200
        )
        items_dropdownmenu_bottom_left_2 = [
            {
                "text": i,
                "viewclass": "OneLineListItem",
                "on_release": lambda x = i: self.action_bottom_left_button_2(x)
            } for i in ('log',
                        'linear',
                        )
        ]
        self.dropdownmenu_bottom_left_2 = MDDropdownMenu(
            caller=self.app.ids.second_screen.ids.button_plot_button_left_2,
            items=items_dropdownmenu_bottom_left_2,
            width_mult=1.5,
            max_height=100
        )

    def action_bottom_left_button_1(self, x):
        self.get_summary_plot_mode(x)
        self.change_text_plot_mode(x)

    def action_bottom_left_button_2(self, x):
        self.get_summary_plot_scale(x)
        self.change_text_plot_scale(x)

    def get_summary_plot_mode(self, mode):
        self.summary_plot_mode = mode

    def get_summary_plot_scale(self, scale):
        self.summary_plot_scale = scale

    def change_text_plot_mode(self, text):
        self.app.ids.second_screen.ids.button_plot_button_left_1.text = text

    def change_text_plot_scale(self, text):
        self.app.ids.second_screen.ids.button_plot_button_left_2.text = text

    def bottom_right_plot(self):
        items_dropdownmenu_bottom_right = [
            {
                "text": i,
                "viewclass": "OneLineListItem",
                "on_release": lambda x = i: self.plot_bottom_right(x),
            } for i in range(0, 50, 5)
        ]
        self.dropdownmenu_bottom_right = MDDropdownMenu(
            caller=self.app.ids.second_screen.ids.button_plot_button_right,
            items=items_dropdownmenu_bottom_right,
            width_mult=1.5,
            max_height=200
        )

    def plot_top_left(self, mode):
        if self.check_top():
            fig = self.core.get_plot_isofileds('6', (self.x, self.y, self.z), self.rumb, mode, 'integral_isofields')
            if 'FigureCanvasKivyAgg' in str(self.app.ids.second_screen.ids.field_top_left.children):
                self.app.ids.second_screen.ids.field_top_left.clear_widgets()
            self.app.ids.second_screen.ids.field_top_left.add_widget(FigureCanvasKivyAgg(fig))

    def plot_top_right(self, mode):
        if self.check_top():
            fig = self.core.get_plot_isofileds('6', (self.x, self.y, self.z), self.rumb, mode, 'discrete_isofields')
            if 'FigureCanvasKivyAgg' in str(self.app.ids.second_screen.ids.plot_top_right.children):
                self.app.ids.second_screen.ids.plot_top_right.clear_widgets()
            self.app.ids.second_screen.ids.plot_top_right.add_widget(FigureCanvasKivyAgg(fig))

    def plot_bottom_left(self):
        model_size = (self.x, self.y, self.z)
        fig = self.core.get_plot_summary_spectres('6',
                                                  model_size,
                                                  self.rumb,
                                                  self.summary_plot_mode,
                                                  self.summary_plot_scale,
                                                  'summary_spectres'
                                                  )
        if 'FigureCanvasKivyAgg' in str(self.app.ids.second_screen.ids.plot_bottom_left.children):
            self.app.ids.second_screen.ids.plot_bottom_left.clear_widgets()
        self.app.ids.second_screen.ids.plot_bottom_left.add_widget(FigureCanvasKivyAgg(fig))

    def plot_bottom_right(self, mode):
        if self.check_top():
            # fig = self.core.get_isofileds('6', (self.x, self.y, self.z), self.rumb, mode, 'discrete')
            # if 'FigureCanvasKivyAgg' in str(self.app.ids.second_screen.ids.plot_top_right.children):
            #     self.app.ids.second_screen.ids.plot_top_right.clear_widgets()
            # self.app.ids.second_screen.ids.plot_top_right.add_widget(FigureCanvasKivyAgg(fig))
            pass

    def check_top(self):
        self.get_RUMB()
        self.get_size()
        return self.x and self.y and self.z and self.rumb

    def get_RUMB(self):
        rumb = self.app.ids.second_screen.ids.RUMB.text
        if int(rumb) % 5 != 0:
            print("кратные 5")
        else:
            self.rumb = rumb

    def get_size(self):
        size = self.app.ids.second_screen.ids.model_size.text
        self.x, self.y, self.z = size.split()

    def get_summary_mode(self):
        self.summary_mode = self.app.ids.second_screen.ids.model_size.text

    def get_scale(self):
        self.scale_welch = self.app.ids.second_screen.ids.model_size.text

    def tests(self, text):
        print(dir(text))
        print(text.text)


MainApp().run()
