# import site
# site.USER_BASE = site.getuserbase()


# build
# from kivy.resources import resource_add_path
# from kivy.factory import Factory
#
# if hasattr(sys, '_MEIPASS'):
#     resource_add_path(os.path.join(sys._MEIPASS))
# # Factory.register("MainScreen", module="widgets.AxisMarkupX")
# Factory.register("SecondScreen", module="NavigationBar")
# Factory.register("NavigationBar", module="SecondScreen")
# dev imports (optional)
# import os
# os.environ["KIVY_NO_CONSOLELOG"] = "1"
import sys

import psycopg2
from kivymd.app import MDApp
from kivymd.uix.button import MDFlatButton
from kivymd.uix.dialog import MDDialog
from kivymd.uix.label import MDLabel
from kivymd.app import Builder
from matplotlib import pyplot as plt
import numpy as np

from matplotlib_backend import FigureCanvasKivyAgg
from kivymd.uix.menu import MDDropdownMenu

# imports widgets
from GUI.IsolatedHighriseScreen import IsolatedHighriseScreen
from GUI.NavigationBar import NavigationBar

# local imports
from core.core import Core


class MainApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rumb = '0'
        self.x = '0.1'
        self.y = '0.1'
        self.z = '0.1'
        self.alpha = '6'
        self.app = None
        try:
            self.core = Core()
        except psycopg2.OperationalError as e:
            self.core = None
            self.error_connection_to_database = e
        self.summary_welch_plot_mode = 'Cx|Cy|CMz'
        self.summary_coefficients_plot_mode = 'Cx|Cy|CMz'
        self.summary_plot_scale = 'linear'

    def build(self):
        if self.core is None:
            dialog = MDDialog(
                text=f"Could not connect to database.\n{self.error_connection_to_database}",
                buttons=[
                    MDFlatButton(
                        text="Ok",
                        theme_text_color="Custom",
                        text_color=self.theme_cls.primary_color,
                        on_release=lambda *args: sys.exit(1)
                    )
                ],
                on_dismis=lambda *args: sys.exit(0)
            )
            return dialog.open()
        self.app = Builder.load_file("GUI/kv/MainScreen.kv")
        self.top_left_plot_dropmenu()
        self.top_right_plot_dropmenu()
        self.bottom_left_plot_dropmenu()
        self.bottom_right_plot_dropmenu()
        return self.app

    def top_left_plot_dropmenu(self):
        items_dropdownmenu_top_left = [
            {
                "text": i,
                "viewclass": "OneLineListItem",
                "on_release": lambda x = i: self.top_left_plot(x),
            } for i in ('mean',
                        'min',
                        'max',
                        'std',
                        )
        ]

        self.dropdownmenu_top_left = MDDropdownMenu(
            caller=self.app.ids.isolated_highrise_screen.ids.button_plot_top_left,
            items=items_dropdownmenu_top_left,
            width_mult=1.5,
            max_height=200
        )

    def top_right_plot_dropmenu(self):
        items_dropdownmenu_top_right = [
            {
                "text": i,
                "viewclass": "OneLineListItem",
                "on_release": lambda x = i: self.top_right_plot(x),
            } for i in ('mean',
                        'min',
                        'max',
                        'std',
                        )
        ]
        self.dropdownmenu_top_right = MDDropdownMenu(
            caller=self.app.ids.isolated_highrise_screen.ids.button_plot_top_right,
            items=items_dropdownmenu_top_right,
            width_mult=1.5,
            max_height=200
        )

    def bottom_left_plot_dropmenu(self):
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
            caller=self.app.ids.isolated_highrise_screen.ids.button_plot_button_left_1,
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
            caller=self.app.ids.isolated_highrise_screen.ids.button_plot_button_left_2,
            items=items_dropdownmenu_bottom_left_2,
            width_mult=1.5,
            max_height=100
        )

    def bottom_right_plot_dropmenu(self):
        items_dropdownmenu_bottom_right_1 = [
            {
                "text": i,
                "viewclass": "OneLineListItem",
                "on_release": lambda x = i: self.action_bottom_right_button_1(x),
            } for i in ('Cx',
                        'Cy',
                        'CMz',
                        'Cx|Cy',
                        'Cx|CMz',
                        'Cy|CMz',
                        'Cx|Cy|CMz',
                        )
        ]
        self.dropdownmenu_bottom_right_1 = MDDropdownMenu(
            caller=self.app.ids.isolated_highrise_screen.ids.button_plot_button_right_1,
            items=items_dropdownmenu_bottom_right_1,
            width_mult=2,
            max_height=200
        )

    def action_bottom_left_button_1(self, x):
        self.get_summary_welch_plot_mode(x)
        self.change_text_plot_mode(x)

    def action_bottom_left_button_2(self, x):
        self.get_summary_plot_scale(x)
        self.change_text_plot_scale(x)

    def action_bottom_right_button_1(self, x):
        self.get_summary_coefficients_plot_mode(x)
        self.change_text_plot_mode_1(x)

    def get_summary_coefficients_plot_mode(self, x):
        self.summary_coefficients_plot_mode = x

    def get_summary_welch_plot_mode(self, mode):
        self.summary_welch_plot_mode = mode

    def get_summary_plot_scale(self, scale):
        self.summary_plot_scale = scale

    def change_text_plot_mode(self, text):
        self.app.ids.isolated_highrise_screen.ids.button_plot_button_left_1.text = text

    def change_text_plot_mode_1(self, text):
        self.app.ids.isolated_highrise_screen.ids.button_plot_button_right_1.text = text

    def change_text_plot_scale(self, text):
        self.app.ids.isolated_highrise_screen.ids.button_plot_button_left_2.text = text

    def top_left_plot(self, mode):
        model_size = (self.x, self.y, self.z)
        fig = self.core.get_plot_isofileds(self.alpha, model_size, self.rumb, mode, 'integral_isofields')
        self.app.ids.isolated_highrise_screen.ids.plot_top_left.clear_widgets()
        self.app.ids.isolated_highrise_screen.ids.plot_top_left.add_widget(FigureCanvasKivyAgg(fig))

    def top_right_plot(self, mode):
        model_size = (self.x, self.y, self.z)
        fig = self.core.get_plot_isofileds(self.alpha, model_size, self.rumb, mode, 'discrete_isofields')
        self.app.ids.isolated_highrise_screen.ids.plot_top_right.clear_widgets()
        self.app.ids.isolated_highrise_screen.ids.plot_top_right.add_widget(FigureCanvasKivyAgg(fig))

    def bottom_left_plot(self):
        model_size = (self.x, self.y, self.z)
        fig = self.core.get_plot_summary_spectres(self.alpha,
                                                  model_size,
                                                  self.rumb,
                                                  self.summary_welch_plot_mode,
                                                  self.summary_plot_scale,
                                                  'summary_spectres'
                                                  )

        self.app.ids.isolated_highrise_screen.ids.plot_bottom_left.clear_widgets()
        self.app.ids.isolated_highrise_screen.ids.plot_bottom_left.add_widget(FigureCanvasKivyAgg(fig))

    def bottom_right_plot(self):
        model_size = (self.x, self.y, self.z)
        fig = self.core.get_plot_summary_coefficients(self.alpha,
                                                      model_size,
                                                      self.rumb,
                                                      self.summary_coefficients_plot_mode,
                                                      'summary_coefficients'
                                                      )
        self.app.ids.isolated_highrise_screen.ids.plot_bottom_right.clear_widgets()
        self.app.ids.isolated_highrise_screen.ids.plot_bottom_right.add_widget(FigureCanvasKivyAgg(fig))

    def get_RUMB(self):
        rumb = self.app.ids.isolated_highrise_screen.ids.RUMB.text
        if int(rumb) % 5 != 0:
            print("кратные 5")
        else:
            self.rumb = rumb

    def get_size(self):
        size = self.app.ids.isolated_highrise_screen.ids.model_size.text
        self.x, self.y, self.z = size.split()

    def get_alpha(self):
        self.alpha = self.app.ids.isolated_highrise_screen.ids.alpha.text


if __name__ == '__main__':
    MainApp().run()
