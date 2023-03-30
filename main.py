import os
import time
import logging

# Local imports
from core import Core
from utils import open_fig

__version__ = "0.0.1"

if __name__ == '__main__':
    from kivy.metrics import dp
    from kivy.uix.popup import Popup
    from kivy.uix.label import Label
    from kivy.core.window import Window
    from kivymd.app import MDApp, Builder
    from kivymd.uix.menu import MDDropdownMenu

    # imports widgets
    from GUI import NavigationBar, IsolatedHighriseScreen

    open('log.log', 'w').close()
else:
    class MDApp(object):
        pass


class MainApp(MDApp):
    logger = logging.getLogger('MainApp'.ljust(15, ' '))
    logger.setLevel(logging.INFO)

    # настройка обработчика и форматировщика
    py_handler = logging.FileHandler("log.log", mode='a')
    py_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")

    # добавление форматировщика к обработчику
    py_handler.setFormatter(py_formatter)
    # добавление обработчика к логгеру
    logger.addHandler(py_handler)

    def __init__(self, **kwargs):
        self.logger.info("Создание графического интерфейса")
        super().__init__(**kwargs)
        self.core = Core(save_mode_fig=True)
        self.alpha = None
        self.angle = None
        self.model_size = None
        self.menu = None

        self.wind_district = None
        self.wind_pressure = None
        self.type_district = None
        self.alpha_type_district = None
        self.k10 = None
        self.z10 = None

        self.app = None
        self.logger.info("Графический интерфейс успешно создан")

    @staticmethod
    def popup(title, error_text):
        Popup(
            title=title,
            title_size=20,
            content=Label(text=error_text, font_size=20),
            size_hint=(None, None),
            size=(350, 150),
            auto_dismiss=True
        ).open()

    def build(self):
        Window.minimum_height = 210
        Window.minimum_width = 600
        self.app = Builder.load_file("GUI/kv/MainScreen.kv")
        self.drop_down_menu()
        self.title = 'Wind Spectrum'

        return self.app

    def save_clipboard(self):
        self.core.save_clipboard()

    def get_angle(self):
        angle = self.app.ids.isolated_highrise_screen.ids.angle.text
        if int(angle) % 5 != 0:
            self.popup('Предупреждение', 'Углы должны быть кратными 5')
        else:
            self.angle = str(int(angle) % 360)

    def get_size(self):
        size = self.app.ids.isolated_highrise_screen.ids.model_size.text
        x, y, z = size.split()
        self.model_size = (x, y, z)

    def get_alpha(self):
        alpha = self.app.ids.isolated_highrise_screen.ids.alpha.text
        if alpha not in ['4', '6']:
            self.popup('Предупреждение', '4 или 6')
        else:
            self.alpha = self.app.ids.isolated_highrise_screen.ids.alpha.text

    def update_parameters(self):
        self.get_size()
        self.get_alpha()
        self.get_angle()

    def report(self):
        self.update_parameters()
        self.core.report_process(self.alpha, self.model_size, self.app.ids.isolated_highrise_screen.ids.report_button)

    def plot_isofields(self, view: str, mode: str):
        self.update_parameters()
        plot_view = None

        if view == 'Дискретные':
            plot_view = 'discrete_isofields'
        elif view == 'Непрерывные':
            plot_view = 'integral_isofields'

        fig = self.core.get_plot_isofields(self.alpha, self.model_size, self.angle, mode, plot_view)
        open_fig(fig)

    def plot_envelopes(self):
        self.update_parameters()
        figs = self.core.get_envelopes(self.alpha, self.model_size, self.angle)
        open_fig(figs)

    def plot_summary_coefficients(self, mode: str):
        self.update_parameters()
        fig = self.core.get_plot_summary_coefficients(self.alpha,
                                                      self.model_size,
                                                      self.angle,
                                                      mode,
                                                      )
        open_fig(fig)

    def plot_spectrum(self, mode: str):
        self.update_parameters()
        fig = self.core.get_plot_summary_spectres(self.alpha,
                                                  self.model_size,
                                                  self.angle,
                                                  mode,
                                                  'log',
                                                  'summary_spectres'
                                                  )
        open_fig(fig)

    def view_isofields(self, view: str):
        items_mode_isofields = [
            {
                "text": i,
                "viewclass": "OneLineListItem",
                "on_release": lambda x = i: self.plot_isofields(view, x),
            } for i in ('mean',
                        'min',
                        'max',
                        'std',
                        )
        ]
        menu_mode_isofields = MDDropdownMenu(
            caller=self.app.ids.isolated_highrise_screen.ids.plots,
            items=items_mode_isofields,
            width_mult=4,
            max_height=dp(224),
            hor_growth="right",
            ver_growth="up",

        )
        menu_mode_isofields.open()

    def get_type_district(self, typ, alpha, k, z):
        self.type_district = typ
        self.alpha_type_district = alpha
        self.k10 = k
        self.z10 = z

    def get_wind_district(self, district, val):
        self.wind_district = district
        self.wind_pressure = val
        print(self.wind_pressure)

    def drop_down_menu(self):
        items_view_isofields = [
            {
                "text": i,
                "viewclass": "OneLineListItem",
                "on_release": lambda x = i: self.view_isofields(x),
            } for i in ('Дискретные',
                        'Непрерывные',
                        )
        ]
        menu_view_isofields = MDDropdownMenu(
            caller=self.app.ids.isolated_highrise_screen.ids.plots,
            items=items_view_isofields,
            width_mult=4,
            max_height=dp(112),
            hor_growth="right",
            ver_growth="up",

        )

        items_mode_summary = [
            {
                "text": i,
                "viewclass": "OneLineListItem",
                "on_release": lambda x = i: self.plot_summary_coefficients(x),
            } for i in ('Cx',
                        'Cy',
                        'CMz',
                        'Cx Cy',
                        'Cx CMz',
                        'Cy CMz',
                        'Cx Cy CMz',
                        )
        ]

        menu_mode_summary = MDDropdownMenu(
            caller=self.app.ids.isolated_highrise_screen.ids.plots,
            items=items_mode_summary,
            width_mult=4,
            max_height=dp(336),
            hor_growth="right",
            ver_growth="up",

        )

        items_mode_spectrum = [
            {
                "text": i,
                "viewclass": "OneLineListItem",
                "on_release": lambda x = i: self.plot_spectrum(x),
            } for i in ('Cx',
                        'Cy',
                        'CMz',
                        'Cx Cy',
                        'Cx CMz',
                        'Cy CMz',
                        'Cx Cy CMz',
                        )
        ]

        menu_mode_spectrum = MDDropdownMenu(
            caller=self.app.ids.isolated_highrise_screen.ids.plots,
            items=items_mode_spectrum,
            width_mult=4,
            max_height=dp(336),
            hor_growth="right",
            ver_growth="up",

        )

        items_menu = [
            {
                "text": 'Изополя',
                "viewclass": "OneLineListItem",
                "on_release": menu_view_isofields.open,
            },
            {
                "text": 'Огибающие',
                "viewclass": "OneLineListItem",
                "on_release": self.plot_envelopes,
            },

            {
                "text": 'Суммарные коэффициенты',
                "viewclass": "OneLineListItem",
                "on_release": menu_mode_summary.open,
            },
            {
                "text": 'Спектры',
                "viewclass": "OneLineListItem",
                "on_release": menu_mode_spectrum.open,
            },
        ]

        self.menu = MDDropdownMenu(
            caller=self.app.ids.isolated_highrise_screen.ids.plots,
            items=items_menu,
            width_mult=4,
            max_height=dp(224),
            hor_growth="right",
            ver_growth="up",
        )

        districts = ('Iа', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII')
        vals = ('0.17', '0.23', '0.30', '0.38', '0.48', '0.60', '0.73', '0.85')

        items_wind_district = [
            {
                "text": f'{district} {val}',
                "viewclass": "OneLineListItem",
                "on_release": lambda x1 = district, x2 = val: self.get_wind_district(x1, x2),
            }
            for district, val in zip(districts, vals)
        ]

        self.wind_district = MDDropdownMenu(
            caller=self.app.ids.isolated_highrise_screen.ids.wind_district_button,
            items=items_wind_district,
            width_mult=4,
            max_height=dp(224),
            hor_growth="right",
            ver_growth="down",
        )

        types = ('A', 'B', 'C')
        alphas = ('0.15', '0.2', '0.25')
        ks = ('1', '0.65', '0.4')
        zs = ('0.76', '1.06', '1.78')

        items_type_district = [
            {
                "text": f'{typ} α {alpha} κ {k} ζ {z}',
                "viewclass": "OneLineListItem",
                "on_release": lambda x1 = typ, x2 = alpha, x3 = k, x4 = z: self.get_type_district(x1, x2, x3, x4),

            }
            for typ, alpha, k, z in zip(types, alphas, ks, zs)
        ]

        self.type_district = MDDropdownMenu(
            caller=self.app.ids.isolated_highrise_screen.ids.type_district_button,
            items=items_type_district,
            width_mult=4,
            max_height=dp(168),
            hor_growth="right",
            ver_growth="down",
        )


if __name__ == "__main__":
    MainApp().run()
    # a = MainApp()
    # for i in range(85,365,5):
    #     print(i)
    #     a.plot_isofields('Непрерывные', 'mean', i)
