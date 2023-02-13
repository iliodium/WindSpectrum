# local imports
from core import Core
from utils import open_fig

if __name__ == '__main__':
    # imports widgets

    from GUI import NavigationBar, IsolatedHighriseScreen

    from kivy.metrics import dp
    from kivy.core.window import Window
    from kivymd.app import MDApp, Builder
    from kivymd.uix.menu import MDDropdownMenu
else:
    class MDApp(object):
        pass


class MainApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.core = Core()
        self.alpha = None
        self.angle = None
        self.menu = None
        self.app = None
        self.model_size = None

    def build(self):
        Window.minimum_height = 210
        Window.minimum_width = 600
        self.app = Builder.load_file("GUI/kv/MainScreen.kv")
        self.drop_down_menu_plots()
        self.title = 'Wind Spectrum'
        return self.app

    def get_angle(self):
        angle = self.app.ids.isolated_highrise_screen.ids.angle.text
        if int(angle) % 5 != 0:
            print("кратные 5")
        else:
            self.angle = angle

    def get_size(self):
        size = self.app.ids.isolated_highrise_screen.ids.model_size.text
        x, y, z = size.split()
        self.model_size = (x, y, z)

    def get_alpha(self):
        self.alpha = self.app.ids.isolated_highrise_screen.ids.alpha.text

    def update_parameters(self):
        self.get_size()
        self.get_alpha()
        self.get_angle()

    def report(self):
        self.update_parameters()
        self.core.report_process(self.alpha, self.model_size)

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
                "on_release": lambda x=i: self.plot_isofields(view, x),
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

    def drop_down_menu_plots(self):
        items_view_isofields = [
            {
                "text": i,
                "viewclass": "OneLineListItem",
                "on_release": lambda x=i: self.view_isofields(x),
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
                "on_release": lambda x=i: self.plot_summary_coefficients(x),
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
                "on_release": lambda x=i: self.plot_spectrum(x),
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


if __name__ == "__main__":
    MainApp().run()
