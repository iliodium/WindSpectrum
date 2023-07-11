if __name__ == '__main__':
    from kivy.core.window import Window
    from kivymd.app import MDApp, Builder
    from kivymd.uix.screenmanager import MDScreenManager

    from core import Core
    from utils import get_logger
    # окна
    from GUI import HomeScreen, IsolatedHighriseScreen, ReportContent
    # виджеты
    from GUI import NavigationBar

    open('log.log', 'w').close()
else:
    class MDApp(object):
        pass


    class MDScreenManager(object):
        pass


# Hierarchy:
#   WindSpectrum (MDApp)
#   |- WindSpectrumScreens (MDScreenManager)
#      |- HomeScreen (MDScreen)
#      |- IsolatedHighriseScreen (MDScreen)
#      ...

class WindSpectrumScreens(MDScreenManager):
    pass


class WindSpectrum(MDApp):
    def __init__(self, core = None, **kwargs):
        self.core = core
        self.logger = get_logger('WindSpectrum')
        self.logger.info("Создание графического интерфейса")
        super().__init__(**kwargs)

    def build(self):
        Window.minimum_height = 500
        Window.minimum_width = 800
        self.title = 'Wind Spectrum'
        self.gui = Builder.load_file("GUI/kv/WindSpectrum.kv")
        self.logger.info("Графический интерфейс успешно создан")
        return self.gui

    def on_start(self):
        self.root.core = self.core


if __name__ == "__main__":
    WindSpectrum(core=Core()).run()
