import os
import sys

from PySide6.QtCore import (QSize,
                            Qt,
                            QTimer,)
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QApplication,
                               QFrame,
                               QHBoxLayout,)
from qfluentwidgets import (FluentIcon,
                            FluentWindow,
                            NavigationItemPosition,
                            SplashScreen,
                            SubtitleLabel,
                            SystemThemeListener,
                            isDarkTheme,
                            setFont,)
from src.ui.config.config import cfg
from src.ui.view.isolated_high_rise_interface import (
    IsolatedHighRiseInterface as _IsolatedHighRiseInterface,)
from src.ui.view.main_interface import MainInterface as _MainInterface


class Widget(QFrame):

    def __init__(
            self,
            text: str,
            parent=None
    ):
        super().__init__(parent=parent)
        self.label = SubtitleLabel(text, self)
        self.hBoxLayout = QHBoxLayout(self)

        setFont(self.label, 24)
        self.label.setAlignment(Qt.AlignCenter)
        self.hBoxLayout.addWidget(self.label, 1, Qt.AlignCenter)
        self.setObjectName(text.replace(' ', '-'))

        # !IMPORTANT: leave some space for title bar
        self.hBoxLayout.setContentsMargins(0, 32, 0, 0)


class MainWindow(FluentWindow):

    def __init__(
            self
    ):
        super().__init__()
        self.initWindow()

        # create system theme listener
        self.themeListener = SystemThemeListener(self)

        # create sub interface
        self.MainInterface = _MainInterface(self)
        self.IsolatedHighRiseInterface = _IsolatedHighRiseInterface(self)

        # enable acrylic effect
        self.navigationInterface.setAcrylicEnabled(True)

        # add items to navigation interface
        self.initNavigation()
        self.splashScreen.finish()

        # start theme listener
        self.themeListener.start()

    def initNavigation(
            self
    ):
        # add navigation items
        self.addSubInterface(self.MainInterface, FluentIcon.HOME, self.tr('Начальная страница'))

        self.WindLoadsInterface = Widget('Wind Loads Interface', self)
        self.AerodynamicInterferenceInterface = Widget('AerodynamicInterferenceInterface', self)
        self.IsolatedInterface = Widget('Isolated Interface', self)
        self.AerodynamicInterferenceOfHighRiseInterface = Widget('Aerodynamic Interference Of High Rise Buildings',
                                                                 self)
        self.IsolatedLowRiseBuildingsWithoutInterface = Widget('Isolated Low Rise Buildings Without Cornice', self)
        self.IsolatedLowRiseBuildingsWithInterface = Widget('Isolated Low Rise Buildings With Cornices', self)
        self.AerodynamicInterferenceOfLowRiseInterface = Widget('Aerodynamic Interference Of Low Rise Buildings', self)
        self.IsolatedLowRiseBuildingsInterface = Widget('Isolated Of Low Rise Buildings', self)

        # Add pages to sub interface
        self.addSubInterface(self.IsolatedInterface,
                             FluentIcon.EDUCATION,
                             self.tr('Изолированные здания'),
                             NavigationItemPosition.SCROLL)
        self.addSubInterface(self.IsolatedHighRiseInterface,
                             '',
                             self.tr('Высотные здания'),
                             parent=self.IsolatedInterface)
        self.addSubInterface(self.IsolatedLowRiseBuildingsWithoutInterface,
                             '',
                             self.tr('Низкоэтажные без карниза'),
                             parent=self.IsolatedInterface)
        self.addSubInterface(self.IsolatedLowRiseBuildingsWithInterface,
                             '',
                             self.tr('Низкоэтажные с карнизом'),
                             parent=self.IsolatedInterface)

        self.addSubInterface(self.AerodynamicInterferenceInterface,
                             FluentIcon.EDUCATION,
                             self.tr('Аэродинамическая интерференция'),
                             NavigationItemPosition.SCROLL)
        self.addSubInterface(self.AerodynamicInterferenceOfHighRiseInterface,
                             '',
                             self.tr('Высотные здания'),
                             parent=self.AerodynamicInterferenceInterface)
        self.addSubInterface(self.AerodynamicInterferenceOfLowRiseInterface,
                             '',
                             self.tr('Низкоэтажные здания'),
                             parent=self.AerodynamicInterferenceInterface)

        self.navigationInterface.addSeparator()

    def initWindow(
            self
    ):
        # Logo of splash screen
        main_logo = QIcon('src/ui/resource/images/main_logo.png')
        # The logo in the upper left corner of the program window
        mini_logo = QIcon('src/ui/resource/images/mini_logo.png')

        self.resize(960, 780)
        self.setMinimumWidth(760)
        self.setWindowIcon(mini_logo)
        self.setWindowTitle('WindSpectrum')

        self.setMicaEffectEnabled(cfg.get(cfg.micaEnabled))

        # create splash screen
        self.splashScreen = SplashScreen(main_logo, self)
        self.splashScreen.setIconSize(QSize(622, 584))
        self.splashScreen.raise_()

        desktop = QApplication.screens()[0].availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w // 2 - self.width() // 2, h // 2 - self.height() // 2)
        self.show()
        QApplication.processEvents()

    def resizeEvent(
            self,
            e
    ):
        super().resizeEvent(e)
        if hasattr(self, 'splashScreen'):
            self.splashScreen.resize(self.size())

    def closeEvent(
            self,
            e
    ):
        self.themeListener.terminate()
        self.themeListener.deleteLater()
        super().closeEvent(e)

    def _onThemeChangedFinished(
            self
    ):
        super()._onThemeChangedFinished()

        # retry
        if self.isMicaEffectEnabled():
            QTimer.singleShot(100, lambda: self.windowEffect.setMicaEffect(self.winId(), isDarkTheme()))


def main():
    # enable dpi scale
    if cfg.get(cfg.dpiScale) != "Auto":
        os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
        os.environ["QT_SCALE_FACTOR"] = str(cfg.get(cfg.dpiScale))

    # create application
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_DontCreateNativeWidgetSiblings)

    # create main window
    w = MainWindow()
    w.show()

    app.exec()
