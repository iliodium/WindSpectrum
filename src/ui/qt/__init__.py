import sys

from PySide6.QtWidgets import (QApplication,
                               QMainWindow,)
from src.ui.qt.classes.main import Ui_MainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)


def main():
    # Create the Qt Application
    app = QApplication(sys.argv)
    # Create a button, connect it and show it
    window = MainWindow()
    window.show()
    print("Stating")
    sys.exit(app.exec())
