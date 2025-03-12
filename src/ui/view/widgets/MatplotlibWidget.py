from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from qfluentwidgets import ScrollArea


class MatplotlibWidget(ScrollArea):
    def __init__(
            self,
            fig,
            title='График',
            parent=None
    ):
        super().__init__(parent)
        container = QWidget()
        self.setWidget(container)
        self.setWidgetResizable(True)  # Обеспечивает изменение размеров содержимого

        self.fig = fig
        self.canvas = FigureCanvasQTAgg(fig)
        # Создаем Toolbar
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        # Добавляем компоновку для отображения FigureCanvas
        self.plotLayout = QVBoxLayout(container)
        self.plotLayout.addWidget(self.canvas)
        self.plotTitle = title
