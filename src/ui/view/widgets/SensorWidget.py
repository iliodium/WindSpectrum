from PySide6.QtCore import Qt, QRectF, QLineF
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QSizePolicy, QWidget
from qfluentwidgets import setFont, TransparentPushButton


class SensorWidget(QWidget):
    def __init__(self, action_button):
        super().__init__()
        self.lines_pos = [0.25, 0.5, 0.75]
        self.lines = []

        self.buttons_pos = []
        self.buttons = []

        self.action_button = action_button

        self.model_size = []

        # Создаем QGraphicsScene и QGraphicsView
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)

        # Устанавливаем политику изменения размеров для QGraphicsView
        self.view.setSizePolicy(
            QSizePolicy.Policy.Expanding,  # Горизонтальная политика
            QSizePolicy.Policy.Expanding  # Вертикальная политика
        )

        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # Убираем горизонтальную полосу
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # Убираем вертикальную полосу

        # Подключаем слот для изменения размера сцены при изменении размеров окна
        self.view.resizeEvent = self.resize_scene

        # Добавляем линии на сцену
        self.add_lines()
        self.add_buttons()

    def resize_scene(self, event):
        """Изменяет размер сцены и обновляет линии при изменении размеров окна."""
        # Получаем новые размеры QGraphicsView
        new_size = self.view.size()
        # Устанавливаем новый размер сцены
        self.scene.setSceneRect(QRectF(0, 0, new_size.width(), new_size.height()))
        # Обновляем линии
        self.update_lines()
        self.update_buttons()

        # Вызываем родительский метод resizeEvent
        super().resizeEvent(event)

    def add_lines(self):
        """Добавляет вертикальные линии на сцену."""
        for _ in self.lines_pos:
            line = self.scene.addLine(QLineF(0, 0, 0, 0), QPen(QColor("black"), 2))
            self.lines.append(line)

        # Обновляем их позиции
        self.update_lines()

    def update_lines(self):
        """Обновляет позиции линий на основе текущего размера сцены."""
        # Получаем текущий размер сцены
        scene_width = self.scene.width()
        scene_height = self.scene.height()
        for pos, line in zip(self.lines_pos, self.lines):
            line.setLine(
                scene_width * pos, 0,  # Начальная точка (x, y)
                scene_width * pos, scene_height  # Конечная точка (x, y)
            )

    def add_buttons(self):
        """Добавляет кнопки на сцену."""
        scene_width = self.scene.width()
        scene_height = self.scene.height()

        for i, pos in enumerate(self.buttons_pos, start=1):
            x, y = pos
            button = TransparentPushButton(str(i))
            button.setFixedSize(50, 30)
            setFont(button, 15)
            button.clicked.connect(lambda checked, idx=i: self.action_button(idx - 1))

            proxy_button = self.scene.addWidget(button)
            self.buttons.append(proxy_button)
            proxy_button.setPos(scene_width * x, scene_height * y)

        # Обновляем позиции кнопок
        self.update_buttons()

    def update_buttons(self):
        """Обновляет позиции кнопок на основе текущего размера сцены."""
        scene_width = self.scene.width() * 0.98
        scene_height = self.scene.height() * 0.98
        for pos, button in zip(self.buttons_pos, self.buttons):
            # Устанавливаем позицию кнопки (смещение по x, y)
            x, y = pos
            button.setPos(scene_width * x, scene_height * y)
