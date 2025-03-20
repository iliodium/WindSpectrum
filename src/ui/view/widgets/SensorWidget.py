from PySide6.QtCore import (QLineF,
                            QRectF,
                            Qt,)
from PySide6.QtGui import (QBrush,
                           QColor,
                           QFont,
                           QPen,)
from PySide6.QtWidgets import (QGraphicsEllipseItem,
                               QGraphicsScene,
                               QGraphicsTextItem,
                               QGraphicsView,
                               QSizePolicy,
                               QWidget,)


class ClickablePoint(QGraphicsEllipseItem):
    def __init__(self, x, y, index, parent=None):
        super().__init__(0, 0, 0, 0, parent)  # Начальный размер будет обновлён
        self.index = index  # Индекс точки для идентификации

        # Создаем точку
        self.relative_x = x  # Относительная позиция по X (от 0 до 1)
        self.relative_y = y  # Относительная позиция по Y (от 0 до 1)
        self.relative_radius = 10  # Относительный радиус (например, 0.05 = 5% от ширины сцены)
        self.setAcceptHoverEvents(True)
        self.setBrush(QBrush(Qt.black))  # Устанавливаем цвет точки

        # Создаем подпись
        self.label = QGraphicsTextItem(str(self.index + 1), self)  # +1 для удобства (начинаем с 1)
        self.label.setDefaultTextColor(Qt.black)  # Цвет текста
        self.label.setFont('Segoe UI')

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.on_click(self.index)

    def set_on_click(self, callback):
        """Метод для установки внешнего обработчика клика"""
        self.on_click = callback

    def updatePositionAndRadius(self, scene_width, scene_height):
        """Обновляет позицию и радиус точки при изменении размера сцены"""
        # Пересчитываем радиус
        scale = min(
            scene_width / self.relative_radius,
            scene_height / self.relative_radius
        ) * 0.2

        radius = scale
        self.setRect(0, 0, radius, radius)  # Устанавливаем новый размер
        # Пересчитываем позицию
        x = scene_width * self.relative_x - radius / 2
        y = scene_height * self.relative_y - radius / 2
        self.setPos(x, y)

        self.label.setPos(-radius * 0.5, radius / 2)

        new_size = int(radius)
        self.label.setFont(QFont(self.label.font().family(), new_size))


class SensorWidget(QWidget):
    def __init__(self, action_button):
        super().__init__()
        self.lines_pos = [0.25, 0.5, 0.75]
        self.lines = []

        self.points_pos = []
        self.points = []

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
        self.add_points()

    def resize_scene(self, event):
        """Изменяет размер сцены и обновляет линии при изменении размеров окна."""
        # Получаем новые размеры QGraphicsView
        new_size = self.view.size()
        # Устанавливаем новый размер сцены
        self.scene.setSceneRect(QRectF(0, 0, new_size.width(), new_size.height()))
        # Обновляем линии
        self.update_lines()
        self.update_points()

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

    def add_points(self):
        """Добавляет кнопки на сцену."""
        for i, pos in enumerate(self.points_pos):
            x, y = pos
            point = ClickablePoint(x, y, i)  # x, y, радиус, индекс
            point.set_on_click(self.action_button)

            self.scene.addItem(point)
            self.points.append(point)

        # Обновляем позиции кнопок
        self.update_points()

    def update_points(self):
        """Обновляет позиции кнопок на основе текущего размера сцены."""
        scene_width = self.scene.width()
        scene_height = self.scene.height()

        for point in self.points:
            # Устанавливаем позицию кнопки (смещение по x, y)
            point.updatePositionAndRadius(scene_width, scene_height)
