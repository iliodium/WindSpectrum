from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel


class ImageLabel(QLabel):
    def __init__(
            self,
            image,
            parent=None
    ):
        super().__init__(parent)
        pixmap = QPixmap(image)
        self.original_pixmap = pixmap
        self.setPixmap(self.original_pixmap)

    def resizeEvent(
            self,
            event
    ):
        # Масштабируем изображение при изменении размеров виджета
        scaled_pixmap = self.original_pixmap.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled_pixmap)
        super().resizeEvent(event)
