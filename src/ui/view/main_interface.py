# coding:utf-8
from PySide6.QtCore import (QRectF,
                            Qt,)
from PySide6.QtGui import (QBrush,
                           QColor,
                           QLinearGradient,
                           QPainter,
                           QPainterPath,
                           QPixmap,)
from PySide6.QtWidgets import (QLabel,
                               QVBoxLayout,
                               QWidget,)
from qfluentwidgets import (FluentIcon,
                            ScrollArea,
                            isDarkTheme,)
from src.ui.common.StyleSheet import StyleSheet
from src.ui.components.LinkCard import LinkCardView
from src.ui.config.config import REPO_URL


class BannerWidget(QWidget):
    """ Banner widget """

    def __init__(
            self,
            parent=None
    ):
        super().__init__(parent=parent)
        self.setFixedHeight(336)

        self.vBoxLayout = QVBoxLayout(self)
        self.galleryLabel = QLabel('Wind Spectrum', self)
        self.banner = QPixmap('src/ui/resource/images/header.png')
        self.linkCardView = LinkCardView(self)

        self.galleryLabel.setObjectName('galleryLabel')

        self.vBoxLayout.setSpacing(0)
        self.vBoxLayout.setContentsMargins(0, 20, 0, 0)
        self.vBoxLayout.addWidget(self.galleryLabel)
        self.vBoxLayout.addWidget(self.linkCardView, 1, Qt.AlignBottom)
        self.vBoxLayout.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.linkCardView.addCard(
            FluentIcon.GITHUB,
            self.tr('GitHub'),
            self.tr(
                'Актуальная версия\nWind Spectrum'),
            REPO_URL
        )

    def paintEvent(
            self,
            e
    ):
        super().paintEvent(e)
        painter = QPainter(self)
        painter.setRenderHints(
            QPainter.SmoothPixmapTransform | QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)

        path = QPainterPath()
        path.setFillRule(Qt.WindingFill)
        w, h = self.width(), self.height()
        path.addRoundedRect(QRectF(0, 0, w, h), 10, 10)
        path.addRect(QRectF(0, h - 50, 50, 50))
        path.addRect(QRectF(w - 50, 0, 50, 50))
        path.addRect(QRectF(w - 50, h - 50, 50, 50))
        path = path.simplified()

        # init linear gradient effect
        gradient = QLinearGradient(0, 0, 0, h)

        # draw background color
        if not isDarkTheme():
            gradient.setColorAt(0, QColor(207, 216, 228, 255))
            gradient.setColorAt(1, QColor(207, 216, 228, 0))
        else:
            gradient.setColorAt(0, QColor(0, 0, 0, 255))
            gradient.setColorAt(1, QColor(0, 0, 0, 0))

        painter.fillPath(path, QBrush(gradient))

        # draw banner image
        pixmap = self.banner.scaled(
            self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        painter.fillPath(path, QBrush(pixmap))


class MainInterface(ScrollArea):
    """ Home interface """

    def __init__(
            self,
            parent=None
    ):
        super().__init__(parent=parent)
        self.banner = BannerWidget(self)
        self.view = QWidget(self)
        StyleSheet.MAIN_INTERFACE.apply(self)

        self.vBoxLayout = QVBoxLayout(self.view)

        self.__initWidget()

    def __initWidget(
            self
    ):
        self.view.setObjectName('view')
        self.setObjectName('MainInterface')
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidget(self.view)
        self.setWidgetResizable(True)

        self.vBoxLayout.setContentsMargins(0, 0, 0, 36)
        self.vBoxLayout.setSpacing(40)
        self.vBoxLayout.addWidget(self.banner)
        self.vBoxLayout.setAlignment(Qt.AlignTop)
