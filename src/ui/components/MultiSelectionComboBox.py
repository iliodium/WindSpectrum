from PySide6.QtCore import (QRect,
                            QRectF,
                            Qt,)
from PySide6.QtGui import (QCursor,
                           QPainter,)
from PySide6.QtWidgets import (QApplication,
                               QComboBox,
                               QListWidgetItem,
                               QVBoxLayout,
                               QWidget,)
from qfluentwidgets import (FluentIcon,
                            ListWidget,)
from src.ui.common.StyleSheet import StyleSheet


class MultiSelectionComboBox(QComboBox):
    def __init__(
            self,
            parent=None,
            placeholder_text=''
    ):
        super().__init__(parent)

        self.list_widget = ListWidget()
        self.setPlaceholderText(placeholder_text)
        self.setEditable(False)
        path = StyleSheet.path(StyleSheet.MULTI_SELECTION_COMBO_BOX)

        with open(path, 'r') as file:
            stylesheet = file.read()
            self.setStyleSheet(stylesheet)

        self.setModel(self.list_widget.model())
        self.setView(self.list_widget)

        self.setFixedWidth(120)

        self.list_widget.itemPressed.connect(self.onItemPressed)
        # to always have a placeholder text
        self.currentTextChanged.connect(lambda: self.setCurrentIndex(-1))
        # draw arrow
        # TODO the arrow is different from ComboBox arrow. make them the same
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        FluentIcon.CHEVRON_RIGHT.render(painter, QRectF(self.width() - 10, self.height() / 2 - 9 / 2, 9, 9))

    def paintEvent(self, e):
        super().paintEvent(e)
        painter = QPainter(self)
        painter.translate(self.width() - 20, 18)
        FluentIcon.ARROW_DOWN.render(painter, QRectF(-5, -5, 9.6, 9.6))

    def hidePopup(
            self
    ):
        list_widget_geometry = self.list_widget.geometry()
        list_widget_geometry_rect = list_widget_geometry.getRect()

        combobox_geometry = self.geometry()
        combobox_rect = combobox_geometry.getRect()

        cursor_pos = self.mapFromGlobal(QCursor.pos())
        # to make the widget close normally
        list_widget_geometry.setRect(0, 0, list_widget_geometry_rect[2],
                                     list_widget_geometry_rect[3] + combobox_rect[3])
        combobox_rect = QRect(0, 0, combobox_rect[2], combobox_rect[3])

        if not list_widget_geometry.contains(cursor_pos) or combobox_rect.contains(cursor_pos):
            super().hidePopup()

    def addItems(
            self,
            texts
    ):
        for t in texts:
            self.addItem(t)

    def addItem(
            self,
            text
    ):
        item = QListWidgetItem(text, self.list_widget)
        item.setCheckState(Qt.CheckState.Unchecked)

    def getSelected(
            self
    ):
        return [item.text() for item in self.list_widget.selectedItems()]

    def onItemPressed(
            self,
            item
    ):

        if item.checkState() == Qt.CheckState.Checked:
            item.setCheckState(Qt.CheckState.Unchecked)
        else:
            item.setCheckState(Qt.CheckState.Checked)

    def checkAllItems(self):
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            item.setCheckState(Qt.CheckState.Checked)

