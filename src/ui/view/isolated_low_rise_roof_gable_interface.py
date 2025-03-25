# coding:utf-8
from qfluentwidgets import ComboBox
from src.ui.common.RoofGableType import RoofGableType
from src.ui.view.roof_Interface import RoofInterface


class IsolatedLowRiseRoofGableInterface(RoofInterface):
    SAMPLE_PERIOD = 7.5
    SAMPLE_FREQUENCY = 781
    NUMBER_OF_TIME_COUNTS = 5858

    REPORT_FOLDER_NAME = "Интерференция кровли низкоэтажного зданий"

    def _init_general_information(
            self
    ):
        super()._init_general_information(RoofGableType)

        self._remove_layout_from_layout(self.vBoxLayoutGenInf, self.hBoxLayoutRoofAngle)

        self.ComboBoxWindAngle = ComboBox()
        self.ComboBoxWindAngle.addItems([
            self.tr(str(i)) for i in [0, 23, 45, 68, 90]
        ])
        self.ComboBoxWindAngle.setFixedWidth(75)

        index = self.hBoxLayoutWindAngle.indexOf(self.lineEditWindAngle)
        self.hBoxLayoutWindAngle.removeWidget(self.lineEditWindAngle)
        self.lineEditWindAngle.deleteLater()
        self.hBoxLayoutWindAngle.insertWidget(index, self.ComboBoxWindAngle)
