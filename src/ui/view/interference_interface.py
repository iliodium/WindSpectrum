# coding:utf-8

from PySide6.QtWidgets import QHBoxLayout
from qfluentwidgets import (LineEdit,
                            StrongBodyLabel,)
from src.ui.common.Buttons import Buttons
from src.ui.view.building_Interface import BuildingInterface
from src.ui.view.Interface import Interface


class InterferenceInterface(Interface):
    def _init_general_information(
            self
    ):
        super()._init_general_information()
        # Building size Interfering
        self.hBoxLayoutBuildingSizeInterfering = QHBoxLayout(self.view)
        self.hBoxLayoutBuildingSizeInterfering.addWidget(StrongBodyLabel(Buttons.INTERFERING_SIZE))
        self.lineEditBuildingSizeInterfering = LineEdit()
        self.lineEditBuildingSizeInterfering.setText(self.tr('10 10 20'))
        self.lineEditBuildingSizeInterfering.setClearButtonEnabled(True)
        self.lineEditBuildingSizeInterfering.setFixedWidth(125)
        self.hBoxLayoutBuildingSizeInterfering.addWidget(self.lineEditBuildingSizeInterfering)
        self.vBoxLayoutGenInf.insertLayout(5, self.hBoxLayoutBuildingSizeInterfering)
