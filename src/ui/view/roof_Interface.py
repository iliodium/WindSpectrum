from PySide6.QtWidgets import QHBoxLayout
from qfluentwidgets import (ComboBox,
                            LineEdit,
                            PushButton,
                            StrongBodyLabel,)
from src.ui.common.Buttons import Buttons
from src.ui.view.Interface import Interface


class RoofInterface(Interface):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self._init_roof_information()

    def _init_general_information(
            self,
            roof_type
    ):
        super()._init_general_information()
        # Roof angle
        self.hBoxLayoutRoofAngle = QHBoxLayout(self.view)
        self.hBoxLayoutRoofAngle.addWidget(StrongBodyLabel(Buttons.ROOF_ANGLE))
        # Create text input widget
        self.lineEditRoofAngle = LineEdit()
        # Set default text
        self.lineEditRoofAngle.setText(self.tr('0'))
        # Set clear button
        self.lineEditRoofAngle.setClearButtonEnabled(True)
        self.lineEditRoofAngle.setFixedWidth(75)
        # Add text input widget to horizontal box layout
        self.hBoxLayoutRoofAngle.addWidget(self.lineEditRoofAngle)
        self.vBoxLayoutGenInf.insertLayout(5, self.hBoxLayoutRoofAngle)

        hBoxLayoutRoofType = QHBoxLayout(self.view)
        hBoxLayoutRoofType.setContentsMargins(0, 0, 0, 0)
        hBoxLayoutRoofType.setSpacing(0)
        PushButtonRoofType = PushButton(Buttons.ROOF_TYPE)
        PushButtonRoofType.setFixedWidth(100)
        hBoxLayoutRoofType.addWidget(PushButtonRoofType)
        hBoxLayoutRoofType.addStretch()  # растягивающее пространство

        # Create combo box
        self.ComboBoxRoofType = ComboBox()
        # Fill the combo box
        self.ComboBoxRoofType.addItems([
            self.tr(i) for i in roof_type
        ])
        # set fixed width of combobox
        self.ComboBoxRoofType.setFixedWidth(125)
        hBoxLayoutRoofType.addWidget(self.ComboBoxRoofType)

        self.vBoxLayoutGenInf.insertLayout(6, hBoxLayoutRoofType)

    def _init_chart_menu(
            self
    ):
        super()._init_chart_menu()

        for index in range(1, self.ComboBoxChartMenu.count())[::-1]:
            self.ComboBoxChartMenu.removeItem(index)

    def _init_roof_information(
            self
    ):
        pass

    def _get_angle(
            self
    ):
        return int(self.ComboBoxWindAngle.text())
