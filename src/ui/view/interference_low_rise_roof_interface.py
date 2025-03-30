# coding:utf-8

from qfluentwidgets import PushButton
from src.ui.common.Buttons import Buttons
from src.ui.common.RoofType import RoofType
from src.ui.view.interference_interface import InterferenceInterface
from src.ui.view.roof_Interface import RoofInterface


class InterferenceLowRiseRoofInterface(InterferenceInterface, RoofInterface):
    SAMPLE_PERIOD = 7.5
    SAMPLE_FREQUENCY = 781
    NUMBER_OF_TIME_COUNTS = 5858

    REPORT_FOLDER_NAME = "Интерференция кровли низкоэтажного зданий"

    def _init_general_information(
            self
    ):
        # InterferenceInterface._init_general_information(self)
        RoofInterface._init_general_information(self, RoofType)

        self.PushButtonArrangementTypeOfInterferingBuildingsInformation = PushButton(
            Buttons.ARRANGEMENT_TYPE_OF_INTERFERING_BUILDINGS)
        self.vBoxLayoutGenInf.addWidget(self.PushButtonArrangementTypeOfInterferingBuildingsInformation)

        self.PushButtonInterferingBuildingsDensityInformation = PushButton(Buttons.INTERFERING_BUILDINGS_DENSITY)
        self.vBoxLayoutGenInf.addWidget(self.PushButtonInterferingBuildingsDensityInformation)

        self.generalInformationContainer.setFixedHeight(600)
