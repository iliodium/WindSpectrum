# coding:utf-8
import asyncio

from qfluentwidgets import ComboBox
from src.submodules.databasetoolkit.isolated_low_rise_roof_gable import (load_id_wind_azimuth,
                                                                         load_pressure_coefficients,)
from src.submodules.utils.scaling import get_model_isolated_low_rise_roof_gable
from src.ui.common.RoofGableType import RoofGableType
from src.ui.view.roof_Interface import RoofInterface


class IsolatedLowRiseRoofGableInterface(RoofInterface):
    SAMPLE_PERIOD = 600
    SAMPLE_FREQUENCY = 23.4375
    NUMBER_OF_TIME_COUNTS = 14063

    REPORT_FOLDER_NAME = None

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

    def _get_id_wind_azimuth(
            self,
            angle
    ):
        id_wind_azimuth = asyncio.run(load_id_wind_azimuth(angle, self.engine, self.DB_URL_SERVER))

        return id_wind_azimuth

    def _get_id_eave(
            self
    ):
        return {'A': 1,
                'O': 2,
                'C': 3,
                'B': 4,
                }[self.ComboBoxRoofType.text()]

    @staticmethod
    def _get_pressure_coefficients_for_definition_angle(
            engine,
            db_url_server,
            id_eave,
            id_wind_azimuth,
            breadth,
            depth,
            height,
    ):
        pressure_coefficients = asyncio.run(
            load_pressure_coefficients(id_eave, id_wind_azimuth, breadth, depth, height, engine, db_url_server))
        return pressure_coefficients

    def _get_pressure_coefficients(
            self
    ):
        model_size = self._get_model_size()
        breadth, depth, height = get_model_isolated_low_rise_roof_gable(model_size[1], model_size[2])
        angle = self._get_angle()
        id_wind_azimuth = self._get_id_wind_azimuth(angle)
        id_eave = self._get_id_eave()

        pressure_coefficients = self._get_pressure_coefficients_for_definition_angle(self.engine,
                                                                                     self.DB_URL_SERVER,
                                                                                     id_eave,
                                                                                     id_wind_azimuth,
                                                                                     breadth,
                                                                                     depth,
                                                                                     height,
                                                                                     )

        return pressure_coefficients
