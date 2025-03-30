# coding:utf-8
import asyncio

import numpy as np
from qfluentwidgets import ComboBox
from src.submodules.databasetoolkit.isolated_low_rise_roof_gable import (load_coordinates,
                                                                         load_face_number,
                                                                         load_id_coordinates_and_surface,
                                                                         load_id_wind_azimuth,
                                                                         load_pressure_coefficients,)
from src.submodules.plot.plotRoof import PlotRoof
from src.submodules.utils.data_features import lambdas
from src.submodules.utils.scaling import get_model_isolated_low_rise_roof_gable
from src.ui.common.ChartMode import ChartMode
from src.ui.common.ChartType import ChartType
from src.ui.common.IsofieldsType import IsofieldsType
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

        return int(id_wind_azimuth)

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
            load_pressure_coefficients(id_eave, id_wind_azimuth, breadth, depth, height, engine, db_url_server))[id_wind_azimuth]
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

    def _get_id_coordinates_and_surface(
            self
    ):
        model_size = self._get_model_size()
        breadth, depth, height = get_model_isolated_low_rise_roof_gable(model_size[1], model_size[2])
        angle = self._get_angle()
        id_wind_azimuth = self._get_id_wind_azimuth(angle)
        id_eave = self._get_id_eave()
        id_x_coordinates, id_y_coordinates, id_surface = asyncio.run(
            load_id_coordinates_and_surface(id_eave, id_wind_azimuth, breadth, depth, height,
                                            self.engine, self.DB_URL_SERVER)
        )

        return id_x_coordinates, id_y_coordinates, id_surface

    def _get_face_number(
            self
    ):
        _, _, id_surface = self._get_id_coordinates_and_surface()
        face_number = asyncio.run(load_face_number(id_surface, self.engine, self.DB_URL_SERVER))

        return face_number

    def _get_coordinates(
            self
    ):
        id_x_coordinates, id_y_coordinates, _ = self._get_id_coordinates_and_surface()

        x, z = asyncio.run(
            load_coordinates(id_x_coordinates, id_y_coordinates, self.engine, self.DB_URL_SERVER)
        )

        return x, z

    def _get_size_and_count_sensors(
            self,
            *args,
            **kwargs
    ):
        model_size = self._get_model_size()
        breadth, depth, height = get_model_isolated_low_rise_roof_gable(model_size[1], model_size[2])
        return breadth, depth, height

    def _plot_isofields(
            self,
            model_size,
            size,
            face_number,
            parameter,
            _pressure_coefficients,
            _coordinates
    ):
        x = [[]]
        z = [[]]
        pressure_coefficients = [[]]
        _pressure_coefficients = lambdas[parameter](_pressure_coefficients)


        for _ in range(len(set(face_number))):
            x.append([])
            z.append([])
            pressure_coefficients.append([])

        for ind, face in enumerate(face_number):
            x[face].append(_coordinates[0][ind])
            z[face].append(_coordinates[1][ind])
            pressure_coefficients[face].append(_pressure_coefficients[ind])

        match self.ComboBoxRoofType.text():
            case RoofGableType.A | RoofGableType.B:
                fig = PlotRoof.isofields_coefficients_A_B([x,z],
                                                          pressure_coefficients,
                                                          size,
                                                          )
            case RoofGableType.O:
                fig = PlotRoof.isofields_coefficients_O([x, z],
                                                          pressure_coefficients,
                                                          size,
                                                          )

            case RoofGableType.C:
                fig = PlotRoof.isofields_coefficients_C([x, z],
                                                          pressure_coefficients,
                                                          size,
                                                          )

        return fig

    def plot_isofields(
            self
    ):
        pressure_coefficients = self._get_pressure_coefficients()
        coordinates = self._get_coordinates()
        size = self._get_size_and_count_sensors()
        model_size = self._get_model_size()
        parameter = ChartMode(self.isofieldsParameters.currentText())
        face_number = self._get_face_number()
        fig = self._plot_isofields(model_size, size,face_number, parameter, pressure_coefficients, coordinates)

        self.add_plot_on_screen(fig, ChartType.ISOFIELDS)
