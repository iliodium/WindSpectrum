# coding:utf-8
import asyncio

import numpy as np
from qfluentwidgets import (ComboBox,
                            PushButton,)
from src.submodules.databasetoolkit.isolated_low_rise_roof import (load_coordinates,
                                                                   load_face_number,
                                                                   load_id_coordinates_and_surface,
                                                                   load_id_roof_angle,
                                                                   load_pressure_coefficients,
                                                                   load_triplets,)
from src.submodules.plot.plotRoof import PlotRoof
from src.submodules.utils.scaling import get_model_isolated_low_rise_roof
from src.ui.common.Buttons import Buttons
from src.ui.common.ChartMode import ChartMode
from src.ui.common.ChartType import ChartType
from src.ui.common.RoofGableType import RoofGableType
from src.ui.common.RoofType import RoofType
from src.ui.view.roof_Interface import RoofInterface


class IsolatedLowRiseRoofInterface(RoofInterface):
    SAMPLE_PERIOD = 15
    SAMPLE_FREQUENCY = 600
    NUMBER_OF_TIME_COUNTS = 9000

    REPORT_FOLDER_NAME = "Интерференция кровли низкоэтажного зданий"

    def _init_general_information(
            self
    ):
        super()._init_general_information(RoofType)

        self.ComboBoxWindAngle = ComboBox()
        self.ComboBoxWindAngle.addItems([
            self.tr(str(i)) for i in range(0, 91, 15)
        ])
        self.ComboBoxWindAngle.setFixedWidth(75)

        index = self.hBoxLayoutWindAngle.indexOf(self.lineEditWindAngle)
        self.hBoxLayoutWindAngle.removeWidget(self.lineEditWindAngle)
        self.lineEditWindAngle.deleteLater()
        self.hBoxLayoutWindAngle.insertWidget(index, self.ComboBoxWindAngle)

        self.generalInformationContainer.setFixedHeight(325)

    def _get_angle_roof(
            self
    ):
        return int(self.ComboBoxRoofAngle.text())

    def _get_id_angle_roof(
            self
    ):
        angle = self._get_angle_roof()
        return asyncio.run(load_id_roof_angle(angle, self.engine, self.DB_URL_SERVER))

    def _get_size_and_count_sensors(
            self,
            *args,
            **kwargs
    ):
        model_size = self._get_model_size()
        triplets = asyncio.run(load_triplets(self._get_id_roof(),self.DB_URL_SERVER))
        breadth, depth, height = get_model_isolated_low_rise_roof(model_size, triplets)
        return breadth, depth, height

    def _get_id_roof(
            self
    ):
        return {RoofType.FLAT: 1,
                RoofType.GABLE: 2,
                RoofType.HIP: 3,
                }[RoofType(self.ComboBoxRoofType.text())]

    @staticmethod
    def _get_pressure_coefficients_for_definition_angle(
            engine,
            db_url_server,
            id_roof,
            angle,
            id_angle_roof,
            breadth,
            depth,
            height,
    ):
        pressure_coefficients = asyncio.run(
            load_pressure_coefficients(id_roof, angle, id_angle_roof, breadth, depth, height, engine, db_url_server))[
            angle]

        return pressure_coefficients

    def _get_pressure_coefficients(
            self
    ):
        breadth, depth, height = self._get_size_and_count_sensors()
        match self.ComboBoxRoofType.text():
            case RoofType.FLAT:
                angle = 0
                id_angle_roof = 1
            case _:
                angle = self._get_angle()
                id_angle_roof = self._get_id_angle_roof()

        id_roof = self._get_id_roof()
        print(id_roof, id_angle_roof, angle,breadth,depth,height)
        pressure_coefficients = self._get_pressure_coefficients_for_definition_angle(self.engine,
                                                                                     self.DB_URL_SERVER,
                                                                                     id_roof,
                                                                                     angle,
                                                                                     id_angle_roof,
                                                                                     breadth,
                                                                                     depth,
                                                                                     height,
                                                                                     )

        return pressure_coefficients

    def _get_id_coordinates_and_surface(
            self
    ):
        breadth, depth, height = self._get_size_and_count_sensors()
        match self.ComboBoxRoofType.text():
            case RoofType.FLAT:
                angle = 0
                id_angle_roof = 1
            case _:
                angle = self._get_angle()
                id_angle_roof = self._get_id_angle_roof()

        id_roof = self._get_id_roof()

        id_x_coordinates, id_y_coordinates, id_surface = asyncio.run(
            load_id_coordinates_and_surface(angle, id_roof, id_angle_roof, breadth, depth, height,
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
        _pressure_coefficients = np.mean(_pressure_coefficients, axis=0)

        for _ in range(len(set(face_number))):
            x.append([])
            z.append([])
            pressure_coefficients.append([])

        for ind, face in enumerate(face_number):
            x[face].append(_coordinates[0][ind])
            z[face].append(_coordinates[1][ind])
            pressure_coefficients[face].append(_pressure_coefficients[ind])

        match self.ComboBoxRoofType.text():
            case RoofType.FLAT:
                fig = PlotRoof.isofields_coefficients_flat_roof([x, z],
                                                                pressure_coefficients,
                                                                size,
                                                                )
            case RoofType.GABLE:
                fig = PlotRoof.isofields_coefficients_gable_roof([x, z],
                                                                 pressure_coefficients,
                                                                 size,
                                                                 self._get_angle_roof()
                                                                 )

            case RoofType.HIP:
                fig = PlotRoof.isofields_coefficients_hip_roof([x, z],
                                                               pressure_coefficients,
                                                               size,
                                                               self._get_angle_roof()
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
        fig = self._plot_isofields(model_size, size, face_number, parameter, pressure_coefficients, coordinates)

        self.add_plot_on_screen(fig, ChartType.ISOFIELDS)
