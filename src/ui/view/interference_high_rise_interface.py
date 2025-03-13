# coding:utf-8
import asyncio
import os

import matplotlib
import numpy as np
import scipy
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QWidget, QVBoxLayout, QLabel
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Mm
from matplotlib import pyplot as plt
from openpyxl import Workbook
from qfluentwidgets import PushButton, LineEdit, StrongBodyLabel, TitleLabel, ComboBox

from compiled_functions import aot_calculations
from src.common.PermutationView import PermutationView
from src.common.TypeOfBasement import TypeOfBasement
from src.common.constants import Uz_a_0_16_x, Uz_a_0_16_z, Uz_a_0_25_x, Uz_a_0_25_z, wind_regions, alpha_standards
from src.submodules.databasetoolkit.isolated import load_pressure_coefficients, find_experiment_by_model_name, \
    load_positions, load_face_number
from src.submodules.plot.plotBuilding import PlotBuilding
from src.submodules.plot.utils import scaling_data
from src.submodules.report_tools.reportFolder import ReportFolder
from src.submodules.report_tools.utils import create_directory_to_report
from src.submodules.report_tools.wordBuilder import WordBuilder
from src.submodules.utils.angle import get_angle_border, get_base_angle, changer_sequence_coefficients
from src.submodules.utils.data_features import lambdas, calculated, warranty_plus, warranty_minus, polar_lambdas
from src.submodules.utils.permutations import get_view_permutation_data, get_sequence_permutation_data
from src.submodules.utils.scaling import get_model_and_scale_factors
from src.submodules.utils.speed_sp import speed_sp_region
from src.submodules.utils.utils import converter_coordinates, get_size_and_count_sensors, tpu_size_to_real
from src.ui.common.Buttons import Buttons
from src.ui.common.ChartMode import ChartMode
from src.ui.common.ChartType import ChartType
from src.ui.common.CoordinateSystem import CoordinateSystem
from src.ui.common.IsofieldsType import IsofieldsType
from src.ui.components.ImageLabel import ImageLabel
from src.ui.view.interface import Interface


from PySide6.QtGui import QPixmap
class InterferenceHighRiseInterface(Interface):
    """Interference High Rise Interface

    constants :

    sample_frequency 781
    sample_period 7.5
    turbulence_intensity 20
    mean_wind_speed 8.2
    principal_building (мм) 70 70 280

    """

    def __init__(
            self,
            parent=None,
            engine=None
    ):

        super().__init__(parent=parent, engine=engine)
        self.setObjectName('InterferenceHighRiseInterface')

        WidgetInterferingInformation = QWidget()
        self.vBoxLayoutSensorsOverview = QVBoxLayout(WidgetInterferingInformation)

        self.image_label = ImageLabel('src/ui/resource/images/Building_arrangments.JPEG')
        self.vBoxLayoutSensorsOverview.addWidget(self.image_label)


        self.StackedLayoutMainMenu.addWidget(WidgetInterferingInformation)

        self.previousIndex = False

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

        self.hBoxLayoutPositionInterfering = QHBoxLayout(self.view)
        self.hBoxLayoutPositionInterfering.addWidget(StrongBodyLabel(Buttons.INTERFERING_POSITION))
        self.lineEditPositionInterfering = LineEdit()
        self.lineEditPositionInterfering.setText(self.tr('15'))
        self.lineEditPositionInterfering.setClearButtonEnabled(True)
        self.lineEditPositionInterfering.setFixedWidth(75)
        self.hBoxLayoutPositionInterfering.addWidget(self.lineEditPositionInterfering)
        self.vBoxLayoutGenInf.insertLayout(6, self.hBoxLayoutPositionInterfering)

        self.PushButtonInterferingInformation = PushButton(Buttons.INTERFERING_INFORMATION)
        self.PushButtonInterferingInformation.clicked.connect(self._switch_stacked_layout_interfering_information)
        self.vBoxLayoutGenInf.addWidget(self.PushButtonInterferingInformation)

        self.generalInformationContainer.setFixedHeight(400)

    def _switch_stacked_layout_sensors_overview(
            self
    ):

        if self.previousIndex:
            self.PushButtonInterferingInformation.setText(Buttons.INTERFERING_INFORMATION)
            self.StackedLayoutMainMenu.setCurrentIndex(0)
            self.previousIndex = False

        super()._switch_stacked_layout_sensors_overview()

    def _switch_stacked_layout_interfering_information(
            self
    ):
        current_index = self.StackedLayoutMainMenu.currentIndex()

        match current_index:
            case 0 | 1:
                self.StackedLayoutMainMenu.setCurrentIndex(2)
                self.PushButtonSensorsOverview.setText(Buttons.SENSORS)
                self.PushButtonInterferingInformation.setText(Buttons.PLOTS)
                self.previousIndex = True

            case 2:
                self.PushButtonSensorsOverview.setText(Buttons.SENSORS)
                self.PushButtonInterferingInformation.setText(Buttons.INTERFERING_INFORMATION)
                self.StackedLayoutMainMenu.setCurrentIndex(0)
                self.previousIndex = False



    def get_pressure_coefficients_for_the_sensor(
            self,
            model_id,
            alpha,
            angle,
            model_name: str,
            sensor_id
    ):
        pressure_coefficients = self.get_pressure_coefficients(model_id, alpha, angle, str(model_name))
        return pressure_coefficients[:, sensor_id]

    def get_pressure_coefficients(
            self,
            model_id,
            alpha,
            angle,
            model_name: str
    ):
        model_name_base = model_name
        turn_flag = False
        if model_name[0] == model_name[1]:
            angle_border = 45
            type_base = TypeOfBasement.SQUARE

        else:
            angle_border = 90
            type_base = TypeOfBasement.RECTANGLE

        if model_name[1] in ['2', '3']:
            model_name_base = model_name[1] + model_name[0] + model_name[2]
            angle = str((int(angle) + 270) % 360)
            turn_flag = True

        # Поворот данных для отображения углов, выходящих за границы имеющихся
        if int(angle) > angle_border:
            permutation_view = get_view_permutation_data(type_base, int(angle))  # вид последовательности данных
            base_angle = get_base_angle(int(angle), permutation_view, type_base)
            sequence_permutation = get_sequence_permutation_data(type_base, permutation_view, int(angle))

            pressure_coefficients = asyncio.run(load_pressure_coefficients(model_id, alpha, self.engine,
                                                                           angle=base_angle))[base_angle]

            pressure_coefficients = changer_sequence_coefficients(pressure_coefficients, permutation_view,
                                                                  model_name, sequence_permutation)
        else:
            pressure_coefficients = asyncio.run(load_pressure_coefficients(model_id, alpha, self.engine,
                                                                           angle=angle))[angle]

        # Поворот модели
        if turn_flag:
            if int(angle) % 90 != 0:
                model_name_base = model_name
            pressure_coefficients = changer_sequence_coefficients(pressure_coefficients, PermutationView.FORWARD,
                                                                  model_name_base,
                                                                  (3, 0, 1, 2))
        return pressure_coefficients

    def get_model_id(
            self,
            model_name,
            alpha
    ):
        model_name_str = str(model_name)
        if model_name_str[1] in ['2', '3']:
            model_name = int(model_name_str[1] + model_name_str[0] + model_name_str[2])

        model_id = asyncio.run(find_experiment_by_model_name(model_name, alpha, self.engine)).model_id

        return model_id

    def get_coordinates(
            self,
            model_id,
            alpha
    ):
        coordinates = asyncio.run(load_positions(model_id, alpha, self.engine))

        return coordinates

    def get_face_number(
            self,
            model_id,
            alpha
    ):
        face_number = asyncio.run(load_face_number(model_id, alpha, self.engine))

        return face_number
