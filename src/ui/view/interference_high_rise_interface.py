# coding:utf-8
import asyncio
from concurrent.futures import (ProcessPoolExecutor,
                                as_completed,)

import numpy as np
from PySide6.QtWidgets import (QHBoxLayout,
                               QVBoxLayout,
                               QWidget,)
from qfluentwidgets import (LineEdit,
                            PushButton,
                            StrongBodyLabel,)
from sqlalchemy import create_engine
from src.common.annotation import ModelSizeType
from src.submodules.databasetoolkit.interference import (find_id_building_by_height,
                                                         load_pressure_coefficients,)
from src.submodules.utils.scaling import (
    get_model_and_scale_factors_interference,)
from src.ui.common.Buttons import Buttons
from src.ui.components.ImageLabel import ImageLabel
from src.ui.view.building_Interface import BuildingInterface
from src.ui.view.interference_interface import InterferenceInterface


class InterferenceHighRiseBuildingInterface(InterferenceInterface, BuildingInterface):
    """Interference High Rise Interface

    constants :

    number of time counts 5 858
    sample_frequency 781
    sample_period 7.5
    turbulence_intensity 20
    mean_wind_speed 8.2
    principal_building (мм) 70 70 280

    """

    SAMPLE_PERIOD = 7.5
    SAMPLE_FREQUENCY = 781
    NUMBER_OF_TIME_COUNTS = 5858

    REPORT_FOLDER_NAME = "Интерференция высокоэтажных зданий"

    def __init__(
            self,
            parent=None,
            config=None
    ):
        super().__init__(parent=parent, config=config)

        self._init_interfering_position_overview()

        self.previousScreenFlag = False

    def _init_interfering_position_overview(
            self
    ):
        WidgetInterferingInformation = QWidget()
        self.vBoxLayoutSensorsOverview = QVBoxLayout(WidgetInterferingInformation)

        self.image_label = ImageLabel('src/ui/resource/images/Building_arrangments.JPEG')
        self.vBoxLayoutSensorsOverview.addWidget(self.image_label)

        self.StackedLayoutMainMenu.addWidget(WidgetInterferingInformation)

    def _init_general_information(
            self
    ):
        super()._init_general_information()
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

        if self.previousScreenFlag:
            self.PushButtonInterferingInformation.setText(Buttons.INTERFERING_INFORMATION)
            self.StackedLayoutMainMenu.setCurrentIndex(0)
            self.previousScreenFlag = False

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
                self.previousScreenFlag = True

            case 2:
                self.PushButtonSensorsOverview.setText(Buttons.SENSORS)
                self.PushButtonInterferingInformation.setText(Buttons.INTERFERING_INFORMATION)
                self.StackedLayoutMainMenu.setCurrentIndex(0)
                self.previousScreenFlag = False

    def _get_model_size_interfering(
            self
    ) -> ModelSizeType:
        return tuple(map(float, self.lineEditBuildingSizeInterfering.text().replace(',', '.').split(' ')))

    def _get_position_interfering(
            self
    ) -> int:
        return int(self.lineEditPositionInterfering.text())

    @staticmethod
    def _get_pressure_coefficients_for_definition_angle_future(
            url,
            db_url_server,
            *args
    ):
        engine = create_engine(url)

        return InterferenceHighRiseBuildingInterface._get_pressure_coefficients_for_definition_angle(
            engine, db_url_server, *args)

    @staticmethod
    def _get_pressure_coefficients_for_definition_angle(
            engine,
            db_url_server,
            position,
            angle,
            id_interfering_building
    ):
        pressure_coefficients = asyncio.run(
            load_pressure_coefficients(position, angle, id_interfering_building, engine, db_url_server))[angle]

        return pressure_coefficients

    def _get_coordinates(
            self
    ):
        x = np.array(
            [3, 13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 113, 123, 133, 143, 153, 163, 173, 183, 193, 203, 213, 223,
             233, 243, 253, 263, 273, 3, 13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 113, 123, 133, 143, 153, 163, 173,
             183, 193, 203, 213, 223, 233, 243, 253, 263, 273, 3, 13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 113, 123,
             133, 143, 153, 163, 173, 183, 193, 203, 213, 223, 233, 243, 253, 263, 273, 3, 13, 23, 33, 43, 53, 63, 73,
             83, 93, 103, 113, 123, 133, 143, 153, 163, 173, 183, 193, 203, 213, 223, 233, 243, 253, 263, 273, 3, 13,
             23, 33, 43, 53, 63, 73, 83, 93, 103, 113, 123, 133, 143, 153, 163, 173, 183, 193, 203, 213, 223, 233, 243,
             253, 263, 273, 3, 13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 113, 123, 133, 143, 153, 163, 173, 183, 193,
             203, 213, 223, 233, 243, 253, 263, 273, 3, 13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 113, 123, 133, 143,
             153, 163, 173, 183, 193, 203, 213, 223, 233, 243, 253, 263, 273, 3, 13, 23, 33, 43, 53, 63, 73, 83, 93,
             103, 113, 123, 133, 143, 153, 163, 173, 183, 193, 203, 213, 223, 233, 243, 253, 263, 273, 3, 13, 23, 33,
             43, 53, 63, 73, 83, 93, 103, 113, 123, 133, 143, 153, 163, 173, 183, 193, 203, 213, 223, 233, 243, 253,
             263, 273]) / 1000

        z = np.array(
            [275, 275, 275, 275, 275, 275, 275, 275, 275, 275, 275, 275, 275, 275, 275, 275, 275, 275, 275, 275, 275,
             275, 275, 275, 275, 275, 275, 275, 260, 260, 260, 260, 260, 260, 260, 260, 260, 260, 260, 260, 260, 260,
             260, 260, 260, 260, 260, 260, 260, 260, 260, 260, 260, 260, 260, 260, 225, 225, 225, 225, 225, 225, 225,
             225, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225, 225,
             190, 190, 190, 190, 190, 190, 190, 190, 190, 190, 190, 190, 190, 190, 190, 190, 190, 190, 190, 190, 190,
             190, 190, 190, 190, 190, 190, 190, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155,
             155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 155, 120, 120, 120, 120, 120, 120, 120,
             120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120,
             85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85,
             85, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
             50, 50, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
             15, 15, 15]
        ) / 1000

        return x, z

    def _get_angle_border(
            self
    ):

        return 355

    def _get_tpu_size_principal_building(
            self
    ):
        return 0.07, 0.07, 0.28

    def _get_count_sensors(
            self
    ):
        count_sensors_on_model = 252
        count_sensors_on_middle_row = 7
        count_sensors_on_side_row = 7

        return count_sensors_on_model, count_sensors_on_middle_row, count_sensors_on_side_row

    def _get_face_number(
            self
    ):
        face_number = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1,
                       1, 1, 1, 2, 2,
                       2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                       2, 3, 3, 3, 3,
                       3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4,
                       4, 4, 4, 4, 4,
                       4, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1,
                       1, 1, 1, 1, 2,
                       2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                       2, 2, 3, 3, 3,
                       3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
                       4, 4, 4, 4, 4,
                       4, 4, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4]

        return face_number

    def _get_size_tpu(
            self
    ):
        size = self._get_tpu_size_principal_building()

        return size

    def _get_size_and_count_sensors(
            self,
            *args,
            **kwargs
    ):
        count_sensors = self._get_count_sensors()
        size = self._get_tpu_size_principal_building()

        return size, count_sensors

    def _get_pressure_coefficients(
            self
    ):
        angle = self._get_angle()
        position = self._get_position_interfering()
        id_interfering_building = self._get_id_interfering_building()

        pressure_coefficients = self._get_pressure_coefficients_for_definition_angle(self.engine,
                                                                                     self.DB_URL_SERVER,
                                                                                     position,
                                                                                     angle,
                                                                                     id_interfering_building)

        return pressure_coefficients

    def _get_id_interfering_building(
            self
    ):
        model_size_interfering = self._get_model_size_interfering()
        position = self._get_position_interfering()
        height = get_model_and_scale_factors_interference(*model_size_interfering, position)
        id_interfering_building = asyncio.run(find_id_building_by_height(height, self.engine, self.DB_URL_SERVER))

        return id_interfering_building

    def _get_pressure_coefficients_storage(
            self
    ):
        pressure_coefficients_storage = {}
        position = self._get_position_interfering()
        id_interfering_building = self._get_id_interfering_building()

        angle_border = 355
        with ProcessPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = {}
            for angle in range(0, angle_border + 5, 5):
                args = (position, angle, id_interfering_building)
                future = executor.submit(
                    self._get_pressure_coefficients_for_definition_angle_future,
                    self.DB_URL_LOCAL, self.DB_URL_SERVER,
                    *args
                )
                futures[future] = angle

            for future in as_completed(futures):
                try:
                    result = future.result()
                    angle = futures[future]
                    pressure_coefficients_storage[angle] = result
                except Exception as e:
                    print(f"Ошибка в задаче: {e}")

        return pressure_coefficients_storage
