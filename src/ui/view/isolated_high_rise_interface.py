# coding:utf-8
import asyncio
from concurrent.futures import (ProcessPoolExecutor,
                                as_completed,)

from sqlalchemy import create_engine

from src.common.PermutationView import PermutationView
from src.common.TypeOfBasement import TypeOfBasement
from src.submodules.databasetoolkit.isolated import (find_experiment_by_model_name,
                                                     load_face_number,
                                                     load_positions,
                                                     load_pressure_coefficients,)
from src.submodules.utils.angle import (changer_sequence_coefficients,
                                        get_angle_border,
                                        get_base_angle,)
from src.submodules.utils.permutations import (get_sequence_permutation_data,
                                               get_view_permutation_data,)
from src.submodules.utils.scaling import get_model_and_scale_factors
from src.submodules.utils.utils import get_size_tpu_and_count_sensors
from src.ui.view.interfaceBuildings import InterfaceBuildings


class IsolatedHighRiseInterfaceBuildings(InterfaceBuildings):
    """Isolated High Rise Interface

    constants :

    number of time counts 32 768
    sample_frequency 1000
    sample_period 32.768

    """
    SAMPLE_PERIOD = 32.768
    SAMPLE_FREQUENCY = 1000
    NUMBER_OF_TIME_COUNTS = 32768

    REPORT_FOLDER_NAME = "Изолированные высокоэтажные здания"

    def __init__(
            self,
            parent=None,
            config=None
    ):

        super().__init__(parent=parent, config=config)
        self.setObjectName('IsolatedHighRiseInterface')

    @staticmethod
    def _get_pressure_coefficients_for_definition_angle_future(
            url,
            db_url_server,
            *args
    ):
        engine = create_engine(url)

        return IsolatedHighRiseInterfaceBuildings._get_pressure_coefficients_for_definition_angle(
            engine, db_url_server, *args)

    @staticmethod
    def _get_pressure_coefficients_for_definition_angle(
            engine,
            db_url_server,
            model_name,
            angle,
            model_id,
            alpha
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

            pressure_coefficients = asyncio.run(load_pressure_coefficients(model_id, alpha, engine, db_url_server,
                                                                           angle=base_angle))[base_angle]

            pressure_coefficients = changer_sequence_coefficients(pressure_coefficients, permutation_view,
                                                                  model_name, sequence_permutation)
        else:
            pressure_coefficients = asyncio.run(load_pressure_coefficients(model_id, alpha, engine, db_url_server,
                                                                           angle=angle))[angle]

        # Поворот модели
        if turn_flag:
            if int(angle) % 90 != 0:
                model_name_base = model_name
            pressure_coefficients = changer_sequence_coefficients(pressure_coefficients, PermutationView.FORWARD,
                                                                  model_name_base,
                                                                  (3, 0, 1, 2))
        return pressure_coefficients

    def _get_pressure_coefficients(
            self
    ):
        alpha = self._get_alpha()
        model_size = self._get_model_size()
        model_name, _ = get_model_and_scale_factors(*model_size, alpha)
        angle = self._get_angle()

        model_id = self.get_model_id(model_name, alpha)

        model_name = str(model_name)

        pressure_coefficients = self._get_pressure_coefficients_for_definition_angle(self.engine,
                                                                                     self.DB_URL_SERVER,
                                                                                     model_name,
                                                                                     angle,
                                                                                     model_id,
                                                                                     alpha)

        return pressure_coefficients

    def get_model_id(
            self,
            model_name,
            alpha
    ):
        model_name_str = str(model_name)
        if model_name_str[1] in ['2', '3']:
            model_name = int(model_name_str[1] + model_name_str[0] + model_name_str[2])

        model_id = asyncio.run(
            find_experiment_by_model_name(model_name, alpha, self.engine, self.DB_URL_SERVER)).model_id

        return model_id

    def _get_coordinates(
            self
    ):
        alpha = self._get_alpha()
        model_size = self._get_model_size()
        model_name, _ = get_model_and_scale_factors(*model_size, alpha)

        model_id = self.get_model_id(model_name, alpha)

        coordinates = asyncio.run(load_positions(model_id, alpha, self.engine, self.DB_URL_SERVER))

        return coordinates

    def _get_face_number(
            self
    ):
        alpha = self._get_alpha()

        model_name = self._get_model_name()
        model_id = self.get_model_id(model_name, alpha)

        face_number = asyncio.run(load_face_number(model_id, alpha, self.engine, self.DB_URL_SERVER))

        return face_number

    def _get_size_tpu(
            self
    ):
        model_name = self._get_model_name()
        model_name_list = [i for i in list(str(model_name))]
        breadth, depth, height = [int(i) / 10 for i in model_name_list]

        return breadth, depth, height

    def _get_size_and_count_sensors(
            self,
            count_sensors_on_model
    ):
        model_name = self._get_model_name()

        size, count_sensors = get_size_tpu_and_count_sensors(
            count_sensors_on_model,
            model_name,
        )
        return size, count_sensors

    def _get_model_name(
            self
    ):
        alpha = self._get_alpha()
        model_size = self._get_model_size()

        model_name, _ = get_model_and_scale_factors(*model_size, alpha)

        return model_name

    def _get_count_sensors(
            self
    ):
        count_sensors_on_model = 252
        count_sensors_on_middle_row = 7
        count_sensors_on_side_row = 7

        return count_sensors_on_model, count_sensors_on_middle_row, count_sensors_on_side_row

    def _get_pressure_coefficients_storage(
            self
    ):
        pressure_coefficients_storage = {}

        alpha = self._get_alpha()
        model_size = self._get_model_size()
        model_name, _ = get_model_and_scale_factors(*model_size, alpha)
        model_name_str = str(model_name)

        model_id = self.get_model_id(model_name, alpha)

        angle_border = self._get_angle_border()

        with ProcessPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = {}
            for angle in range(0, angle_border + 5, 5):
                args = (model_name_str, angle, model_id, alpha)

                future = self._run_future(executor,
                                          self._get_pressure_coefficients_for_definition_angle_future,
                                          self.DB_URL_LOCAL, self.DB_URL_SERVER,
                                          *args)

                futures[future] = angle

            for future in as_completed(futures):
                try:
                    result = future.result()
                    angle = futures[future]
                    pressure_coefficients_storage[angle] = result
                except Exception as e:
                    print(f"Ошибка в задаче: {e}")

            executor.shutdown(wait=True)

        return pressure_coefficients_storage

    def _get_angle_border(
            self
    ):
        model_name = self._get_model_name()
        angle_border = get_angle_border(str(model_name))

        return angle_border
