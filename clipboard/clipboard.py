import os
import time
import pickle
from typing import Tuple

import toml
from numpy import array, any

# local imports
from plot.plot import Plot
from databasetoolkit.databasetoolkit import DataBaseToolkit
from utils.utils import (get_logger,
                         calculate_cx_cy,
                         calculate_cmz,
                         changer_sequence_numbers,
                         changer_sequence_coefficients,
                         get_model_and_scale_factors,
                         get_view_permutation_data,
                         get_base_angle,
                         get_sequence_permutation_data,
                         converter_coordinates_to_real,
                         )


class Clipboard:
    """Буфер и взаимодействует с DataBaseToolkit.
    Присутствует возможность отключения сохранения данных в памяти.
    """
    config = toml.load('config.toml')

    # Данные из бд
    _save_pressure_coefficients = config['clipboard']['data_from_db']['pressure_coefficients']
    _save_coordinates = config['clipboard']['data_from_db']['coordinates']
    _save_average_wind_speed = config['clipboard']['data_from_db']['average_wind_speed']
    _save_face_number = config['clipboard']['data_from_db']['face_number']
    _save_cx = config['clipboard']['data_from_db']['cx']
    _save_cy = config['clipboard']['data_from_db']['cy']
    _save_cmz = config['clipboard']['data_from_db']['cmz']

    # Графики
    _save_pseudocolor_coefficients = config['clipboard']['plots']['pseudocolor_coefficients']
    _save_isofields_coefficients = config['clipboard']['plots']['isofields_coefficients']
    _save_isofields_pressure = config['clipboard']['plots']['isofields_pressure']
    _save_summary_spectres_cx = config['clipboard']['plots']['summary_spectres_cx']
    _save_summary_spectres_cy = config['clipboard']['plots']['summary_spectres_cy']
    _save_summary_spectres_cmz = config['clipboard']['plots']['summary_spectres_cmz']
    _save_summary_coefficients_cx = config['clipboard']['plots']['summary_coefficients_cx']
    _save_summary_coefficients_cy = config['clipboard']['plots']['summary_coefficients_cy']
    _save_summary_coefficients_cmz = config['clipboard']['plots']['summary_coefficients_cmz']
    _save_summary_coefficients_polar = config['clipboard']['plots']['summary_coefficients_polar']
    _save_envelopes = config['clipboard']['plots']['envelopes']
    _save_model_polar = config['clipboard']['plots']['model_polar']
    _save_model_3d = config['clipboard']['plots']['model_3d']

    # Настройки
    _read_clipboard_binary = config['clipboard']['settings']['read_clipboard_binary']

    def __init__(self, ex_clipboard = None):
        self.logger = get_logger('Clipboard')
        self.logger.info('Создание буфера')
        self.database_obj = DataBaseToolkit()
        self.save_mode = True
        if ex_clipboard:
            self.clipboard_dict = ex_clipboard
            self.logger.info('Буфер создан на основе переданного')

        elif os.path.exists('clipboard\\clipboard_binary') and self._read_clipboard_binary:
            with open(f'{os.getcwd()}\\clipboard\\clipboard_binary', "rb") as clipboard_binary:
                self.clipboard_dict = pickle.loads(clipboard_binary.read())

            self.logger.info('Буфер создан на основе локального файла')

        else:
            self._init_clipboard_dict()
            self.logger.info('Буфер успешно создан')

    def _init_clipboard_dict(self):
        """Создание буфера
        const_stuff сюда входят некоторые параметры модели(координаты, скорость ветра, нумерация датчиков),
        потому что они идентичны для всех углов атаки ветра выбранной модели.
        Также в const_stuff входят графики целиком описывающие модель.
        """
        self.clipboard_dict = dict({'4': dict(),
                                    '6': dict()
                                    })

        experiments = self.database_obj.get_experiments()

        self.clipboard_dict['4'] = experiments['4']
        for key in list(experiments['4'].keys()):
            if key[0] != key[1]:
                self.clipboard_dict['4'][key[1] + key[0] + key[2]] = dict()

        self.clipboard_dict['6'] = experiments['6']
        for key in list(experiments['6'].keys()):
            if key[0] != key[1]:
                self.clipboard_dict['6'][key[1] + key[0] + key[2]] = dict()

        angles = range(0, 360, 5)

        for alpha in self.clipboard_dict.keys():
            for model_name in self.clipboard_dict[alpha].keys():
                for angle in angles:
                    self.clipboard_dict[alpha][model_name][str(angle)] = dict()

                self.clipboard_dict[alpha][model_name]['const_stuff'] = dict()

        unique_model_names = set(self.clipboard_dict['4'].keys()).union(set(self.clipboard_dict['6'].keys()))

        for umn in unique_model_names:
            self.clipboard_dict[umn] = dict()

        # Уникальные данные\графики для конкретного размера
        self.clipboard_dict['unique_stuff_for_size'] = dict()

    def get_pressure_coefficients(self, alpha: str, model_name: str, angle: str):
        model_name_base = model_name
        turn_flag = False
        if model_name[0] == model_name[1]:
            angle_border = 45
            type_base = 'square'

        else:
            angle_border = 90
            type_base = 'rectangle'

        if model_name[1] in ['2', '3']:
            model_name_base = model_name[1] + model_name[0] + model_name[2]
            angle = str((int(angle) + 270) % 360)
            turn_flag = True

        # Поворот данных для отображения углов, выходящих за границы имеющихся
        if int(angle) > angle_border:
            permutation_view = get_view_permutation_data(type_base, int(angle))  # вид последовательности данных
            base_angle = get_base_angle(int(angle), permutation_view, type_base)
            sequence_permutation = get_sequence_permutation_data(type_base, permutation_view, int(angle))

            pressure_coefficients = self.get_pressure_coefficients_from_clipboard(alpha, model_name_base,
                                                                                  str(base_angle))

            pressure_coefficients = changer_sequence_coefficients(pressure_coefficients, permutation_view,
                                                                  model_name, sequence_permutation)
        else:
            pressure_coefficients = self.get_pressure_coefficients_from_clipboard(alpha, model_name_base, angle)

        # Поворот модели
        if turn_flag:
            if int(angle) % 90 != 0:
                model_name_base = model_name
            pressure_coefficients = changer_sequence_coefficients(pressure_coefficients, 'forward', model_name_base,
                                                                  (3, 0, 1, 2))
        return pressure_coefficients

    def get_pressure_coefficients_from_clipboard(self, alpha: str, model_name: str, angle: str):
        """Возвращает коэффициенты давления из буфера"""
        self.logger.info(f"Запрос коэффициентов давления "
                         f"модель = {model_name} альфа = {alpha} угол = {angle.rjust(2, '0')} из буфера")

        if not any(self.clipboard_dict[alpha][model_name][angle].get('pressure_coefficients')):
            pressure_coefficients = array(self.database_obj.get_pressure_coefficients(alpha, model_name, angle))
            if self.save_mode:
                self.clipboard_dict[alpha][model_name][angle]['pressure_coefficients'] = pressure_coefficients
            else:
                self.logger.info(f"       Коэффициенты давления модель = {model_name} "
                                 f"альфа = {alpha} угол = {angle.rjust(2, '0')} не сохранены в буфер")
        else:
            pressure_coefficients = self.clipboard_dict[alpha][model_name][angle]['pressure_coefficients']
            self.logger.info(f"Запрос коэффициентов давления модель = {model_name} "
                             f"альфа = {alpha} угол = {angle.rjust(2, '0')} из буфера успешно выполнен")

        pressure_coefficients = pressure_coefficients / 1000

        return pressure_coefficients

    def get_coordinates(self, alpha: str, model_scale: str):
        """Возвращает координаты датчиков из буфера"""
        self.logger.info(f"Запрос координаты датчиков модель = {model_scale} альфа = {alpha} из буфера")

        if model_scale[1] in ['2', '3']:
            model_scale = model_scale[1] + model_scale[0] + model_scale[2]

        if not self.clipboard_dict[alpha][model_scale]['const_stuff'].get('x') or \
                not self.clipboard_dict[alpha][model_scale]['const_stuff'].get('z'):
            x, z = self.database_obj.get_coordinates(alpha, model_scale)
            if self.save_mode:
                self.clipboard_dict[alpha][model_scale]['const_stuff']['x'] = x
                self.clipboard_dict[alpha][model_scale]['const_stuff']['z'] = z
            else:
                self.logger.info(f"       Координаты датчиков модель = {model_scale} альфа = {alpha} "
                                 f"не сохранены в буфер")
        else:
            x = self.clipboard_dict[alpha][model_scale]['const_stuff']['x']
            z = self.clipboard_dict[alpha][model_scale]['const_stuff']['z']

            self.logger.info(f"Запрос координат датчиков модель = {model_scale} альфа = {alpha} "
                             f"из буфера успешно выполнен")

        return [x, z]

    def get_uh_average_wind_speed(self, alpha: str, model_name: str):
        """Возвращает среднюю скорость ветра из буфера"""
        self.logger.info(f"Запрос средней скорости ветра модель = {model_name} альфа = {alpha} из буфера")

        if not self.clipboard_dict[alpha][model_name]['const_stuff'].get('uh_average_wind_speed'):
            average_wind_speed = self.database_obj.get_uh_average_wind_speed(alpha, model_name)
            if self.save_mode:
                self.clipboard_dict[alpha][model_name]['const_stuff']['uh_average_wind_speed'] = average_wind_speed
            else:
                self.logger.info(f"       Cредняя скорость ветра "
                                 f"модель = {model_name} альфа = {alpha} не сохранена в буфер")
        else:
            average_wind_speed = self.clipboard_dict[alpha][model_name]['const_stuff']['uh_average_wind_speed']

            self.logger.info(f"Запрос средней скорости ветра "
                             f"модель = {model_name} альфа = {alpha} из буфера успешно выполнен")

        return average_wind_speed

    def get_face_number(self, alpha: str, model_name: str):
        """Возвращает нумерацию датчиков из буфера"""
        self.logger.info(f"Запрос нумерации датчиков модель = {model_name} альфа = {alpha} из буфера")

        model_name_n = None
        f = False

        if model_name[1] in ['2', '3']:
            model_name_n = model_name
            model_name = model_name[1] + model_name[0] + model_name[2]
            f = True

        if not self.clipboard_dict[alpha][model_name]['const_stuff'].get('face_number'):
            face_number = self.database_obj.get_face_number(alpha, model_name)
            if self.save_mode:
                self.clipboard_dict[alpha][model_name]['const_stuff']['face_number'] = face_number
            else:
                self.logger.info(f"       Нумерация датчиков модель = {model_name} альфа = {alpha} "
                                 f"не сохранена в буфер")
        else:
            face_number = self.clipboard_dict[alpha][model_name]['const_stuff']['face_number']

            self.logger.info(f"Запрос нумерации датчиков модель = {model_name} альфа = {alpha} "
                             f"из буфера успешно выполнен")

        if f:
            face_number = changer_sequence_numbers(face_number, model_name_n, (3, 0, 1, 2))

        return face_number

    def get_cx_cy(self, alpha: str, model_name: str, angle: str):
        """Возвращает суммарные коэффициенты Cx Cy из буфера"""
        self.logger.info(f"Запрос суммарных коэффициентов Cx Cy "
                         f"модель = {model_name} альфа = {alpha} угол = {angle.rjust(2, '0')} из буфера")

        if not any(self.clipboard_dict[alpha][model_name][angle].get('Cx')) or \
                not any(self.clipboard_dict[alpha][model_name][angle].get('Cy')):
            pressure_coefficients = self.get_pressure_coefficients(alpha, model_name, angle)

            cx, cy = calculate_cx_cy(model_name, pressure_coefficients)
            if self.save_mode:
                self.clipboard_dict[alpha][model_name][angle]['Cx'] = cx
                self.clipboard_dict[alpha][model_name][angle]['Cy'] = cy
            else:
                self.logger.info(f"       Cуммарные коэффициенты  Cx Cy "
                                 f"модель = {model_name} альфа = {alpha} угол = {angle.rjust(2, '0')} "
                                 f"не сохранена в буфер")
        else:
            cx = self.clipboard_dict[alpha][model_name][angle]['Cx']
            cy = self.clipboard_dict[alpha][model_name][angle]['Cy']

            self.logger.info(f"Запрос суммарных коэффициентов Cx Cy модель = {model_name} альфа = {alpha} "
                             f"угол = {angle.rjust(2, '0')} из буфера успешно выполнен")

        return cx, cy

    def get_cmz(self, alpha: str, model_name: str, angle: str):
        """Возвращает CMz из буфера"""
        self.logger.info(f"Запрос CMz модель = {model_name} альфа = {alpha} угол = {angle.rjust(2, '0')} из буфера")

        if not any(self.clipboard_dict[alpha][model_name][angle].get('CMz')):
            coefficients = self.get_pressure_coefficients(alpha, model_name, angle)
            coordinates = self.get_coordinates(alpha, model_name)
            cmz = calculate_cmz(model_name, angle, coefficients, coordinates)
            if self.save_mode:
                self.clipboard_dict[alpha][model_name][angle]['CMz'] = cmz
            else:
                self.logger.info(f"       CMz модель = {model_name} альфа = {alpha} угол = {angle.rjust(2, '0')} "
                                 f"не сохранена в буфер")
        else:
            cmz = self.clipboard_dict[alpha][model_name][angle]['CMz']

            self.logger.info(f"Запрос CMz модель = {model_name} альфа = {alpha} "
                             f"угол = {angle.rjust(2, '0')} из буфера успешно выполнен")

        return cmz

    def get_isofields(self,
                      alpha: str,
                      model_size: Tuple[str, str, str],
                      angle: str,
                      mode: str,
                      type_plot: str,
                      pressure_plot_parameters = None):
        self.logger.info(f'Запрос {type_plot} альфа = {alpha} размер = {" ".join(model_size)} '
                         f'угол = {angle.rjust(2, "0")} режим = {mode.ljust(4, " ")} из буфера')
        flag_save = False

        id_fig = f'{type_plot}_{mode}_{"_".join(model_size)}'

        model_scale, scale_factors = get_model_and_scale_factors(*model_size, alpha)

        if not self.clipboard_dict[alpha][model_scale][angle].get(id_fig):
            pressure_coefficients = self.get_pressure_coefficients(alpha, model_scale, angle)
            coordinates = self.get_coordinates(alpha, model_scale)

            self.logger.info(f'Отрисовка {type_plot} альфа = {alpha} размер = {" ".join(model_size)} '
                             f'угол = {angle.rjust(2, "0")} режим = {mode.ljust(4, " ")}')

            fig = Plot.isofields_coefficients(model_scale,
                                              model_size,
                                              scale_factors,
                                              alpha,
                                              mode,
                                              angle,
                                              pressure_coefficients,
                                              coordinates,
                                              pressure_plot_parameters)

            if fig is not None:
                self.logger.info(f'Отрисовка {type_plot} альфа = {alpha} размер = {" ".join(model_size)} '
                                 f'угол = {angle.rjust(2, "0")} режим = {mode.ljust(4, " ")} успешно выполнена')
            else:
                self.logger.warning(f'Отрисовка {type_plot} альфа = {alpha} размер = {" ".join(model_size)} '
                                    f'угол = {angle.rjust(2, "0")} режим = {mode.ljust(4, " ")} не выполнена')

            if type_plot == 'isofields_pressure':
                if self._save_isofields_pressure:
                    self.clipboard_dict[alpha][model_scale][angle][id_fig] = fig
                    flag_save = True

            elif type_plot == 'isofields_coefficients':
                if self._save_isofields_coefficients:
                    self.clipboard_dict[alpha][model_scale][angle][id_fig] = fig
                    flag_save = True

            if flag_save:
                self.logger.info(f'       {type_plot} альфа = {alpha} размер = {" ".join(model_size)} '
                                 f'угол = {angle.rjust(2, "0")} режим = {mode.ljust(4, " ")} сохранены в буфер')

            else:
                self.logger.info(f'       {type_plot} альфа = {alpha} размер = {" ".join(model_size)} '
                                 f'угол = {angle.rjust(2, "0")} режим = {mode.ljust(4, " ")} не сохранены в буфер')

        else:
            fig = self.clipboard_dict[alpha][model_scale][angle][id_fig]
            self.logger.info(f'Запрос {type_plot} альфа = {alpha} размер = {" ".join(model_size)} '
                             f'угол = {angle.rjust(2, "0")} режим = {mode.ljust(4, " ")} из буфера успешно выполнен')

        return fig

    def get_pseudocolor_coefficients(self,
                                     alpha: str,
                                     model_size: Tuple[str, str, str],
                                     angle: str,
                                     mode: str,
                                     ):
        type_plot = 'pseudocolor_coefficients'

        self.logger.info(f'Запрос {type_plot} альфа = {alpha} размер = {" ".join(model_size)} '
                         f'угол = {angle.rjust(2, "0")} режим = {mode.ljust(4, " ")} из буфера')

        id_fig = f'{type_plot}_{mode}_{"_".join(model_size)}'

        model_scale, scale_factors = get_model_and_scale_factors(*model_size, alpha)

        if not self.clipboard_dict[alpha][model_scale][angle].get(id_fig):
            pressure_coefficients = self.get_pressure_coefficients(alpha, model_scale, angle)
            coordinates = self.get_coordinates(alpha, model_scale)

            self.logger.info(f'Отрисовка {type_plot} альфа = {alpha} размер = {" ".join(model_size)} '
                             f'угол = {angle.rjust(2, "0")} режим = {mode.ljust(4, " ")}')

            fig = Plot.pseudocolor_coefficients(model_scale, mode, angle, alpha, pressure_coefficients, coordinates)

            if fig is not None:
                self.logger.info(f'Отрисовка {type_plot} альфа = {alpha} размер = {" ".join(model_size)} '
                                 f'угол = {angle.rjust(2, "0")} режим = {mode.ljust(4, " ")} успешно выполнена')
            else:
                self.logger.warning(f'Отрисовка {type_plot} альфа = {alpha} размер = {" ".join(model_size)} '
                                    f'угол = {angle.rjust(2, "0")} режим = {mode.ljust(4, " ")} не выполнена')

            if self._save_pseudocolor_coefficients:
                self.clipboard_dict[alpha][model_scale][angle][id_fig] = fig

                self.logger.info(f'       {type_plot} альфа = {alpha} размер = {" ".join(model_size)} '
                                 f'угол = {angle.rjust(2, "0")} режим = {mode.ljust(4, " ")} сохранены в буфер')

            else:
                self.logger.info(f'       {type_plot} альфа = {alpha} размер = {" ".join(model_size)} '
                                 f'угол = {angle.rjust(2, "0")} режим = {mode.ljust(4, " ")} не сохранены в буфер')

        else:
            fig = self.clipboard_dict[alpha][model_scale][angle][id_fig]
            self.logger.info(f'Запрос {type_plot} альфа = {alpha} размер = {" ".join(model_size)} '
                             f'угол = {angle.rjust(2, "0")} режим = {mode.ljust(4, " ")} из буфера успешно выполнен')

        return fig

    def get_data_summary(self, alpha: str, angle: str, mode: str, model_scale: str):
        data = dict()
        f_cx = 'cx' in mode
        f_cy = 'cy' in mode
        f_cmz = 'cmz' in mode

        if f_cx or f_cy:
            cx, cy = self.get_cx_cy(alpha, model_scale, str(angle))
            if f_cx:
                data['cx'] = cx
            if f_cy:
                data['cy'] = cy

        if f_cmz:
            data['cmz'] = self.get_cmz(alpha, model_scale, str(angle))

        return data

    def get_summary_spectres(self,
                             alpha: str,
                             model_size: Tuple[str, str, str],
                             angle: str,
                             mode: str,
                             scale: str,
                             type_plot: str):
        mode = mode.lower().replace(' ', '_')
        model_scale, _ = get_model_and_scale_factors(*model_size, alpha)
        id_fig = f'{type_plot}_{mode}_{scale}_{"_".join(model_size)}'

        message = f'{" ".join(list(model_size))} {alpha} {int(angle):02} {mode}'

        self.logger.info(f'Запрос суммарных спектров {message} из буфера')
        if not self.clipboard_dict[alpha][model_scale][angle].get(id_fig):
            f_cx = 'cx' in mode
            f_cy = 'cy' in mode
            f_cmz = 'cmz' in mode

            data = self.get_data_summary(alpha, angle, mode, model_scale)

            self.logger.info(f'Отрисовка суммарных спектров {message}')
            fig = Plot.welch_graphs(data, model_size, alpha, angle)

            if all((self._save_summary_spectres_cx or f_cx, self._save_summary_spectres_cy or f_cy,
                    self._save_summary_spectres_cmz or f_cmz)):
                self.clipboard_dict[alpha][model_scale][angle][id_fig] = fig
            else:
                self.logger.info(f'       Cуммарные спектры {message} не сохранены в буфер')
        else:
            fig = self.clipboard_dict[alpha][model_scale][angle][id_fig]
            self.logger.info(f'Запрос суммарных спектров {message} из буфера успешно выполнен')

        return fig

    def get_summary_coefficients(self,
                                 alpha: str,
                                 model_size: Tuple[str, str, str],
                                 angle: str,
                                 mode: str, ):
        mode = mode.lower()
        model_scale, _ = get_model_and_scale_factors(*model_size, alpha)
        type_plot = 'summary_coefficients'
        id_fig = f'{type_plot}_{mode}_{"_".join(model_size)}'
        if not self.clipboard_dict[alpha][model_scale][angle].get(id_fig):
            f_cx = 'cx' in mode
            f_cy = 'cy' in mode
            f_cmz = 'cmz' in mode
            data = self.get_data_summary(alpha, angle, mode, model_scale)
            fig = Plot.summary_coefficients(data, model_scale, alpha, str(angle))

            if all((self._save_summary_coefficients_cx or f_cx, self._save_summary_coefficients_cy or f_cy,
                    self._save_summary_coefficients_cmz or f_cmz)):
                self.clipboard_dict[alpha][model_scale][angle][id_fig] = fig
            else:
                self.logger.info(f'       Cуммарные коэф график не сохранены в буфер')

        else:
            fig = self.clipboard_dict[alpha][model_scale][angle][id_fig]
            self.logger.info(f'Запрос суммарных коэф из буфера успешно выполнен')

        return fig

    def get_plot_pressure_tap_locations(self, model_size, alpha):
        id_fig = f'model_polar_{"_".join(model_size)}'
        if not self.clipboard_dict['unique_stuff_for_size'].get(id_fig):
            model_scale, scale_factors = get_model_and_scale_factors(*model_size, alpha)
            coordinates = self.get_coordinates(alpha, model_scale)
            coordinates = converter_coordinates_to_real(*coordinates, model_size, model_scale)
            fig = Plot.model_pic(model_size, model_scale, coordinates)
            if self._save_model_polar:
                self.clipboard_dict['unique_stuff_for_size'][id_fig] = fig
            else:
                ...
        else:
            fig = self.clipboard_dict['unique_stuff_for_size'][id_fig]

        return fig

    def get_plot_model_3d(self, model_size):
        id_fig = f'model_3d_{"_".join(model_size)}'
        if not self.clipboard_dict['unique_stuff_for_size'].get(id_fig):
            fig = Plot.model_3d(model_size)
            if self._save_model_3d:
                self.clipboard_dict['unique_stuff_for_size'][id_fig] = fig
            else:
                ...
        else:
            fig = self.clipboard_dict['unique_stuff_for_size'][id_fig]

        return fig

    def get_model_polar(self, model_size):
        id_fig = f'model_polar_{"_".join(model_size)}'
        if not self.clipboard_dict['unique_stuff_for_size'].get(id_fig):
            fig = Plot.model_polar(model_size)
            if self._save_model_3d:
                self.clipboard_dict['unique_stuff_for_size'][id_fig] = fig
            else:
                ...
        else:
            fig = self.clipboard_dict['unique_stuff_for_size'][id_fig]

        return fig


if __name__ == '__main__':
    c = Clipboard()
    # print(c.get_face_number('4','111'))
