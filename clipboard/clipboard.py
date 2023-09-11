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
                         get_model_and_scale_factors_interference,
                         get_view_permutation_data,
                         get_base_angle,
                         get_sequence_permutation_data,
                         converter_coordinates_to_real, get_coordinates_interference,
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

    def __init__(self, ex_clipboard=None):
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
        self.clipboard_dict = {'isolated': {'4': dict(),
                                            '6': dict(),
                                            },
                               'interference': {140: dict(),
                                                196: dict(),
                                                280: dict(),
                                                420: dict(),
                                                560: dict(),
                                                },
                               'without_eaves': {
                                   'flat roof': dict(),
                                   'gable roof': dict(),
                                   'hipped roof': dict(),
                               },
                               'with_eaves': {
                                   'gable roof': dict(),
                               },
                               'non_isolated': {
                                   'flat roof': dict(),
                                   'gable roof': dict(),
                                   'hip roof': dict(),
                               }}

        # isolated
        experiments = self.database_obj.get_experiments()

        self.clipboard_dict['isolated']['4'] = experiments['4']
        for key in list(experiments['4'].keys()):
            if key[0] != key[1]:
                self.clipboard_dict['isolated']['4'][key[1] + key[0] + key[2]] = dict()

        self.clipboard_dict['isolated']['6'] = experiments['6']
        for key in list(experiments['6'].keys()):
            if key[0] != key[1]:
                self.clipboard_dict['isolated']['6'][key[1] + key[0] + key[2]] = dict()

        angles = range(0, 360, 5)

        for alpha in self.clipboard_dict['isolated'].keys():
            for model_name in self.clipboard_dict['isolated'][alpha].keys():
                for angle in angles:
                    self.clipboard_dict['isolated'][alpha][model_name][str(angle)] = dict()

                self.clipboard_dict['isolated'][alpha][model_name]['const_stuff'] = dict()

        unique_model_names = set(self.clipboard_dict['isolated']['4'].keys()).union(
            set(self.clipboard_dict['isolated']['6'].keys()))

        for umn in unique_model_names:
            self.clipboard_dict['isolated'][umn] = dict()

        # Уникальные данные\графики для конкретного размера
        self.clipboard_dict['isolated']['unique_stuff_for_size'] = dict()

        # interference
        for hr in self.clipboard_dict['interference'].keys():
            if hr in (196, 560):
                for c in (28, 33, 34, 37):
                    self.clipboard_dict['interference'][hr][c] = dict()
            else:
                for c in range(1, 38):
                    self.clipboard_dict['interference'][hr][c] = dict()

            for c in self.clipboard_dict['interference'][hr].keys():
                for angle in range(0, 360, 5):
                    self.clipboard_dict['interference'][hr][c][angle] = dict()

                self.clipboard_dict['interference'][hr][c]['const_stuff'] = dict()

        # without_eaves
        t = 'with_eaves'
        for rt in self.clipboard_dict[t].keys():
            self.clipboard_dict[t][rt][16] = dict()
            for depth in (16, 24, 40):
                self.clipboard_dict[t][rt][16][depth] = dict()
                for height in (4, 8, 12, 16):
                    self.clipboard_dict[t][rt][16][depth][height] = dict()
                    for pitch in (0, 5, 10, 14, 18, 22, 27, 30, 45):
                        self.clipboard_dict[t][rt][16][depth][height][pitch] = dict()
                        for angle in (0, 15, 30, 45, 60, 75, 90):
                            self.clipboard_dict[t][rt][16][depth][height][pitch][angle] = dict()

                        self.clipboard_dict[t][rt][16][depth][height][pitch]['const_stuff'] = dict()

        # with_eaves
        t = 'with_eaves'
        b = 16
        d = 24
        p = 26.7
        for rt in self.clipboard_dict[t].keys():
            self.clipboard_dict[t][rt][b] = dict()
            self.clipboard_dict[t][rt][b][d] = dict()
            for height in (6, 12, 18):
                self.clipboard_dict[t][rt][b][d][height] = dict()
                self.clipboard_dict[t][rt][b][d][height][p] = dict()
                for angle in (0, 15, 30, 45, 60, 75, 90):
                    self.clipboard_dict[t][rt][b][d][height][p][angle] = dict()

                self.clipboard_dict[t][rt][b][d][height][p]['const_stuff'] = dict()

        # non_isolated
        t = 'non_isolated'
        b = 16
        d = 24
        for rt in self.clipboard_dict[t].keys():
            self.clipboard_dict[t][rt] = dict()
            for arr_o in ('Regular', 'Staggerd', 'Random'):
                self.clipboard_dict[t][rt][arr_o] = dict()
                for area in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6):
                    self.clipboard_dict[t][rt][arr_o][area] = dict()
                    self.clipboard_dict[t][rt][arr_o][area][b] = dict()
                    self.clipboard_dict[t][rt][arr_o][area][b][d] = dict()
                    for height in (6, 12, 18):
                        self.clipboard_dict[t][rt][arr_o][area][b][d][height] = dict()
                        for pitch in (0, 26.7, 45):
                            self.clipboard_dict[t][rt][arr_o][area][b][d][height][pitch] = dict()
                            for angle in (0, 23, 45, 68, 90, 113, 135, 158, 180, 203, 225, 248, 270, 293, 315, 338):
                                self.clipboard_dict[t][rt][arr_o][area][b][d][height][pitch][angle] = dict()

                            self.clipboard_dict[t][rt][arr_o][area][b][d][height][pitch]['const_stuff'] = dict()

    def get_pressure_coefficients(self, db, **kwargs):
        if db == 'isolated':
            alpha = kwargs['alpha']
            model_name = kwargs['model_name']
            angle = kwargs['angle']

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

                pressure_coefficients = self.get_pressure_coefficients_from_clipboard(db='isolated', alpha=alpha,
                                                                                      model_name=model_name_base,
                                                                                      angle=str(base_angle))

                pressure_coefficients = changer_sequence_coefficients(pressure_coefficients, permutation_view,
                                                                      model_name, sequence_permutation)
            else:
                pressure_coefficients = self.get_pressure_coefficients_from_clipboard(db='isolated', alpha=alpha,
                                                                                      model_name=model_name_base,
                                                                                      angle=angle)

            # Поворот модели
            if turn_flag:
                if int(angle) % 90 != 0:
                    model_name_base = model_name
                pressure_coefficients = changer_sequence_coefficients(pressure_coefficients, 'forward', model_name_base,
                                                                      (3, 0, 1, 2))
        elif db == 'interference':
            case = kwargs['case']
            model_name = kwargs['model_name']
            angle = kwargs['angle']
            pressure_coefficients = self.get_pressure_coefficients_from_clipboard(db='interference', case=case,
                                                                                  model_name=model_name,
                                                                                  angle=angle)

        return pressure_coefficients

    def get_pressure_coefficients_from_clipboard(self, db, **kwargs):
        """Возвращает коэффициенты давления из буфера"""
        if db == 'isolated':
            alpha = kwargs['alpha']
            model_name = kwargs['model_name']
            angle = kwargs['angle']
            self.logger.info(f"Запрос коэффициентов давления "
                             f"модель = {model_name} альфа = {alpha} угол = {angle.rjust(2, '0')} из буфера")

            if not any(self.clipboard_dict['isolated'][alpha][model_name][angle].get('pressure_coefficients')):
                pressure_coefficients = array(
                    self.database_obj.get_pressure_coefficients('isolated', alpha=alpha, model_name=model_name,
                                                                angle=angle))
                if self.save_mode:
                    self.clipboard_dict['isolated'][alpha][model_name][angle][
                        'pressure_coefficients'] = pressure_coefficients
                else:
                    self.logger.info(f"       Коэффициенты давления модель = {model_name} "
                                     f"альфа = {alpha} угол = {angle.rjust(2, '0')} не сохранены в буфер")
            else:
                pressure_coefficients = self.clipboard_dict['isolated'][alpha][model_name][angle][
                    'pressure_coefficients']
                self.logger.info(f"Запрос коэффициентов давления модель = {model_name} "
                                 f"альфа = {alpha} угол = {angle.rjust(2, '0')} из буфера успешно выполнен")

            pressure_coefficients = pressure_coefficients / 1000

        elif db == 'interference':
            case = kwargs['case']
            model_name = kwargs['model_name']
            angle = kwargs['angle']

            if not any(self.clipboard_dict['interference'][model_name][case][angle].get('pressure_coefficients')):
                pressure_coefficients = array(
                    self.database_obj.get_pressure_coefficients('interference', case=case, model_name=model_name,
                                                                angle=angle))
                if self.save_mode:
                    self.clipboard_dict['interference'][model_name][case][angle][
                        'pressure_coefficients'] = pressure_coefficients
            else:
                pressure_coefficients = self.clipboard_dict['interference'][model_name][case][angle][
                    'pressure_coefficients']

            pressure_coefficients = pressure_coefficients / 1000

        return pressure_coefficients

    def get_coordinates(self, db='isolated', **kwargs):
        """Возвращает координаты датчиков из буфера"""
        if db == 'isolated':
            model_scale = kwargs['model_scale']
            alpha = kwargs['alpha']

            self.logger.info(f"Запрос координаты датчиков модель = {model_scale} альфа = {alpha} из буфера")

            if model_scale[1] in ['2', '3']:
                model_scale = model_scale[1] + model_scale[0] + model_scale[2]

            if not self.clipboard_dict['isolated'][alpha][model_scale]['const_stuff'].get('x') or \
                    not self.clipboard_dict['isolated'][alpha][model_scale]['const_stuff'].get('z'):
                x, z = self.database_obj.get_coordinates(alpha, model_scale)
                if self.save_mode:
                    self.clipboard_dict['isolated'][alpha][model_scale]['const_stuff']['x'] = x
                    self.clipboard_dict['isolated'][alpha][model_scale]['const_stuff']['z'] = z
                else:
                    self.logger.info(f"       Координаты датчиков модель = {model_scale} альфа = {alpha} "
                                     f"не сохранены в буфер")
            else:
                x = self.clipboard_dict['isolated'][alpha][model_scale]['const_stuff']['x']
                z = self.clipboard_dict['isolated'][alpha][model_scale]['const_stuff']['z']

                self.logger.info(f"Запрос координат датчиков модель = {model_scale} альфа = {alpha} "
                                 f"из буфера успешно выполнен")
        elif db == 'interference':
            model_scale = kwargs['model_scale']
            case = kwargs['case']

            if not self.clipboard_dict['interference'][model_scale][case]['const_stuff'].get('x') or \
                    not self.clipboard_dict['interference'][model_scale][case]['const_stuff'].get('z'):
                pressure_coefficients = self.get_pressure_coefficients('interference', case=case,
                                                                       model_name=model_scale,
                                                                       angle=0)

                x, z = get_coordinates_interference(len(pressure_coefficients[0]), model_scale / 1000)
                if self.save_mode:
                    self.clipboard_dict['interference'][model_scale][case]['const_stuff']['x'] = x
                    self.clipboard_dict['interference'][model_scale][case]['const_stuff']['z'] = z

            else:
                x = self.clipboard_dict['interference'][model_scale][case]['const_stuff']['x']
                z = self.clipboard_dict['interference'][model_scale][case]['const_stuff']['z']

        return [x, z]

    def get_uh_average_wind_speed(self, alpha: str, model_name: str):
        """Возвращает среднюю скорость ветра из буфера"""
        self.logger.info(f"Запрос средней скорости ветра модель = {model_name} альфа = {alpha} из буфера")

        if not self.clipboard_dict['isolated'][alpha][model_name]['const_stuff'].get('uh_average_wind_speed'):
            average_wind_speed = self.database_obj.get_uh_average_wind_speed(alpha, model_name)
            if self.save_mode:
                self.clipboard_dict['isolated'][alpha][model_name]['const_stuff'][
                    'uh_average_wind_speed'] = average_wind_speed
            else:
                self.logger.info(f"       Cредняя скорость ветра "
                                 f"модель = {model_name} альфа = {alpha} не сохранена в буфер")
        else:
            average_wind_speed = self.clipboard_dict['isolated'][alpha][model_name]['const_stuff'][
                'uh_average_wind_speed']

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

        if not self.clipboard_dict['isolated'][alpha][model_name]['const_stuff'].get('face_number'):
            face_number = self.database_obj.get_face_number(alpha, model_name)
            if self.save_mode:
                self.clipboard_dict['isolated'][alpha][model_name]['const_stuff']['face_number'] = face_number
            else:
                self.logger.info(f"       Нумерация датчиков модель = {model_name} альфа = {alpha} "
                                 f"не сохранена в буфер")
        else:
            face_number = self.clipboard_dict['isolated'][alpha][model_name]['const_stuff']['face_number']

            self.logger.info(f"Запрос нумерации датчиков модель = {model_name} альфа = {alpha} "
                             f"из буфера успешно выполнен")

        if f:
            face_number = changer_sequence_numbers(face_number, model_name_n, (3, 0, 1, 2))

        return face_number

    def get_cx_cy(self, db='isolated', **kwargs):
        """Возвращает суммарные коэффициенты Cx Cy из буфера"""
        if db == 'isolated':
            model_name = kwargs['model_scale']
            alpha = kwargs['alpha']
            angle = str(kwargs['angle'])

            self.logger.info(f"Запрос суммарных коэффициентов Cx Cy "
                             f"модель = {model_name} альфа = {alpha} угол = {angle.rjust(2, '0')} из буфера")

            if not any(self.clipboard_dict['isolated'][alpha][model_name][angle].get('Cx')) or \
                    not any(self.clipboard_dict['isolated'][alpha][model_name][angle].get('Cy')):
                pressure_coefficients = self.get_pressure_coefficients('isolated', alpha=alpha, model_name=model_name,
                                                                       angle=angle)
                coordinates = self.get_coordinates('isolated', alpha=alpha, model_scale=model_name)

                cx, cy = calculate_cx_cy(db='isolated', model_name=model_name,
                                         pressure_coefficients=pressure_coefficients,coordinates=coordinates)
                if self.save_mode:
                    self.clipboard_dict['isolated'][alpha][model_name][angle]['Cx'] = cx
                    self.clipboard_dict['isolated'][alpha][model_name][angle]['Cy'] = cy
                else:
                    self.logger.info(f"       Cуммарные коэффициенты  Cx Cy "
                                     f"модель = {model_name} альфа = {alpha} угол = {angle.rjust(2, '0')} "
                                     f"не сохранена в буфер")
            else:
                cx = self.clipboard_dict['isolated'][alpha][model_name][angle]['Cx']
                cy = self.clipboard_dict['isolated'][alpha][model_name][angle]['Cy']

                self.logger.info(f"Запрос суммарных коэффициентов Cx Cy модель = {model_name} альфа = {alpha} "
                                 f"угол = {angle.rjust(2, '0')} из буфера успешно выполнен")
        elif db == 'interference':
            model_name = kwargs['model_scale']
            case = kwargs['case']
            angle = int(kwargs['angle'])

            if not any(self.clipboard_dict['interference'][model_name][case][angle].get('Cx')) or \
                    not any(self.clipboard_dict['interference'][model_name][case][angle].get('Cy')):
                pressure_coefficients = self.get_pressure_coefficients('interference', case=case, model_name=model_name,
                                                                       angle=angle)

                cx, cy = calculate_cx_cy(db='interference', height=model_name / 1000,
                                         pressure_coefficients=pressure_coefficients, )
                if self.save_mode:
                    self.clipboard_dict['interference'][model_name][case][angle]['Cx'] = cx
                    self.clipboard_dict['interference'][model_name][case][angle]['Cy'] = cy

            else:
                cx = self.clipboard_dict['interference'][model_name][case][angle]['Cx']
                cy = self.clipboard_dict['interference'][model_name][case][angle]['Cy']

        return cx, cy

    def get_cmz(self, db='isolated', **kwargs):
        """Возвращает CMz из буфера"""
        if db == 'isolated':
            model_name = kwargs['model_scale']
            alpha = kwargs['alpha']
            angle = str(kwargs['angle'])
            self.logger.info(f"Запрос CMz модель = {model_name} альфа = {alpha} угол = {angle.rjust(2, '0')} из буфера")

            if not any(self.clipboard_dict['isolated'][alpha][model_name][angle].get('CMz')):
                coefficients = self.get_pressure_coefficients('isolated', alpha=alpha, model_name=model_name,
                                                              angle=angle)
                coordinates = self.get_coordinates(db='isolated', alpha=alpha, model_scale=model_name)
                cmz = calculate_cmz(db='isolated', model_name=model_name, angle=angle,
                                    pressure_coefficients=coefficients, coordinates=coordinates)
                if self.save_mode:
                    self.clipboard_dict['isolated'][alpha][model_name][angle]['CMz'] = cmz
                else:
                    self.logger.info(f"       CMz модель = {model_name} альфа = {alpha} угол = {angle.rjust(2, '0')} "
                                     f"не сохранена в буфер")
            else:
                cmz = self.clipboard_dict['isolated'][alpha][model_name][angle]['CMz']

                self.logger.info(f"Запрос CMz модель = {model_name} альфа = {alpha} "
                                 f"угол = {angle.rjust(2, '0')} из буфера успешно выполнен")
        elif db == 'interference':
            model_name = kwargs['model_scale']
            case = kwargs['case']
            angle = int(kwargs['angle'])

            if not any(self.clipboard_dict['interference'][model_name][case][angle].get('CMz')):
                coefficients = self.get_pressure_coefficients(db='interference', case=case, model_name=model_name,
                                                              angle=angle)
                coordinates = self.get_coordinates(db='interference', case=case, model_scale=model_name)
                cmz = calculate_cmz(db='interference', height=model_name / 1000, angle=angle,
                                    pressure_coefficients=coefficients, coordinates=coordinates)

                if self.save_mode:
                    self.clipboard_dict['interference'][model_name][case][angle]['CMz'] = cmz
            else:
                cmz = self.clipboard_dict['interference'][model_name][case][angle]['CMz']

        return cmz

    def get_isofields(self, db='isolated', **kwargs):
        if db == 'isolated':
            type_plot = kwargs['type_plot']
            angle = str(kwargs['angle'])
            alpha = kwargs['alpha']
            mode = kwargs['mode']
            model_size = kwargs['model_size']
            pressure_plot_parameters = kwargs['pressure_plot_parameters']

            self.logger.info(f'Запрос {type_plot} альфа = {alpha} размер = {" ".join(model_size)} '
                             f'угол = {angle} режим = {mode.ljust(4, " ")} из буфера')
            flag_save = False

            id_fig = f'{type_plot}_{mode}_{"_".join(model_size)}'

            model_scale, scale_factors = get_model_and_scale_factors(*model_size, alpha)

            if not self.clipboard_dict['isolated'][alpha][model_scale][angle].get(id_fig):
                pressure_coefficients = self.get_pressure_coefficients('isolated', alpha=alpha, model_name=model_scale,
                                                                       angle=angle)
                coordinates = self.get_coordinates(db, alpha=alpha, model_scale=model_scale)

                self.logger.info(f'Отрисовка {type_plot} альфа = {alpha} размер = {" ".join(model_size)} '
                                 f'угол = {angle.rjust(2, "0")} режим = {mode.ljust(4, " ")}')

                fig = Plot.isofields_coefficients(db='isolated',
                                                  model_scale=model_scale,
                                                  model_size=model_size,
                                                  scale_factors=scale_factors,
                                                  alpha=alpha,
                                                  mode=mode,
                                                  angle=angle,
                                                  pressure_coefficients=pressure_coefficients,
                                                  coordinates=coordinates,
                                                  pressure_plot_parameters=pressure_plot_parameters)

                if fig is not None:
                    self.logger.info(f'Отрисовка {type_plot} альфа = {alpha} размер = {" ".join(model_size)} '
                                     f'угол = {angle.rjust(2, "0")} режим = {mode.ljust(4, " ")} успешно выполнена')
                else:
                    self.logger.warning(f'Отрисовка {type_plot} альфа = {alpha} размер = {" ".join(model_size)} '
                                        f'угол = {angle.rjust(2, "0")} режим = {mode.ljust(4, " ")} не выполнена')

                if type_plot == 'isofields_pressure':
                    if self._save_isofields_pressure:
                        self.clipboard_dict['isolated'][alpha][model_scale][angle][id_fig] = fig
                        flag_save = True

                elif type_plot == 'isofields_coefficients':
                    if self._save_isofields_coefficients:
                        self.clipboard_dict['isolated'][alpha][model_scale][angle][id_fig] = fig
                        flag_save = True

                if flag_save:
                    self.logger.info(f'       {type_plot} альфа = {alpha} размер = {" ".join(model_size)} '
                                     f'угол = {angle.rjust(2, "0")} режим = {mode.ljust(4, " ")} сохранены в буфер')

                else:
                    self.logger.info(f'       {type_plot} альфа = {alpha} размер = {" ".join(model_size)} '
                                     f'угол = {angle.rjust(2, "0")} режим = {mode.ljust(4, " ")} не сохранены в буфер')

            else:
                fig = self.clipboard_dict['isolated'][alpha][model_scale][angle][id_fig]
                self.logger.info(f'Запрос {type_plot} альфа = {alpha} размер = {" ".join(model_size)} '
                                 f'угол = {angle.rjust(2, "0")} режим = {mode.ljust(4, " ")} из буфера успешно выполнен')
        elif db == 'interference':
            type_plot = kwargs['type_plot']
            angle = kwargs['angle']
            case = kwargs['case']
            mode = kwargs['mode']
            model_size = kwargs['model_size']
            pressure_plot_parameters = kwargs['pressure_plot_parameters']

            id_fig = f'{type_plot}_{mode}_{"_".join(model_size)}'

            model_scale, scale_factors = get_model_and_scale_factors_interference(*model_size)
            print(model_scale, scale_factors)
            if not self.clipboard_dict['interference'][model_scale][case][angle].get(id_fig):
                pressure_coefficients = self.get_pressure_coefficients('interference', case=case,
                                                                       model_name=model_scale,
                                                                       angle=angle)
                coordinates = get_coordinates_interference(len(pressure_coefficients[0]), model_scale / 1000)

                fig = Plot.isofields_coefficients(db='interference',
                                                  model_scale=model_scale,
                                                  model_size=model_size,
                                                  scale_factors=scale_factors,
                                                  case=case,
                                                  mode=mode,
                                                  angle=angle,
                                                  pressure_coefficients=pressure_coefficients,
                                                  coordinates=coordinates,
                                                  pressure_plot_parameters=pressure_plot_parameters)

                if type_plot == 'isofields_pressure':
                    if self._save_isofields_pressure:
                        self.clipboard_dict['interference'][model_scale][case][angle][id_fig] = fig

                elif type_plot == 'isofields_coefficients':
                    if self._save_isofields_coefficients:
                        self.clipboard_dict['interference'][model_scale][case][angle][id_fig] = fig

            else:
                fig = self.clipboard_dict['interference'][model_scale][case][angle][id_fig]

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

        if not self.clipboard_dict['isolated'][alpha][model_scale][angle].get(id_fig):
            pressure_coefficients = self.get_pressure_coefficients('isolated', alpha=alpha, model_name=model_scale,
                                                                   angle=angle)
            coordinates = self.get_coordinates('isolated', alpha=alpha, model_scale=model_scale)

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
                self.clipboard_dict['isolated'][alpha][model_scale][angle][id_fig] = fig

                self.logger.info(f'       {type_plot} альфа = {alpha} размер = {" ".join(model_size)} '
                                 f'угол = {angle.rjust(2, "0")} режим = {mode.ljust(4, " ")} сохранены в буфер')

            else:
                self.logger.info(f'       {type_plot} альфа = {alpha} размер = {" ".join(model_size)} '
                                 f'угол = {angle.rjust(2, "0")} режим = {mode.ljust(4, " ")} не сохранены в буфер')

        else:
            fig = self.clipboard_dict['isolated'][alpha][model_scale][angle][id_fig]
            self.logger.info(f'Запрос {type_plot} альфа = {alpha} размер = {" ".join(model_size)} '
                             f'угол = {angle.rjust(2, "0")} режим = {mode.ljust(4, " ")} из буфера успешно выполнен')

        return fig

    def get_data_summary(self, db='isolated', **kwargs):
        if db == 'isolated':
            mode = kwargs['mode']
            alpha = kwargs['alpha']
            model_scale = kwargs['model_scale']
            angle = kwargs['angle']

            data = dict()
            f_cx = 'cx' in mode
            f_cy = 'cy' in mode
            f_cmz = 'cmz' in mode

            if f_cx or f_cy:
                cx, cy = self.get_cx_cy(db='isolated', angle=angle,
                                        model_scale=model_scale, alpha=alpha)
                if f_cx:
                    data['cx'] = cx
                if f_cy:
                    data['cy'] = cy

            if f_cmz:
                data['cmz'] = self.get_cmz(db='isolated', angle=angle,
                                           model_scale=model_scale, alpha=alpha)

        elif db == 'interference':
            mode = kwargs['mode']
            model_scale = kwargs['model_scale']
            angle = kwargs['angle']
            case = kwargs['case']

            data = dict()
            f_cx = 'cx' in mode
            f_cy = 'cy' in mode
            f_cmz = 'cmz' in mode

            if f_cx or f_cy:
                cx, cy = self.get_cx_cy(db='interference', height=model_scale / 1000, angle=angle,
                                        model_scale=model_scale, case=case)
                if f_cx:
                    data['cx'] = cx
                if f_cy:
                    data['cy'] = cy

            if f_cmz:
                data['cmz'] = self.get_cmz(db='interference', height=model_scale / 1000, angle=angle,
                                           model_scale=model_scale, case=case)

        return data

    def get_summary_spectres(self, db='isolated', **kwargs):
        type_plot = kwargs['type_plot']
        scale = kwargs['scale']
        if db == 'isolated':
            mode = kwargs['mode']
            model_size = kwargs['model_size']
            alpha = kwargs['alpha']
            angle = kwargs['angle']

            mode = mode.lower().replace(' ', '_')
            model_scale, _ = get_model_and_scale_factors(*model_size, alpha)
            id_fig = f'{type_plot}_{mode}_{scale}_{"_".join(model_size)}'

            message = f'{" ".join(list(model_size))} {alpha} {int(angle):02} {mode}'

            self.logger.info(f'Запрос суммарных спектров {message} из буфера')
            if not self.clipboard_dict['isolated'][alpha][model_scale][angle].get(id_fig):
                f_cx = 'cx' in mode
                f_cy = 'cy' in mode
                f_cmz = 'cmz' in mode

                data = self.get_data_summary(db='isolated', alpha=alpha, angle=angle, mode=mode,
                                             model_scale=model_scale)

                self.logger.info(f'Отрисовка суммарных спектров {message}')
                fig = Plot.welch_graphs(db='isolated', data=data, model_size=model_size, alpha=alpha, angle=angle)

                if all((self._save_summary_spectres_cx or f_cx, self._save_summary_spectres_cy or f_cy,
                        self._save_summary_spectres_cmz or f_cmz)):
                    self.clipboard_dict['isolated'][alpha][model_scale][angle][id_fig] = fig
                else:
                    self.logger.info(f'       Cуммарные спектры {message} не сохранены в буфер')
            else:
                fig = self.clipboard_dict['isolated'][alpha][model_scale][angle][id_fig]
                self.logger.info(f'Запрос суммарных спектров {message} из буфера успешно выполнен')
        elif db == 'interference':
            mode = kwargs['mode']
            case = kwargs['case']
            model_size = kwargs['model_size']
            angle = int(kwargs['angle'])

            mode = mode.lower().replace(' ', '_')
            model_scale, _ = get_model_and_scale_factors_interference(*model_size)
            id_fig = f'interference_spec_{mode}_{"_".join(model_size)}'
            if not self.clipboard_dict['interference'][model_scale][case][angle].get(id_fig):
                f_cx = 'cx' in mode
                f_cy = 'cy' in mode
                f_cmz = 'cmz' in mode

                data = self.get_data_summary(db='interference', mode=mode, case=case, angle=angle,
                                             model_scale=model_scale)

                fig = Plot.welch_graphs(db='interference', data=data, model_size=model_size, case=case, angle=angle)

                if all((self._save_summary_spectres_cx or f_cx, self._save_summary_spectres_cy or f_cy,
                        self._save_summary_spectres_cmz or f_cmz)):
                    self.clipboard_dict['interference'][model_scale][case][angle][id_fig] = fig
            else:
                fig = self.clipboard_dict['interference'][model_scale][case][angle][id_fig]

        return fig

    def get_summary_coefficients(self,
                                 db='isolated',
                                 **kwargs):
        if db == 'isolated':
            mode = kwargs['mode']
            model_size = kwargs['model_size']
            alpha = kwargs['alpha']
            angle = kwargs['angle']

            mode = mode.lower()
            model_scale, _ = get_model_and_scale_factors(*model_size, alpha)
            type_plot = 'summary_coefficients'
            id_fig = f'{type_plot}_{mode}_{"_".join(model_size)}'
            if not self.clipboard_dict['isolated'][alpha][model_scale][angle].get(id_fig):
                f_cx = 'cx' in mode
                f_cy = 'cy' in mode
                f_cmz = 'cmz' in mode
                data = self.get_data_summary(db='isolated', alpha=alpha, angle=angle, mode=mode,
                                             model_scale=model_scale)
                fig = Plot.summary_coefficients(db='isolated', alpha=alpha, angle=angle, mode=mode,
                                                model_scale=model_scale, data=data)

                if all((self._save_summary_coefficients_cx or f_cx, self._save_summary_coefficients_cy or f_cy,
                        self._save_summary_coefficients_cmz or f_cmz)):
                    self.clipboard_dict['isolated'][alpha][model_scale][angle][id_fig] = fig
                else:
                    self.logger.info(f'       Cуммарные коэф график не сохранены в буфер')

            else:
                fig = self.clipboard_dict['isolated'][alpha][model_scale][angle][id_fig]
                self.logger.info(f'Запрос суммарных коэф из буфера успешно выполнен')
        elif db == 'interference':
            mode = kwargs['mode']
            model_size = kwargs['model_size']
            case = kwargs['case']
            angle = int(kwargs['angle'])

            mode = mode.lower()
            model_scale, _ = get_model_and_scale_factors_interference(*model_size)
            type_plot = 'summary_coefficients'
            id_fig = f'{type_plot}_{mode}_{"_".join(model_size)}_{case}'
            if not self.clipboard_dict[db][model_scale][case][angle].get(id_fig):
                f_cx = 'cx' in mode
                f_cy = 'cy' in mode
                f_cmz = 'cmz' in mode
                data = self.get_data_summary(db=db, case=case, angle=angle, mode=mode,
                                             model_scale=model_scale)
                fig = Plot.summary_coefficients(db=db, case=case, angle=angle, mode=mode,
                                                model_scale=model_scale, data=data)

                if all((self._save_summary_coefficients_cx or f_cx, self._save_summary_coefficients_cy or f_cy,
                        self._save_summary_coefficients_cmz or f_cmz)):
                    self.clipboard_dict[db][model_scale][case][angle][id_fig] = fig

            else:
                fig = self.clipboard_dict[db][model_scale][case][angle][id_fig]

        return fig

    def get_plot_pressure_tap_locations(self, model_size, alpha):
        id_fig = f'model_polar_{"_".join(model_size)}'
        if not self.clipboard_dict['isolated']['unique_stuff_for_size'].get(id_fig):
            model_scale, scale_factors = get_model_and_scale_factors(*model_size, alpha)
            coordinates = self.get_coordinates('isolated', alpha=alpha, model_scale=model_scale)
            coordinates = converter_coordinates_to_real(*coordinates, model_size, model_scale)
            fig = Plot.model_pic(model_size, model_scale, coordinates)
            if self._save_model_polar:
                self.clipboard_dict['isolated']['unique_stuff_for_size'][id_fig] = fig
            else:
                ...
        else:
            fig = self.clipboard_dict['isolated']['unique_stuff_for_size'][id_fig]

        return fig

    def get_plot_model_3d(self, model_size):
        id_fig = f'model_3d_{"_".join(model_size)}'
        if not self.clipboard_dict['isolated']['unique_stuff_for_size'].get(id_fig):
            fig = Plot.model_3d(model_size)
            if self._save_model_3d:
                self.clipboard_dict['isolated']['unique_stuff_for_size'][id_fig] = fig
            else:
                ...
        else:
            fig = self.clipboard_dict['isolated']['unique_stuff_for_size'][id_fig]

        return fig

    def get_model_polar(self, model_size):
        id_fig = f'model_polar_{"_".join(model_size)}'
        if not self.clipboard_dict['isolated']['unique_stuff_for_size'].get(id_fig):
            fig = Plot.model_polar(model_size)
            if self._save_model_3d:
                self.clipboard_dict['isolated']['unique_stuff_for_size'][id_fig] = fig
            else:
                ...
        else:
            fig = self.clipboard_dict['isolated']['unique_stuff_for_size'][id_fig]

        return fig


if __name__ == '__main__':
    c = Clipboard()
    # print(c.get_face_number('4','111'))
