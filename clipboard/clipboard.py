import os
import pickle
import logging
from typing import Tuple
from multiprocessing import Manager

from numpy import array, any

# local imports
from utils.utils import calculate_cx_cy, calculate_cmz, changer_sequence_numbers, changer_sequence_coefficients
from databasetoolkit.databasetoolkit import DataBaseToolkit


class Clipboard:
    logger = logging.getLogger('Clipboard'.ljust(15, ' '))
    logger.setLevel(logging.INFO)

    # настройка обработчика и форматировщика
    py_handler = logging.FileHandler("log.log", mode='a')
    py_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")

    # добавление форматировщика к обработчику
    py_handler.setFormatter(py_formatter)
    # добавление обработчика к логгеру
    logger.addHandler(py_handler)

    def __init__(self, ex_clipboard = None):
        self.logger.info('Создание буфера')
        self.database_obj = DataBaseToolkit()
        if ex_clipboard:
            self.clipboard_dict = ex_clipboard
            self.logger.info('Буфер создан на основе переданного')

        elif os.path.exists('clipboard\\clipboard.pkl'):
            self.clipboard_dict = dict()
            file = open('clipboard\\clipboard.pkl', 'rb')
            local_clipboard = pickle.load(file)
            self.clipboard_dict.update(local_clipboard)
            file.close()
            self.logger.info('Буфер создан на основе локального файла')

        else:
            self.manager = Manager()
            self.init_clipboard_dict()
            self.logger.info('Буфер успешно создан')

    def init_clipboard_dict(self):
        """Создание буфера"""
        self.clipboard_dict = self.manager.dict({'4': self.manager.dict(),
                                                 '6': self.manager.dict()
                                                 })
        experiments = self.database_obj.get_experiments(self.manager.dict)
        # print(experiments['4'])
        # print(experiments['6'])
        self.clipboard_dict['4'] = experiments['4']
        self.clipboard_dict['6'] = experiments['6']

        for alpha in self.clipboard_dict.keys():
            for model_name in self.clipboard_dict[alpha].keys():
                for angle in self.clipboard_dict[alpha][model_name].keys():
                    self.clipboard_dict[alpha][model_name][angle] = self.manager.dict()

                self.clipboard_dict[alpha][model_name]['const_parameters'] = self.manager.dict()
                self.clipboard_dict[alpha][model_name]['model_attributes'] = self.manager.dict()

    def get_pressure_coefficients(self, alpha: str, model_name: str, angle: str):
        """Возвращает коэффициенты давления из буфера"""
        self.logger.info(f"Запрос коэффициентов давления "
                         f"модель = {model_name} альфа = {alpha} угол = {angle.rjust(2, '0')} из буфера")

        model_name_n = None
        f = False

        if model_name[1] in ['2', '3']:
            model_name_n = model_name
            model_name = model_name[1] + model_name[0] + model_name[2]
            f = True

        if not any(self.clipboard_dict[alpha][model_name][angle].get('pressure_coefficients')):
            self.clipboard_dict[alpha][model_name][angle]['pressure_coefficients'] = \
                array(self.database_obj.get_pressure_coefficients(alpha, model_name, angle))
        pressure_coefficients = self.clipboard_dict[alpha][model_name][angle]['pressure_coefficients'] / 1000

        self.logger.info(f"Запрос коэффициентов давления модель = {model_name} "
                         f"альфа = {alpha} угол = {angle.rjust(2, '0')} из буфера успешно выполнен")
        if f:
            pressure_coefficients = changer_sequence_coefficients(pressure_coefficients, 'forward', model_name_n,
                                                                  (3, 0, 1, 2), f)

        return pressure_coefficients

    def get_coordinates(self, alpha: str, model_scale: str):
        """Возвращает координаты датчиков из буфера"""
        self.logger.info(f"Запрос координаты датчиков модель = {model_scale} альфа = {alpha} из буфера")

        if model_scale[1] in ['2', '3']:
            model_scale = model_scale[1] + model_scale[0] + model_scale[2]

        if not self.clipboard_dict[alpha][model_scale]['const_parameters'].get('x') or \
                not self.clipboard_dict[alpha][model_scale]['const_parameters'].get('z'):
            self.clipboard_dict[alpha][model_scale]['const_parameters']['x'], \
            self.clipboard_dict[alpha][model_scale]['const_parameters']['z'] = \
                self.database_obj.get_coordinates(alpha, model_scale)

        self.logger.info(f"Запрос координат датчиков модель = {model_scale} альфа = {alpha} из буфера успешно выполнен")

        return [self.clipboard_dict[alpha][model_scale]['const_parameters']['x'],
                self.clipboard_dict[alpha][model_scale]['const_parameters']['z']]

    def get_uh_average_wind_speed(self, alpha: str, model_name: str):
        """Возвращает среднюю скорость ветра из буфера"""
        self.logger.info(f"Запрос средней скорости ветра модель = {model_name} альфа = {alpha} из буфера")

        if not self.clipboard_dict[alpha][model_name]['const_parameters'].get('uh_average_wind_speed'):
            self.clipboard_dict[alpha][model_name]['const_parameters']['uh_average_wind_speed'] = \
                self.database_obj.get_uh_average_wind_speed(alpha, model_name)

        self.logger.info(f"Запрос редней скорости ветра "
                         f"модель = {model_name} альфа = {alpha} из буфера успешно выполнен")

        return self.clipboard_dict[alpha][model_name]['const_parameters']['uh_average_wind_speed']

    def get_face_number(self, alpha: str, model_name: str):
        """Возвращает нумерацию датчиков из буфера"""
        self.logger.info(f"Запрос нумерации датчиков модель = {model_name} альфа = {alpha} из буфера")

        model_name_n = None
        f = False

        if model_name[1] in ['2', '3']:
            model_name_n = model_name
            model_name = model_name[1] + model_name[0] + model_name[2]
            f = True

        if not self.clipboard_dict[alpha][model_name]['const_parameters'].get('face_number'):
            self.clipboard_dict[alpha][model_name]['const_parameters']['face_number'] = \
                self.database_obj.get_face_number(alpha, model_name)

        self.logger.info(f"Запрос нумерации датчиков модель = {model_name} альфа = {alpha} из буфера успешно выполнен")

        numbers = self.clipboard_dict[alpha][model_name]['const_parameters']['face_number']

        if f:
            numbers = changer_sequence_numbers(numbers, model_name_n, (3, 0, 1, 2))

        return numbers

    def get_cx_cy(self, alpha: str, model_name: str, angle: str):
        """Возвращает суммарные коэффициенты Cx Cy из буфера"""
        self.logger.info(f"Запрос суммарных коэффициентов Cx Cy "
                         f"модель = {model_name} альфа = {alpha} угол = {angle.rjust(2, '0')} из буфера")

        if not any(self.clipboard_dict[alpha][model_name][angle].get('Cx')) or \
                not any(self.clipboard_dict[alpha][model_name][angle].get('Cy')):
            coefficients = self.get_pressure_coefficients(alpha, model_name, angle)
            cx, cy = calculate_cx_cy(model_name, coefficients)
            self.clipboard_dict[alpha][model_name][angle]['Cx'] = cx
            self.clipboard_dict[alpha][model_name][angle]['Cy'] = cy

        self.logger.info(f"Запрос суммарных коэффициентов Cx Cy модель = {model_name} альфа = {alpha} "
                         f"угол = {angle.rjust(2, '0')} из буфера успешно выполнен")

        return self.clipboard_dict[alpha][model_name][angle]['Cx'], self.clipboard_dict[alpha][model_name][angle]['Cy']

    def get_cmz(self, alpha: str, model_name: str, angle: str):
        """Возвращает CMz из буфера"""
        self.logger.info(f"Запрос CMz модель = {model_name} альфа = {alpha} угол = {angle.rjust(2, '0')} из буфера")

        if not any(self.clipboard_dict[alpha][model_name][angle].get('CMz')):
            coefficients = self.get_pressure_coefficients(alpha, model_name, angle)
            coordinates = self.get_coordinates(alpha, model_name)
            cmz = calculate_cmz(model_name, coefficients, coordinates)
            self.clipboard_dict[alpha][model_name][angle]['CMz'] = cmz

        self.logger.info(f"Запрос CMz модель = {model_name} альфа = {alpha} "
                         f"угол = {angle.rjust(2, '0')} из буфера успешно выполнен")

        return self.clipboard_dict[alpha][model_name][angle]['CMz']


if __name__ == '__main__':
    c = Clipboard()
    # print(c.get_face_number('4','111'))
    print(c.get_face_number('4', '132'))
    # from concurrent.futures import ThreadPoolExecutor
    #
    # d = Clipboard()
    # angles = [str(i) for i in range(0, 50, 5)][:2]
    # a = [('6', '115', i) for i in angles]
    # # d.get_cmz('6', '115','0')
    # with ThreadPoolExecutor(max_workers=2) as executor:
    #     q = list(executor.map(lambda p: d.get_pressure_coefficients(*p), a))
    # print(len(q))
    # print(q[0])
    # print(q[1])
    1
