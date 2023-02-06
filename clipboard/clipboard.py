# local imports
from utils.utils import calculate_cx_cy, calculate_cmz
from databasetoolkit.databasetoolkit import DataBaseToolkit

from numpy import array, any
from typing import Tuple


class Clipboard:
    def __init__(self):
        self.clipboard_dict = None
        self.database_obj = DataBaseToolkit()
        # self.manager = self.database_obj.manager
        self.init_clipboard_dict()

    def init_clipboard_dict(self):
        """Создание буфера"""
        self.clipboard_dict = dict({'4': dict(),
                                    '6': dict()
                                    })

        experiments = self.database_obj.get_experiments()
        self.clipboard_dict['4'] = experiments['4']
        self.clipboard_dict['6'] = experiments['6']

        for alpha in self.clipboard_dict.keys():
            for model_name in self.clipboard_dict[alpha].keys():
                for angle in self.clipboard_dict[alpha][model_name].keys():
                    self.clipboard_dict[alpha][model_name][angle] = dict()

                self.clipboard_dict[alpha][model_name]['const_parameters'] = dict()
                self.clipboard_dict[alpha][model_name]['model_attributes'] = dict()

    def get_pressure_coefficients(self, alpha: str, model_name: str, angle: str):
        """Возвращает коэффициенты давления из буфера"""
        print(f"Запрос коэффициентов давления модель = {model_name} альфа = {alpha} угол = {angle} из буфера")

        if not any(self.clipboard_dict[alpha][model_name][angle].get('pressure_coefficients')):
            self.clipboard_dict[alpha][model_name][angle]['pressure_coefficients'] = \
                array(self.database_obj.get_pressure_coefficients(alpha, model_name, angle))
        pressure_coefficients = self.clipboard_dict[alpha][model_name][angle]['pressure_coefficients'] / 1000

        print(f"Запрос коэффициентов давления "
              f"модель = {model_name} альфа = {alpha} угол = {angle} из буфера успешно выполнен")

        return pressure_coefficients

    def get_coordinates(self, alpha: str, model_scale: str):
        """Возвращает координаты датчиков из буфера"""
        print(f"Запрос координаты датчиков модель = {model_scale} альфа = {alpha} из буфера")

        if not self.clipboard_dict[alpha][model_scale]['const_parameters'].get('x') or \
                not self.clipboard_dict[alpha][model_scale]['const_parameters'].get('z'):
            self.clipboard_dict[alpha][model_scale]['const_parameters']['x'], \
            self.clipboard_dict[alpha][model_scale]['const_parameters']['z'] = \
                self.database_obj.get_coordinates(alpha, model_scale)

        print(f"Запрос координат датчиков модель = {model_scale} альфа = {alpha} из буфера успешно выполнен")

        return self.clipboard_dict[alpha][model_scale]['const_parameters']['x'], \
               self.clipboard_dict[alpha][model_scale]['const_parameters']['z']

    def get_uh_average_wind_speed(self, alpha: str, model_name: str):
        """Возвращает среднюю скорость ветра из буфера"""
        print(f"Запрос средней скорости ветра модель = {model_name} альфа = {alpha} из буфера")

        if not self.clipboard_dict[alpha][model_name]['const_parameters'].get('uh_average_wind_speed'):
            self.clipboard_dict[alpha][model_name]['const_parameters']['uh_average_wind_speed'] = \
                self.database_obj.get_uh_average_wind_speed(alpha, model_name)

        print(f"Запрос редней скорости ветра модель = {model_name} альфа = {alpha} из буфера успешно выполнен")

        return self.clipboard_dict[alpha][model_name]['const_parameters']['uh_average_wind_speed']

    def get_face_number(self, alpha: str, model_name: str):
        """Возвращает нумерацию датчиков из буфера"""
        print(f"Запрос нумерации датчиков модель = {model_name} альфа = {alpha} из буфера")

        if not self.clipboard_dict[alpha][model_name]['const_parameters'].get('face_number'):
            self.clipboard_dict[alpha][model_name]['const_parameters']['face_number'] = \
                self.database_obj.get_face_number(alpha, model_name)

        print(f"Запрос нумерации датчиков модель = {model_name} альфа = {alpha} из буфера успешно выполнен")

        return self.clipboard_dict[alpha][model_name]['const_parameters']['face_number']

    def get_cx_cy(self, alpha: str, model_name: str, angle: str):
        """Возвращает суммарные коэффициенты Cx Cy из буфера"""
        print(f"Запрос суммарных коэффициентов Cx Cy модель = {model_name} альфа = {alpha} угол = {angle} из буфера")

        if not any(self.clipboard_dict[alpha][model_name][angle].get('Cx')) or \
                not any(self.clipboard_dict[alpha][model_name][angle].get('Cy')):
            coefficients = self.get_pressure_coefficients(alpha, model_name, angle)
            cx, cy = calculate_cx_cy(model_name, coefficients)
            self.clipboard_dict[alpha][model_name][angle]['Cx'] = cx
            self.clipboard_dict[alpha][model_name][angle]['Cy'] = cy

        print(f"Запрос суммарных коэффициентов Cx Cy "
              f"модель = {model_name} альфа = {alpha} угол = {angle} из буфера успешно выполнен")

        return self.clipboard_dict[alpha][model_name][angle]['Cx'], self.clipboard_dict[alpha][model_name][angle]['Cy']

    def get_cmz(self, alpha: str, model_name: str, angle: str, model_size: Tuple[str, str, str]):
        """Возвращает CMz из буфера"""
        print(f"Запрос CMz модель = {model_name} размеры = {' '.join(model_size)} "
              f"альфа = {alpha} угол = {angle} из буфера")

        if not any(self.clipboard_dict[alpha][model_name][angle].get('CMz')):
            coefficients = self.get_pressure_coefficients(alpha, model_name, angle)
            coordinates = self.get_coordinates(alpha, model_name)
            cmz = calculate_cmz(model_name, model_size, coefficients, coordinates)
            self.clipboard_dict[alpha][model_name][angle]['CMz'] = cmz

        print(f"Запрос CMz модель = {model_name} размеры = {' '.join(model_size)} "
              f"альфа = {alpha} угол = {angle} из буфера успешно выполнен")

        return self.clipboard_dict[alpha][model_name][angle]['CMz']


if __name__ == '__main__':
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
    print(len([str(i) for i in range(0, 50, 5)]))
