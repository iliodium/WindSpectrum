# local imports
from utils.utils import calculate_cx_cy, calculate_cmz
from databasetoolkit.databasetoolkit import DataBaseToolkit

from copy import deepcopy
from numpy import array, any


class Clipboard:
    def __init__(self):
        self.clipboard_dict = None
        self.database_obj = DataBaseToolkit()
        self.init_clipboard_dict()

    def init_clipboard_dict(self):
        self.clipboard_dict = {'4': dict(),
                               '6': dict()
                               }

        experiments = self.database_obj.get_experiments()
        self.clipboard_dict['4'] = experiments['4']
        self.clipboard_dict['6'] = experiments['6']

        for alpha in self.clipboard_dict.keys():
            for model_name in self.clipboard_dict[alpha].keys():
                for angle in self.clipboard_dict[alpha][model_name].keys():
                    self.clipboard_dict[alpha][model_name][angle] = dict()
                self.clipboard_dict[alpha][model_name]['const_parameters'] = dict()

    def get_pressure_coefficients(self, alpha: str, model_name: str, angle: str):
        if not any(self.clipboard_dict[alpha][model_name][angle].get('pressure_coefficients')):
            self.clipboard_dict[alpha][model_name][angle]['pressure_coefficients'] = \
                array(self.database_obj.get_pressure_coefficients(alpha, model_name, angle))
        return self.clipboard_dict[alpha][model_name][angle]['pressure_coefficients'] / 1000

    def get_coordinates(self, alpha: str, model_name: str):
        if not self.clipboard_dict[alpha][model_name]['const_parameters'].get('x') or \
                not self.clipboard_dict[alpha][model_name]['const_parameters'].get('z'):
            self.clipboard_dict[alpha][model_name]['const_parameters']['x'], \
            self.clipboard_dict[alpha][model_name]['const_parameters']['z'] = \
                self.database_obj.get_coordinates(alpha, model_name)
        return self.clipboard_dict[alpha][model_name]['const_parameters']['x'], \
               self.clipboard_dict[alpha][model_name]['const_parameters']['z']

    def get_uh_average_wind_speed(self, alpha: str, model_name: str):
        if not self.clipboard_dict[alpha][model_name]['const_parameters'].get('uh_average_wind_speed'):
            self.clipboard_dict[alpha][model_name]['const_parameters']['uh_average_wind_speed'] = \
                self.database_obj.get_uh_average_wind_speed(alpha, model_name)
        return self.clipboard_dict[alpha][model_name]['const_parameters']['uh_average_wind_speed']

    def get_face_number(self, alpha: str, model_name: str):
        if not self.clipboard_dict[alpha][model_name]['const_parameters'].get('face_number'):
            self.clipboard_dict[alpha][model_name]['const_parameters']['face_number'] = \
                self.database_obj.get_face_number(alpha, model_name)
        return self.clipboard_dict[alpha][model_name]['const_parameters']['face_number']

    def get_cx_cy(self, alpha: str, model_name: str, angle: str):
        if not any(self.clipboard_dict[alpha][model_name][angle].get('Cx')) or \
                not any(self.clipboard_dict[alpha][model_name][angle].get('Cy')):
            coefficients = self.get_pressure_coefficients(alpha, model_name, angle)
            cx, cy = calculate_cx_cy(model_name, coefficients)
            self.clipboard_dict[alpha][model_name][angle]['Cx'] = cx
            self.clipboard_dict[alpha][model_name][angle]['Cy'] = cy

        return self.clipboard_dict[alpha][model_name][angle]['Cx'], self.clipboard_dict[alpha][model_name][angle]['Cy']

    def get_cmz(self, alpha: str, model_name: str, angle: str):
        if not any(self.clipboard_dict[alpha][model_name][angle].get('CMz')):
            coefficients = self.get_pressure_coefficients(alpha, model_name, angle)
            coordinates = self.get_coordinates(alpha, model_name)
            cmz = calculate_cmz(model_name, coefficients, coordinates)
            self.clipboard_dict[alpha][model_name][angle]['CMz'] = cmz

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
