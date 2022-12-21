# local imports
from databasetoolkit.databasetoolkit import DataBaseToolkit

from numpy import array


class Clipboard:
    def __init__(self):
        self.clipboard_dict = None
        self.database_obj = DataBaseToolkit()
        self.init_clipboard()

    def init_clipboard(self):
        self.clipboard_dict = {'4': dict(),
                               '6': dict()
                               }

        # параметры уникальный для каждого угла атаки
        parameters = {'pressure_coefficients': array([]),
                      'discrete': {
                          'isofields_min': None,
                          'isofields_mean': None,
                          'isofields_max': None,
                          'isofields_std': None,
                      },
                      'integral': {
                          'isofields_min': None,
                          'isofields_mean': None,
                          'isofields_max': None,
                          'isofields_std': None,
                      },
                      }

        # параметры уникальный для каждого эксперимента
        const_parameters = {'x': None,
                            'z': None,
                            'face_number': None,
                            'uh_average_wind_speed': None
                            }
        experiments = self.database_obj.get_experiments()
        self.clipboard_dict['4'] = experiments['4']
        self.clipboard_dict['6'] = experiments['6']

        for alpha in self.clipboard_dict.keys():
            for model_name in self.clipboard_dict[alpha].keys():
                for angle in self.clipboard_dict[alpha][model_name].keys():
                    self.clipboard_dict[alpha][model_name][angle] = parameters
                self.clipboard_dict[alpha][model_name]['const_parameters'] = const_parameters

    def get_pressure_coefficients(self, alpha, model_name, angle):
        if not self.clipboard_dict[alpha][model_name][angle]['pressure_coefficients'].any():
            self.clipboard_dict[alpha][model_name][angle]['pressure_coefficients'] = \
                array(self.database_obj.get_pressure_coefficients(alpha, model_name, angle))
        return self.clipboard_dict[alpha][model_name][angle]['pressure_coefficients'] / 1000

    def get_coordinates(self, alpha, model_name):
        if not self.clipboard_dict[alpha][model_name]['const_parameters']['x'] and \
                not self.clipboard_dict[alpha][model_name]['const_parameters']['z']:
            self.clipboard_dict[alpha][model_name]['const_parameters']['x'], \
            self.clipboard_dict[alpha][model_name]['const_parameters']['z'] = \
                self.database_obj.get_coordinates(alpha, model_name)
        return self.clipboard_dict[alpha][model_name]['const_parameters']['x'], \
               self.clipboard_dict[alpha][model_name]['const_parameters']['z']

    def get_uh_average_wind_speed(self, alpha, model_name):
        if not self.clipboard_dict[alpha][model_name]['const_parameters']['uh_average_wind_speed']:
            self.clipboard_dict[alpha][model_name]['const_parameters']['uh_average_wind_speed'] = \
                self.database_obj.get_uh_average_wind_speed(alpha, model_name)
        return self.clipboard_dict[alpha][model_name]['const_parameters']['uh_average_wind_speed']

    def get_face_number(self, alpha, model_name):
        if not self.clipboard_dict[alpha][model_name]['const_parameters']['face_number']:
            self.clipboard_dict[alpha][model_name]['const_parameters']['face_number'] = \
                self.database_obj.get_face_number(alpha, model_name)
        return self.clipboard_dict[alpha][model_name]['const_parameters']['face_number']


if __name__ == '__main__':
    d = Clipboard()
    print(d.clipboard_dict)
    # print(d.get_pressure_coefficients('4', '111', '0')[:10])
    print('---')
    print(d.get_coordinates('4', '111'))
