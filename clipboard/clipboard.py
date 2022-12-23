# local imports
from databasetoolkit.databasetoolkit import DataBaseToolkit

from copy import deepcopy
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

        # параметры уникальный для каждого угла атаки эксперимента
        parameters = {'pressure_coefficients': array([]),
                      'discrete_isofields': {
                          'isofields_min': None,
                          'isofields_mean': None,
                          'isofields_max': None,
                          'isofields_std': None,
                      },
                      'integral_isofields': {
                          'isofields_min': None,
                          'isofields_mean': None,
                          'isofields_max': None,
                          'isofields_std': None,
                      },
                      'CMz': array([]),
                      'Cx': array([]),
                      'Cy': array([]),
                      'summary_coefficients': {
                          'plot_Cx': None,
                          'plot_Cy': None,
                          'plot_CMz': None,
                          'plot_Cx|Cy': None,
                          'plot_Cx|CMz': None,
                          'plot_Cy|CMz': None,
                          'plot_Cx|Cy|CMz': None,
                      },
                      'summary_spectres': {

                          'plot_Cx_log': None,
                          'plot_Cy_log': None,
                          'plot_CMz_log': None,
                          'plot_Cx|Cy_log': None,
                          'plot_Cx|CMz_log': None,
                          'plot_Cy|CMz_log': None,
                          'plot_Cx|Cy|CMz_log': None,

                          'plot_Cx_linear': None,
                          'plot_Cy_linear': None,
                          'plot_CMz_linear': None,
                          'plot_Cx|Cy_linear': None,
                          'plot_Cx|CMz_linear': None,
                          'plot_Cy|CMz_linear': None,
                          'plot_Cx|Cy|CMz_linear': None,
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
                    self.clipboard_dict[alpha][model_name][angle] = deepcopy(parameters)
                self.clipboard_dict[alpha][model_name]['const_parameters'] = deepcopy(const_parameters)

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
    d.clipboard_dict['4']['111']['0']['discrete_isofields']['isofields_min'] = 10
    print(d.clipboard_dict)
