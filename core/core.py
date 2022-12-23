import numpy as np
import asyncio
# local imports
from plot.plot import Plot
from clipboard.clipboard import Clipboard
from utils.utils import get_cmz, get_cx_cy


class Core:
    def __init__(self):
        self.clipboard_obj = Clipboard()

    def get_plot_isofileds(self, alpha, model_size, angle, mode, type_plot):
        x, y, z = [str(int(10 * i)) for i in map(float, model_size)]
        model_name = ''.join((x, y, z))
        if not self.clipboard_obj.clipboard_dict[alpha][model_name][angle][type_plot][f'isofields_{mode}']:
            if not self.clipboard_obj.clipboard_dict[alpha][model_name][angle]['pressure_coefficients'].any():
                self.clipboard_obj.clipboard_dict[alpha][model_name][angle]['pressure_coefficients'] = \
                    self.clipboard_obj.get_pressure_coefficients(alpha, model_name, angle)

            pressure_coefficients = self.clipboard_obj.clipboard_dict[alpha][model_name][angle]['pressure_coefficients']

            if not self.clipboard_obj.clipboard_dict[alpha][model_name]['const_parameters']['x'] or \
                    not self.clipboard_obj.clipboard_dict[alpha][model_name]['const_parameters']['z']:
                self.clipboard_obj.clipboard_dict[alpha][model_name]['const_parameters']['x'], \
                self.clipboard_obj.clipboard_dict[alpha][model_name]['const_parameters']['z'] = \
                    self.clipboard_obj.get_coordinates(alpha, model_name)

            coordinates = self.clipboard_obj.clipboard_dict[alpha][model_name]['const_parameters']['x'], \
                          self.clipboard_obj.clipboard_dict[alpha][model_name]['const_parameters']['z']

            if type_plot == 'discrete_isofields':
                fig = Plot.discrete_isofield(model_name, alpha, angle, mode, pressure_coefficients, coordinates)
            elif type_plot == 'integral_isofields':
                fig = Plot.integral_isofield(model_name, alpha, angle, mode, pressure_coefficients, coordinates)
            self.clipboard_obj.clipboard_dict[alpha][model_name][angle][type_plot][f'isofields_{mode}'] = fig

        return self.clipboard_obj.clipboard_dict[alpha][model_name][angle][type_plot][f'isofields_{mode}']

    def get_plot_summary_spectres(self, alpha, model_size, angle, mode, scale, type_plot):
        x, y, z = [str(int(10 * i)) for i in map(float, model_size)]
        model_name = ''.join((x, y, z))
        data = {'Cx': None,
                'Cy': None,
                'CMz': None,
                }
        cx, cy, cmz = None, None, None
        if not self.clipboard_obj.clipboard_dict[alpha][model_name]['const_parameters']['uh_average_wind_speed']:
            self.clipboard_obj.clipboard_dict[alpha][model_name]['const_parameters']['uh_average_wind_speed'] = \
                self.clipboard_obj.get_uh_average_wind_speed(alpha, model_name)
        speed = self.clipboard_obj.clipboard_dict[alpha][model_name]['const_parameters']['uh_average_wind_speed']

        if not self.clipboard_obj.clipboard_dict[alpha][model_name][angle]['pressure_coefficients'].any():
            self.clipboard_obj.clipboard_dict[alpha][model_name][angle]['pressure_coefficients'] = \
                self.clipboard_obj.get_pressure_coefficients(alpha, model_name, angle)

        pressure_coefficients = self.clipboard_obj.clipboard_dict[alpha][model_name][angle]['pressure_coefficients']

        if not self.clipboard_obj.clipboard_dict[alpha][model_name]['const_parameters']['x'] or \
                not self.clipboard_obj.clipboard_dict[alpha][model_name]['const_parameters']['z']:
            self.clipboard_obj.clipboard_dict[alpha][model_name]['const_parameters']['x'], \
            self.clipboard_obj.clipboard_dict[alpha][model_name]['const_parameters']['z'] = \
                self.clipboard_obj.get_coordinates(alpha, model_name)

        coordinates = self.clipboard_obj.clipboard_dict[alpha][model_name]['const_parameters']['x'], \
                      self.clipboard_obj.clipboard_dict[alpha][model_name]['const_parameters']['z']

        if not self.clipboard_obj.clipboard_dict[alpha][model_name][angle][type_plot][f'plot_{mode}_{scale}']:
            t_cx, t_cy = None, None
            if 'Cx' in mode:
                if not self.clipboard_obj.clipboard_dict[alpha][model_name][angle]['Cx'].any():
                    t_cx, t_cy = get_cx_cy(model_name, pressure_coefficients)
                    self.clipboard_obj.clipboard_dict[alpha][model_name][angle]['Cx'] = t_cx
                    cx = t_cx
                else:
                    cx = self.clipboard_obj.clipboard_dict[alpha][model_name][angle]['Cx']

            if 'Cy' in mode:
                if not self.clipboard_obj.clipboard_dict[alpha][model_name][angle]['Cy'].any():
                    if t_cy is None:
                        t_cx, t_cy = get_cx_cy(model_name, pressure_coefficients)
                    self.clipboard_obj.clipboard_dict[alpha][model_name][angle]['Cy'] = t_cy
                    cy = t_cy
                else:
                    cy = self.clipboard_obj.clipboard_dict[alpha][model_name][angle]['Cy']

            if 'CMz' in mode:
                if not self.clipboard_obj.clipboard_dict[alpha][model_name][angle]['CMz'].any():
                    cmz = get_cmz(model_name, pressure_coefficients, coordinates)
                    self.clipboard_obj.clipboard_dict[alpha][model_name][angle]['CMz'] = cmz
                else:
                    cmz = self.clipboard_obj.clipboard_dict[alpha][model_name][angle]['CMz']

            data['Cx'] = cx
            data['Cy'] = cy
            data['CMz'] = cmz

            fig = Plot.welch_graphs(model_name, alpha, angle, speed, scale, mode, data)
            self.clipboard_obj.clipboard_dict[alpha][model_name][angle][type_plot][f'plot_{mode}_{scale}'] = fig

        return self.clipboard_obj.clipboard_dict[alpha][model_name][angle][type_plot][f'plot_{mode}_{scale}']

    def get_plot_summary_coefficients(self, alpha, model_size, angle, mode, type_plot):
        x, y, z = [str(int(10 * i)) for i in map(float, model_size)]
        model_name = ''.join((x, y, z))
        data = {'Cx': None,
                'Cy': None,
                'CMz': None,
                }
        cx, cy, cmz = None, None, None

        if not self.clipboard_obj.clipboard_dict[alpha][model_name][angle]['pressure_coefficients'].any():
            self.clipboard_obj.clipboard_dict[alpha][model_name][angle]['pressure_coefficients'] = \
                self.clipboard_obj.get_pressure_coefficients(alpha, model_name, angle)

        pressure_coefficients = self.clipboard_obj.clipboard_dict[alpha][model_name][angle]['pressure_coefficients']

        if not self.clipboard_obj.clipboard_dict[alpha][model_name][angle][type_plot][f'plot_{mode}']:
            t_cx, t_cy = None, None
            if 'Cx' in mode:
                if not self.clipboard_obj.clipboard_dict[alpha][model_name][angle]['Cx'].any():
                    t_cx, t_cy = get_cx_cy(model_name, pressure_coefficients)
                    self.clipboard_obj.clipboard_dict[alpha][model_name][angle]['Cx'] = t_cx
                    cx = t_cx
                else:
                    cx = self.clipboard_obj.clipboard_dict[alpha][model_name][angle]['Cx']

            if 'Cy' in mode:
                if not self.clipboard_obj.clipboard_dict[alpha][model_name][angle]['Cy'].any():
                    if t_cy is None:
                        t_cx, t_cy = get_cx_cy(model_name, pressure_coefficients)
                    self.clipboard_obj.clipboard_dict[alpha][model_name][angle]['Cy'] = t_cy
                    cy = t_cy
                else:
                    cy = self.clipboard_obj.clipboard_dict[alpha][model_name][angle]['Cy']

            if 'CMz' in mode:
                if not self.clipboard_obj.clipboard_dict[alpha][model_name][angle]['CMz'].any():

                    if not self.clipboard_obj.clipboard_dict[alpha][model_name]['const_parameters']['x'] or \
                            not self.clipboard_obj.clipboard_dict[alpha][model_name]['const_parameters']['z']:
                        self.clipboard_obj.clipboard_dict[alpha][model_name]['const_parameters']['x'], \
                        self.clipboard_obj.clipboard_dict[alpha][model_name]['const_parameters']['z'] = \
                            self.clipboard_obj.get_coordinates(alpha, model_name)

                    coordinates = self.clipboard_obj.clipboard_dict[alpha][model_name]['const_parameters']['x'], \
                                  self.clipboard_obj.clipboard_dict[alpha][model_name]['const_parameters']['z']

                    cmz = get_cmz(model_name, pressure_coefficients, coordinates)

                    self.clipboard_obj.clipboard_dict[alpha][model_name][angle]['CMz'] = cmz
                else:
                    cmz = self.clipboard_obj.clipboard_dict[alpha][model_name][angle]['CMz']

            data['Cx'] = cx
            data['Cy'] = cy
            data['CMz'] = cmz

            fig = Plot.summary_coefficients(model_name, alpha, angle, mode, data)
            self.clipboard_obj.clipboard_dict[alpha][model_name][angle][type_plot][f'plot_{mode}'] = fig
        print(self.clipboard_obj.clipboard_dict[alpha][model_name][angle][type_plot][f'plot_{mode}'].number,'number')
        return self.clipboard_obj.clipboard_dict[alpha][model_name][angle][type_plot][f'plot_{mode}']


if __name__ == '__main__':
    pass
