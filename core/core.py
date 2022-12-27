import numpy as np
import asyncio
# local imports
from plot.plot import Plot
from clipboard.clipboard import Clipboard


class Core:
    def __init__(self):
        self.clipboard_obj = Clipboard()

    def get_plot_isofileds(self, alpha, model_size, angle, mode, type_plot):
        x, y, z = [str(int(10 * i)) for i in map(float, model_size)]
        model_name = ''.join((x, y, z))
        if not self.clipboard_obj.clipboard_dict[alpha][model_name][angle][type_plot][f'isofields_{mode}']:

            pressure_coefficients = self.clipboard_obj.get_pressure_coefficients(alpha, model_name, angle)
            coordinates = self.clipboard_obj.get_coordinates(alpha, model_name)

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

        speed = self.clipboard_obj.get_uh_average_wind_speed(alpha, model_name)

        if not self.clipboard_obj.clipboard_dict[alpha][model_name][angle][type_plot][f'plot_{mode}_{scale}']:
            cx, cy, cmz = None, None, None

            if 'Cx' in mode or 'Cy' in mode:
                cx, cy = self.clipboard_obj.get_cx_cy(alpha, model_name, angle)
                if 'Cx' not in mode:
                    cx = None
                if 'Cy' not in mode:
                    cy = None

            if 'CMz' in mode:
                cmz = self.clipboard_obj.get_cmz(alpha, model_name, angle)

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

        if not self.clipboard_obj.clipboard_dict[alpha][model_name][angle][type_plot][f'plot_{mode}']:
            cx, cy, cmz = None, None, None

            if 'Cx' in mode or 'Cy' in mode:
                cx, cy = self.clipboard_obj.get_cx_cy(alpha, model_name, angle)
                if 'Cx' not in mode:
                    cx = None
                if 'Cy' not in mode:
                    cy = None

            if 'CMz' in mode:
                cmz = self.clipboard_obj.get_cmz(alpha, model_name, angle)

            data['Cx'] = cx
            data['Cy'] = cy
            data['CMz'] = cmz

            fig = Plot.summary_coefficients(model_name, alpha, angle, mode, data)
            self.clipboard_obj.clipboard_dict[alpha][model_name][angle][type_plot][f'plot_{mode}'] = fig

        return self.clipboard_obj.clipboard_dict[alpha][model_name][angle][type_plot][f'plot_{mode}']


if __name__ == '__main__':
    print(1)
