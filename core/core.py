# local imports
from plot.plot import Plot
from clipboard.clipboard import Clipboard


class Core:
    def __init__(self):
        self.clipboard_obj = Clipboard()

    def get_isofileds(self, alpha, model_size, angle, mode, type_plot):
        x, y, z = [str(int(10 * i)) for i in map(float, model_size)]
        model_name = ''.join((x, y, z))
        if not self.clipboard_obj.clipboard_dict[alpha][model_name][angle][type_plot][f'isofields_{mode}']:
            pressure = self.clipboard_obj.get_pressure_coefficients(alpha, model_name, angle)
            coordinates = self.clipboard_obj.get_coordinates(alpha, model_name)
            if type_plot == 'discrete':
                fig = Plot.discrete_isofield(model_name, mode, pressure, coordinates)
            else:
                fig = Plot.integral_isofield(model_name, mode, pressure, coordinates, alpha)
            self.clipboard_obj.clipboard_dict[alpha][model_name][angle][type_plot][f'isofields_{mode}'] = fig

        return self.clipboard_obj.clipboard_dict[alpha][model_name][angle][type_plot][f'isofields_{mode}']
