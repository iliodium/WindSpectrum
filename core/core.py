# local imports
from plot.plot import Plot
from clipboard.clipboard import Clipboard


class Core:
    def __init__(self):
        self.clipboard_obj = Clipboard()

    def get_plots(self, alpha, model_size, angle):
        x, y, z = [10 * i for i in map(int, model_size)]
        model_name = ''.join((x, y, z))
        self.clipboard_obj.get_pressure_coefficients(alpha, model_name, angle)

