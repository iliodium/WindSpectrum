# local imports
from databasetoolkit.databasetoolkit import DataBaseToolkit


class Clipboard:
    def __init__(self):
        self.clipboard_dict = None
        self.database_obj = DataBaseToolkit()
        self.init_clipboard()

    def init_clipboard(self):
        self.clipboard_dict = {'4': dict(),
                               '6': dict()
                               }
        experiments = self.database_obj.get_experiments()
        self.clipboard_dict['4'] = experiments['4']
        self.clipboard_dict['6'] = experiments['6']

    def get_pressure_coefficients(self, alpha, model_name, angle):
        if not self.clipboard_dict[alpha][model_name][angle]['pressure_coefficients']:
            self.clipboard_dict[alpha][model_name][angle]['pressure_coefficients'] = \
                self.database_obj.get_pressure_coefficients(alpha, model_name, angle)
        return self.clipboard_dict[alpha][model_name][angle]['pressure_coefficients']


if __name__ == '__main__':
    d = Clipboard()
    print(d.get_pressure_coefficients('4', '111', '0')[:10])
    print('---')
    print(d.get_pressure_coefficients('4', '111', '0')[:10])
