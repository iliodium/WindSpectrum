<<<<<<< HEAD
import os
import time
import numpy as np
from typing import Tuple
from docx import Document
from multiprocessing import Process
from docx.shared import Inches, Pt, Mm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from concurrent.futures import ThreadPoolExecutor

# local imports
from plot.plot import Plot
from utils.utils import *
=======
# local imports
>>>>>>> b28311c22755b3806c0ef8b7874d2ae258d6b3ea
from clipboard.clipboard import Clipboard
from plot.plot import Plot


class Core:
    def __init__(self):
        self.clipboard_obj = Clipboard()

    def get_plot_isofields(self, alpha: str, model_size: Tuple[str, str, str], angle: str, mode: str, type_plot: str):
        fig = None
        model_scale, scale_factors = get_model_and_scale_factors(*model_size, alpha)

        id_fig = f'{type_plot}_{mode}_{"_".join(model_size)}'

        if not self.clipboard_obj.clipboard_dict[alpha][model_scale][angle].get(id_fig):

            pressure_coefficients = self.clipboard_obj.get_pressure_coefficients(alpha, model_scale, angle)
            coordinates = self.clipboard_obj.get_coordinates(alpha, model_scale)

            if type_plot == 'discrete_isofields':
                fig = Plot.discrete_isofield(model_scale, mode, pressure_coefficients, coordinates)
            elif type_plot == 'integral_isofields':
                fig = Plot.integral_isofield(model_scale,
                                             model_size,
                                             scale_factors,
                                             alpha,
                                             mode,
                                             pressure_coefficients,
                                             coordinates,
                                             )

            self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][id_fig] = fig
        fig = self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][id_fig]

        return fig

    def get_plot_summary_spectres(self, alpha: str, model_size: tuple, angle: str, mode: str, scale, type_plot: str):
        mode = mode.replace(' ', '_')
        model_scale, scale_factors = get_model_and_scale_factors(*model_size, alpha)
        id_fig = f'{type_plot}_{mode}_{scale}_{"_".join(model_size)}'
        data = dict()

        if not self.clipboard_obj.clipboard_dict[alpha][model_scale][angle].get(id_fig):
            if 'Cx' in mode or 'Cy' in mode:
                cx, cy = self.clipboard_obj.get_cx_cy(alpha, model_scale, angle)
                if 'Cx' in mode:
                    data['Cx'] = cx
                if 'Cy' in mode:
                    data['Cy'] = cy

            if 'CMz' in mode:
                data['CMz'] = self.clipboard_obj.get_cmz(alpha, model_scale, angle)

            fig = Plot.welch_graphs(data)
            self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][id_fig] = fig

        fig = self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][id_fig]

        return fig

    def get_plot_summary_coefficients(self, alpha: str, model_size: tuple, angle: str, mode: str, type_plot: str):
        model_scale, scale_factors = get_model_and_scale_factors(*model_size, alpha)
        id_fig = f'{type_plot}_{mode}_{"_".join(model_size)}'
        data = dict()

        if not self.clipboard_obj.clipboard_dict[alpha][model_scale][angle].get(id_fig):
            if 'Cx' in mode or 'Cy' in mode:
                cx, cy = self.clipboard_obj.get_cx_cy(alpha, model_scale, angle)
                if 'Cx' in mode:
                    data['Cx'] = cx
                if 'Cy' in mode:
                    data['Cy'] = cy

            if 'CMz' in mode:
                data['CMz'] = self.clipboard_obj.get_cmz(alpha, model_scale, angle)

            fig = Plot.summary_coefficients(data)
            self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][id_fig] = fig

        fig = self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][id_fig]
        return fig

    def get_envelopes(self, alpha: str, model_scale: str, angle: str):
        if not self.clipboard_obj.clipboard_dict[alpha][model_scale][angle].get('envelopes'):
            pressure_coefficients = self.clipboard_obj.get_pressure_coefficients(alpha, model_scale, angle)
            figs = Plot.envelopes(pressure_coefficients)
            for ind, val in enumerate(figs):
                self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][f'envelopes_{ind}'] = val
            self.clipboard_obj.clipboard_dict[alpha][model_scale][angle]['envelopes'] = True

        figs = []
        i = 0
        while True:
            if not self.clipboard_obj.clipboard_dict[alpha][model_scale][angle].get(f'envelopes_{i}'):
                break
            else:
                figs.append(self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][f'envelopes_{i}'])
                i += 1
        return figs

    def envelopes_thread(self, alpha: str, model_scale: str, angle: str, name_report):
        """функция для запуска потока генерации графиков огибающих модели и их сохранения."""
        figs = self.get_envelopes(alpha, model_scale, angle)

        path_folder = f'{os.getcwd()}\\{name_report}\\Огибающие\\Огибающие {model_scale} {alpha} {int(angle):02}'

        if not os.path.isdir(path_folder):
            os.mkdir(path_folder)
        for i in range(len(figs)):
            plot_name = f'Огибающие {model_scale} {alpha} {int(angle):02} {i * 100} - {(i + 1) * 100}'
            figs[i].savefig(f'{path_folder}\\{plot_name}')

    def welch_graphs_thread(self, alpha: str, model_scale: str, angle: str, name_report):
        """функция для запуска потока генерации графиков спектральной плотности мощности и их сохранения."""
        sum_cmz = self.clipboard_obj.get_cmz(alpha, model_scale, angle)
        sum_cx, sum_cy = self.clipboard_obj.get_cx_cy(alpha, model_scale, angle)

        fig = Plot.old_welch_graphs(sum_cx, sum_cy, sum_cmz)

        path_folder = f'{os.getcwd()}\\{name_report}\\Спектральная плотность мощности\\Логарифмическая шкала'
        plot_name = f'Спектральная плотность мощности {model_scale} {alpha} {int(angle):02}'

        fig.savefig(f'{path_folder}\\{plot_name}')

    def report(self, alpha: str, model_size: Tuple[str, str, str]):
        """Создание отчёта для выбранной конфигурации.
        Отчёт включает:
            огибающие,
            суммарные аэродинамические коэффициенты,
            спектры,
            характеристика по датчикам,
        """
        print('REPORT')
        t1 = time.time()
        model_scale, scale_factors = get_model_and_scale_factors(*model_size, alpha)
        breadth, depth, height = model_size
        name_report = f'Отчет ширина {breadth} глубина {depth} высота {height} альфа {alpha}'
        path_report = f'{os.getcwd()}\\{name_report}'
        generate_directory_for_report(name_report)

        if model_scale[0] == model_scale[1]:
            angle_border = 50
        else:
            angle_border = 95

        angles = [str(i) for i in range(0, angle_border, 5)]

        args_pressure_coefficients = [(alpha, model_scale, i) for i in angles]

        # with ThreadPoolExecutor(max_workers=12) as executor:
        #     executor.map(lambda i: self.clipboard_obj.get_pressure_coefficients(*i), args_pressure_coefficients)

        print('Начало формирования отчета')

        # # Отображение модели
        # print('Отображение модели')
        # coordinates = self.clipboard_obj.get_coordinates(alpha, model_scale)
        # fig = Plot.model_pic(model_scale, coordinates)
        # fig.savefig(f'{path_report}\\Модель\\Развертка модели')
        # fig = Plot.model_polar(model_scale)
        # fig.savefig(f'{path_report}\\Модель\\Модель в полярной системе координат')
        # fig = Plot.model_cube(model_scale)
        # fig.savefig(f'{path_report}\\Модель\\Модель 3D')
        # # Изополя
        # print('Изополя')
        # type_plot = 'integral_isofields'
        # for angle in range(0, 50, 5):
        #     angle = str(angle)
        #     for mode in ('max',
        #                  'mean',
        #                  'min',
        #                  'std',
        #                  ):
        #         fig = self.get_plot_isofields(alpha, model_size, angle, mode, type_plot)
        #         file_name = f'Изополя {model_scale} {alpha} {int(angle):02} {mode}'
        #         fig.savefig(f'{path_report}\\Изополя ветровых нагрузок и воздействий\\Непрерывные\\'
        #                     f'{mode.upper()}\\{file_name}')
        #
        # # Огибающие
        # print('Огибающие')
        #
        # args_envelopes = [(alpha, model_scale, i, name_report) for i in angles]
        #
        # with ThreadPoolExecutor(max_workers=10) as executor:
        #     executor.map(lambda i: self.envelopes_thread(*i), args_envelopes)

        # Спектральная плотность мощности суммарных аэродинамических коэффициентов
        print('Спектральная плотность мощности суммарных аэродинамических коэффициентов')
        args_welch_graphs = [(alpha, model_scale, i, name_report) for i in angles]

        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(lambda i: self.welch_graphs_thread(*i), args_welch_graphs)

        # # подготовка к суммарным
        # args_summary_coefficients = [(alpha, model_scale, i) for i in angles]
        # with ThreadPoolExecutor(max_workers=10) as executor:
        #     list_cmz = list(executor.map(lambda i: self.clipboard_obj.get_cmz(*i), args_summary_coefficients))
        #
        # with ThreadPoolExecutor(max_workers=10) as executor:
        #     cx_cy = np.array(list(executor.map(lambda i: self.clipboard_obj.get_cx_cy(*i), args_summary_coefficients)))
        # list_cx = cx_cy[:, 0]
        # list_cy = cx_cy[:, 1]

        # # Суммарные аэродинамические коэффициенты в декартовой системе координат
        # print('Суммарные аэродинамические коэффициенты')
        # for angle in range(0, 50, 5):
        #     angle = str(angle)
        #     plot_name = 'summary_coefficients_Cx_Cy_CMz'
        #     if not self.clipboard_obj.clipboard_dict[alpha][model_scale][angle].get(plot_name):
        #         data = {
        #             'Cx': list_cx[int(angle) // 5],
        #             'Cy': list_cy[int(angle) // 5],
        #             'CMz': list_cmz[int(angle) // 5],
        #         }
        #         fig = Plot.summary_coefficients(data)
        #         self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][plot_name] = fig
        #     fig = self.clipboard_obj.clipboard_dict[alpha][model_scale][angle].get(plot_name)
        #
        #     file_name = f'Суммарные аэродинамические коэффициенты Cx_Cy_CMz {int(angle):02}'
        #     fig.savefig(f'{path_report}\\Суммарные аэродинамические коэффициенты\\Декартовая система координат\\'
        #                 f'{file_name}')
        #
        # # Суммарные аэродинамические коэффициенты в полярной системе координат
        # print('Суммарные аэродинамические коэффициенты в полярной системе координат')
        # mods = {
        #     'MEAN': np.mean,
        #     'RMS': rms,
        #     'STD': np.std,
        #     'MAX': np.max,
        #     'MIN': np.min,
        #     'РАСЧЕТНОЕ': rach,
        #     'ОБЕСП+': obes_m,
        #     'ОБЕСП-': obes_p
        # }
        # for name, mode in mods.items():
        #     plot_name = f'summary_coefficients_Cx_Cy_CMz_{name}_polar'
        #     if not self.clipboard_obj.clipboard_dict[alpha][model_scale]['const_parameters'].get(plot_name):
        #         with ThreadPoolExecutor(max_workers=10) as executor:
        #             list_norm_cmz = list(executor.map(mode, list_cmz))
        #             list_norm_cx = list(executor.map(mode, list_cx))
        #             list_norm_cy = list(executor.map(mode, list_cy))
        #
        #         x_scale, y_scale = Plot.scaling_data(list_norm_cx, list_norm_cy)
        #         cmz_scale = Plot.scaling_data(list_norm_cmz)
        #
        #         data = {
        #             'Cx': x_scale,
        #             'Cy': y_scale,
        #             'CMz': cmz_scale,
        #         }
        #
        #         fig = Plot.polar_plot(data, name)
        #         self.clipboard_obj.clipboard_dict[alpha][model_scale]['const_parameters'][plot_name] = fig
        #
        #     fig = self.clipboard_obj.clipboard_dict[alpha][model_scale]['const_parameters'].get(plot_name)
        #     file_name = f'Суммарные аэродинамические коэффициенты Cx_Cy_CMz {name} в полярной системе координат'
        #     fig.savefig(f'{path_report}\\Суммарные аэродинамические коэффициенты\\Полярная система координат\\'
        #                 f'{file_name}')

        print('Формирование завершено')
        print(time.time() - t1)


if __name__ == '__main__':
    # c = Core()
    # # c.get_plot_isofields('6', '111', '0', 'mean', 'integral_isofields')
    # alpha = '4'
    # model_size = (
    #     '1',
    #     '1',
    #     '3',
    # )
    # angle = '0'
    # mode = 'mean'
    # type_plot = 'integral_isofields'
    # c.get_plot_isofields(alpha, model_size, angle, mode, type_plot)
    pass
