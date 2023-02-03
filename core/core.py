import os
import time

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from docx import Document
from multiprocessing import Process
from docx.shared import Inches, Pt, Mm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from concurrent.futures import ThreadPoolExecutor

import utils.utils
# local imports
from plot.plot import Plot
from utils.utils import *

# local imports
from clipboard.clipboard import Clipboard
from plot.plot import Plot


class Core:
    _count_threads = 18

    def __init__(self):
        self.clipboard_obj = Clipboard()

    def get_plot_isofields(self,
                           alpha: str,
                           model_size: Tuple[str, str, str],
                           angle: str,
                           mode: str,
                           type_plot: str):
        print(f'Запрос {type_plot} альфа = {alpha} '
              f'размер = {" ".join(model_size)} угол = {angle} режим = {mode} из буфера')
        fig = None
        model_scale, scale_factors = get_model_and_scale_factors(*model_size, alpha)

        id_fig = f'{type_plot}_{mode}_{"_".join(model_size)}'

        if not self.clipboard_obj.clipboard_dict[alpha][model_scale][angle].get(id_fig):
            print(f'Отрисовка {type_plot} альфа = {alpha} '
                  f'размер = {" ".join(model_size)} угол = {angle} режим = {mode}')
            pressure_coefficients = self.clipboard_obj.get_pressure_coefficients(alpha, model_scale, angle)
            coordinates = self.clipboard_obj.get_coordinates(alpha, model_scale)

            if type_plot == 'discrete_isofields':
                fig = Plot.discrete_isofield(model_scale, mode, angle, alpha, pressure_coefficients, coordinates)
            elif type_plot == 'integral_isofields':
                fig = Plot.integral_isofield(model_scale,
                                             model_size,
                                             scale_factors,
                                             alpha,
                                             mode,
                                             angle,
                                             pressure_coefficients,
                                             coordinates,
                                             )

            self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][id_fig] = fig

        fig = self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][id_fig]

        print(f'Запрос {type_plot} альфа = {alpha} '
              f'размер = {" ".join(model_size)} угол = {angle} режим = {mode} из буфера успешно выполнен')

        return fig

    def get_plot_summary_spectres(self,
                                  alpha: str,
                                  model_size: Tuple[str, str, str],
                                  angle: str,
                                  mode: str,
                                  scale: str,
                                  type_plot: str):
        message = f'{" ".join(list(model_size))} {alpha} {int(angle):02} {mode}'
        print(f'Запрос суммарных спектров {message} из буфера')

        mode = mode.replace(' ', '_')
        model_scale, scale_factors = get_model_and_scale_factors(*model_size, alpha)
        id_fig = f'{type_plot}_{mode}_{scale}_{"_".join(model_size)}'
        data = dict()

        if not self.clipboard_obj.clipboard_dict[alpha][model_scale][angle].get(id_fig):
            print(f'Отрисовка суммарных спектров {message}')

            if 'Cx' in mode or 'Cy' in mode:
                cx, cy = self.clipboard_obj.get_cx_cy(alpha, model_scale, angle)
                if 'Cx' in mode:
                    data['Cx'] = cx
                if 'Cy' in mode:
                    data['Cy'] = cy

            if 'CMz' in mode:
                data['CMz'] = self.clipboard_obj.get_cmz(alpha, model_scale, angle, model_size)

            fig = Plot.welch_graphs(data, model_size, alpha, angle)
            self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][id_fig] = fig
        fig = self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][id_fig]

        print(f'Запрос суммарных спектров {message} из буфера успешно выполнен')

        return fig

    def get_plot_summary_coefficients(self,
                                      alpha: str,
                                      model_size: Tuple[str, str, str],
                                      angle: str,
                                      mode: str, ):
        print(f'Запрос суммарных коэффициентов размеры = {" ".join(list(model_size))} '
              f'альфа = {alpha} угол = {int(angle):02}  режим = {mode} из буфера')

        model_scale, scale_factors = get_model_and_scale_factors(*model_size, alpha)
        type_plot = 'summary_coefficients'
        id_fig = f'{type_plot}_{mode}_{"_".join(model_size)}'
        data = dict()

        if not self.clipboard_obj.clipboard_dict[alpha][model_scale][angle].get(id_fig):
            print(f'Отрисовка суммарных коэффициентов {" ".join(list(model_size))} {alpha} {int(angle):02} {mode}')
            if 'Cx' in mode or 'Cy' in mode:
                cx, cy = self.clipboard_obj.get_cx_cy(alpha, model_scale, angle)
                if 'Cx' in mode:
                    data['Cx'] = cx
                if 'Cy' in mode:
                    data['Cy'] = cy

            if 'CMz' in mode:
                data['CMz'] = self.clipboard_obj.get_cmz(alpha, model_scale, angle, model_size)

            fig = Plot.summary_coefficients(data, model_scale, alpha, angle)
            self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][id_fig] = fig

        fig = self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][id_fig]

        print(f'Запрос суммарных коэффициентов размеры = {" ".join(list(model_size))} '
              f'альфа = {alpha} угол = {int(angle):02}  режим = {mode} из буфера успешно выполнен')

        return fig

    def get_plot_summary_coefficients_polar(self,
                                            alpha: str,
                                            model_scale: str,
                                            model_size: Tuple[str, str, str],
                                            angle_border: int,
                                            mode: str):

        print(f'Запрос суммарных коэффициентов в полярной системе координат размеры = {" ".join(list(model_size))} '
              f'альфа = {alpha} режим = {mode} из буфера')

        mods = {
            'MEAN': np.mean,
            'RMS': rms,
            'STD': np.std,
            'MAX': np.max,
            'MIN': np.min,
            'РАСЧЕТНОЕ': rach,
            'ОБЕСП+': obes_m,
            'ОБЕСП-': obes_p
        }

        plot_name = f'summary_coefficients_Cx_Cy_CMz_polar_{"_".join(model_size)}'
        if not self.clipboard_obj.clipboard_dict[alpha][model_scale]['model_attributes'].get(plot_name):
            args_cmz = [(alpha, model_scale, str(angle), model_size) for angle in range(0, angle_border, 5)]
            with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
                list_cmz = list(executor.map(lambda i: self.clipboard_obj.get_cmz(*i), args_cmz))

            args_cx_cy = [(alpha, model_scale, str(angle)) for angle in range(0, angle_border, 5)]
            with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
                cx_cy = np.array(list(executor.map(lambda i: self.clipboard_obj.get_cx_cy(*i), args_cx_cy)))

            list_cx = cx_cy[:, 0]
            list_cy = cx_cy[:, 1]

            list_norm_cmz = list(map(mods[mode], list_cmz))
            list_norm_cx = list(map(mods[mode], list_cx))
            list_norm_cy = list(map(mods[mode], list_cy))

            x_scale, y_scale = Plot.scaling_data(list_norm_cx, list_norm_cy)
            cmz_scale = Plot.scaling_data(list_norm_cmz)

            data = {
                'Cx': x_scale,
                'Cy': y_scale,
                'CMz': cmz_scale,
            }

            fig = Plot.polar_plot(data, mode, model_size, alpha)
            self.clipboard_obj.clipboard_dict[alpha][model_scale]['model_attributes'][plot_name] = fig

        fig = self.clipboard_obj.clipboard_dict[alpha][model_scale]['model_attributes'].get(plot_name)

        print(f'Запрос суммарных коэффициентов в полярной системе координат размеры = {" ".join(list(model_size))} '
              f'альфа = {alpha} режим = {mode} из буфера успешно выполнен')

        return fig

    def get_envelopes(self,
                      alpha: str,
                      model_scale: str,
                      angle: str):
        print(f'Запрос огибающих параметры = {" ".join(list(model_scale))} '
              f'альфа = {alpha} угол = {int(angle):02} из буфера')
        if not self.clipboard_obj.clipboard_dict[alpha][model_scale][angle].get('envelopes'):
            print(f'Отрисовка огибающих {" ".join(list(model_scale))} {alpha} {int(angle):02}')

            pressure_coefficients = self.clipboard_obj.get_pressure_coefficients(alpha, model_scale, angle)
            figs = Plot.envelopes(pressure_coefficients, alpha, model_scale, angle)

            for ind, val in enumerate(figs):
                self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][f'envelopes_{ind}'] = val
            self.clipboard_obj.clipboard_dict[alpha][model_scale][angle]['envelopes'] = True
            print(f'Отрисовка огибающих {" ".join(list(model_scale))} {alpha} {int(angle):02} завершена')

        figs = []
        i = 0
        while True:
            if not self.clipboard_obj.clipboard_dict[alpha][model_scale][angle].get(f'envelopes_{i}'):
                break
            else:
                figs.append(self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][f'envelopes_{i}'])
                i += 1

        print(f'Запрос огибающих параметры = {" ".join(list(model_scale))} '
              f'альфа = {alpha} угол = {int(angle):02} из буфера успешно выполнен')

        return figs

    def draw_envelopes(self,
                       alpha: str,
                       model_scale: str,
                       angle_border: int,
                       path_report: str):
        print(f'Отрисовка огибающих параметры = {" ".join(model_scale)} альфа = {alpha}')

        args_envelopes = [(alpha, model_scale, str(angle), path_report) for angle in range(0, angle_border, 5)]
        with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
            executor.map(lambda i: self.envelopes_thread(*i), args_envelopes)

        print(f'Отрисовка огибающих параметры = {" ".join(model_scale)} альфа = {alpha} завершена')

    def envelopes_thread(self,
                         alpha: str,
                         model_scale: str,
                         angle: str,
                         path_report: str):
        """Функция для запуска потока генерации графиков огибающих модели и их сохранения.
        Если не будет некоторых огибающих, отрисовка не запустится.
        Чтобы запустить, нужно очистить папку с огибающими для конкретного угла.
        """
        path_folder = f'{path_report}\\Огибающие\\Огибающие {model_scale} {alpha} {int(angle):02}'

        if not os.path.exists(path_folder):
            os.mkdir(path_folder)

            figs = self.get_envelopes(alpha, model_scale, angle)

            for i in range(len(figs)):
                file_name = f'Огибающие {model_scale} {alpha} {int(angle):02} {i * 100} - {(i + 1) * 100}'
                if not os.path.exists(f'{path_report}\\{file_name}.png'):
                    figs[i].savefig(f'{path_folder}\\{file_name}')

    def draw_welch_graphs(self,
                          alpha: str,
                          model_scale: str,
                          model_size: Tuple[str, str, str],
                          angle_border: int,
                          path_report: str):
        print(f'Отрисовка спектров суммарных коэффициентов параметры = {" ".join(model_scale)} альфа = {alpha}')

        args_welch_graphs = [(alpha, model_scale, str(angle), model_size, path_report)
                             for angle in range(0, angle_border, 5)]

        with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
            executor.map(lambda i: self.welch_graphs_thread(*i), args_welch_graphs)

        print(f'Отрисовка спектров суммарных коэффициентов '
              f'параметры = {" ".join(model_scale)} альфа = {alpha} завершена')

    def welch_graphs_thread(self,
                            alpha: str,
                            model_scale: str,
                            angle: str,
                            model_size: Tuple[str, str, str],
                            path_report: str):
        """функция для запуска потока генерации графиков спектральной плотности мощности и их сохранения."""

        path_folder = f'{path_report}\\Спектральная плотность мощности\\Логарифмическая шкала'
        plot_name = f'Спектральная плотность мощности {model_scale} {alpha} {int(angle):02}'

        if not os.path.exists(f'{path_folder}\\{plot_name}.png'):
            fig = self.get_plot_summary_spectres(alpha, model_size, angle, 'Cx Cy CMz', 'log', 'summary_spectres')
            fig.savefig(f'{path_folder}\\{plot_name}')

    def draw_isofields(self,
                       alpha: str,
                       model_size: Tuple[str, str, str],
                       angle_border: int,
                       path_report: str):
        print(f'Отрисовка изополей размеры = {" ".join(model_size)} альфа = {alpha}')

        mods = ('max',
                'mean',
                'min',
                'std',
                )

        args_isofields = [(alpha, model_size, str(angle), mode, path_report)
                          for angle in range(0, angle_border, 5) for mode in mods]

        with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
            executor.map(lambda i: self.isofields_thread(*i), args_isofields)

        print(f'Отрисовка изополей размеры = {" ".join(model_size)} альфа = {alpha} завершена')

    def isofields_thread(self,
                         alpha: str,
                         model_size: Tuple[str, str, str],
                         angle: str,
                         mode: str,
                         path_report: str):
        """Функция для запуска потока генерации изополей и их сохранения."""
        type_plot = 'integral_isofields'
        file_name = f'Изополя {" ".join(model_size)} {alpha} {int(angle):02} {mode}'

        if not os.path.exists(f'{path_report}\\Изополя ветровых нагрузок и воздействий\\Непрерывные\\'
                              f'{mode.upper()}\\{file_name}.png'):
            fig = self.get_plot_isofields(alpha, model_size, str(angle), mode, type_plot)
            fig.savefig(f'{path_report}\\Изополя ветровых нагрузок и воздействий\\Непрерывные\\'
                        f'{mode.upper()}\\{file_name}')

    def draw_model(self,
                   alpha: str,
                   model_size: Tuple[str, str, str],
                   model_scale: str,
                   path_report: str):
        print(f'Отрисовка модели размеры = {" ".join(model_size)} альфа = {alpha}')

        if not os.path.exists(f'{path_report}\\Модель\\Развертка модели.png'):
            print('Генерация развертки модели')
            coordinates = self.clipboard_obj.get_coordinates(alpha, model_scale)
            fig = Plot.model_pic(model_size, model_scale, coordinates)
            fig.savefig(f'{path_report}\\Модель\\Развертка модели')
        else:
            print('Развертка модели уже существует')

        if not os.path.exists(f'{path_report}\\Модель\\Модель в полярной системе координат.png'):
            print('Генерация модели в полярной системе координат')
            fig = Plot.model_polar(model_size)
            fig.savefig(f'{path_report}\\Модель\\Модель в полярной системе координат')
        else:
            print('Модель в полярной системе координат уже существует')

        if not os.path.exists(f'{path_report}\\Модель\\Модель 3D.png'):
            print('Генерация модели в трехмерном виде')
            fig = Plot.model_cube(model_size)
            fig.savefig(f'{path_report}\\Модель\\Модель 3D')
        else:
            print('Модель в трехмерном виде уже существует')

        print(f'Отрисовка модели размеры = {" ".join(model_size)} альфа = {alpha} завершена')

    def draw_summary_coefficients(self,
                                  alpha: str,
                                  model_size: Tuple[str, str, str],
                                  angle_border: int,
                                  path_report: str):
        print(f'Отрисовка суммарных коэффициентов размеры = {" ".join(model_size)} альфа = {alpha}')

        args_summary_coefficients = [(alpha, model_size, str(angle), path_report)
                                     for angle in range(0, angle_border, 5)]

        with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
            executor.map(lambda i: self.summary_coefficients_thread(*i), args_summary_coefficients)

        print(f'Отрисовка суммарных коэффициентов размеры = {" ".join(model_size)} альфа = {alpha} завершена')

    def summary_coefficients_thread(self,
                                    alpha: str,
                                    model_size: Tuple[str, str, str],
                                    angle: str,
                                    path_report: str):
        file_name = f'Суммарные аэродинамические коэффициенты Cx_Cy_CMz {" ".join(model_size)} {alpha} {int(angle):02}'
        if not os.path.exists(
                f'{path_report}\\Суммарные аэродинамические коэффициенты\\Декартовая система координат\\'
                f'{file_name}.png'):
            fig = self.get_plot_summary_coefficients(alpha, model_size, angle, 'Cx_Cy_CMz')

            fig.savefig(f'{path_report}\\Суммарные аэродинамические коэффициенты\\Декартовая система координат\\'
                        f'{file_name}')

    def draw_summary_coefficients_polar(self,
                                        alpha: str,
                                        model_scale: str,
                                        model_size: Tuple[str, str, str],
                                        angle_border: int,
                                        path_report: str):
        print(f'Отрисовка суммарных аэродинамических коэффициентов в полярной системе координат '
              f'размеры = {" ".join(model_size)} альфа = {alpha}')
        mods = {
            'MEAN': np.mean,
            'RMS': rms,
            'STD': np.std,
            'MAX': np.max,
            'MIN': np.min,
            'РАСЧЕТНОЕ': rach,
            'ОБЕСП+': obes_m,
            'ОБЕСП-': obes_p
        }

        mods_for_generation = []

        for mode in mods.keys():
            file_name = f'Суммарные аэродинамические коэффициенты Cx Cy CMz ' \
                        f'{" ".join(model_size)} {mode} в полярной системе координат'

            if not os.path.exists(f'{path_report}\\Cуммарные аэродинамические коэффициенты\\'
                                  f'Полярная система координат\\{file_name}.png'):
                mods_for_generation.append(mode)

        args_summary_polar = [(alpha, model_scale, model_size, angle_border, mode) for mode in mods_for_generation]

        with ThreadPoolExecutor(max_workers=len(mods_for_generation)) as executor:
            polar_plots = list(executor.map(lambda i: self.get_plot_summary_coefficients_polar(*i), args_summary_polar))

        for mode, plot in zip(mods_for_generation, polar_plots):
            file_name = f'Суммарные аэродинамические коэффициенты Cx Cy CMz ' \
                        f'{" ".join(model_size)} {mode} в полярной системе координат'

            plot.savefig(f'{path_report}\\Суммарные аэродинамические коэффициенты\\Полярная система координат\\'
                         f'{file_name}')

        print(f'Отрисовка суммарных аэродинамических коэффициентов в полярной системе координат '
              f'размеры = {" ".join(model_size)} альфа = {alpha} завершена')

    def report(self, alpha: str, model_size: Tuple[str, str, str]):
        """Создание отчёта для выбранной конфигурации.
        Отчёт включает:
            огибающие,
            суммарные аэродинамические коэффициенты,
            спектры,
            характеристика по датчикам,
        """
        print(f'Начало формирования отчета размеры = {" ".join(model_size)} альфа = {alpha}')

        report_path = 'D:\\Projects\\WindSpectrum'

        model_scale, scale_factors = get_model_and_scale_factors(*model_size, alpha)
        breadth, depth, height = model_size
        name_report = f'Отчет ширина {breadth} глубина {depth} высота {height} альфа {alpha}'
        path_report = f'{report_path}\\{name_report}'
        generate_directory_for_report(path_report)

        if model_scale[0] == model_scale[1]:
            angle_border = 50
        else:
            angle_border = 95

        # Отображение модели
        self.draw_model(alpha, model_size, model_scale, path_report)

        # Изополя
        self.draw_isofields(alpha, model_size, angle_border, path_report)

        # # Огибающие
        # self.draw_envelopes(alpha, model_scale, angle_border, path_report)
        #
        # # Суммарные аэродинамические коэффициенты в декартовой системе координат
        # self.draw_summary_coefficients(alpha, model_size, angle_border, path_report)
        #
        # # Суммарные аэродинамические коэффициенты в полярной системе координат
        # self.draw_summary_coefficients_polar(alpha, model_scale, model_size, angle_border, path_report)
        #
        # # Спектральная плотность мощности суммарных аэродинамических коэффициентов
        # self.draw_welch_graphs(alpha, model_scale, model_size, angle_border, path_report)

        print(f'Формирование отчета размеры = {" ".join(model_size)} альфа = {alpha} завершено')


if __name__ == '__main__':
    c = Core()
    # fig = c.get_plot_summary_coefficients('6', ('0.1', '0.1', '0.1'), '0', 'Cx Cy CMz')
    # utils.utils.open_fig(fig)
    t1 = time.time()
    c.report('6', ('1', '1', '1'))
    print(time.time() - t1)
    # print('=========================================')
    # c.report('6', ('2', '2', '2'))
    # print(os.listdir('D:\\'))
