import os
import time
import glob
import pickle
import logging
from typing import Tuple
from multiprocessing import Process
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from docx import Document
from docx.shared import Pt, Mm
from docx.enum.text import WD_ALIGN_PARAGRAPH

# Local imports
from plot.plot import Plot
from clipboard.clipboard import Clipboard
from utils.utils import get_model_and_scale_factors, rms, rach, obes_m, obes_p, converter_coordinates_to_real, \
    converter_coordinates, generate_directory_for_report, get_view_permutation_data, get_sequence_permutation_data, \
    changer_sequence_coefficients, get_base_angle


class Core:
    logger = logging.getLogger('Core'.ljust(15, ' '))
    logger.setLevel(logging.INFO)

    # настройка обработчика и форматировщика
    py_handler = logging.FileHandler("log.log", mode='a')
    py_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")

    # добавление форматировщика к обработчику
    py_handler.setFormatter(py_formatter)
    # добавление обработчика к логгеру
    logger.addHandler(py_handler)
    _count_threads = 20  # количество запускаемых потоков при работе

    def __init__(self, ex_clipboard = None):
        """Создание объекта для работы с буфером"""
        self.logger.info('Создание ядра')
        self.clipboard_obj = Clipboard(ex_clipboard)
        self.logger.info('Ядро успешно создано')

    def save_clipboard(self):
        output = open(f'{os.getcwd()}\\clipboard\\clipboard.pkl', 'wb')
        pickle.dump(self.clipboard_obj.clipboard_dict, output)
        output.close()

    def get_plot_isofields(self,
                           alpha: str,
                           model_size: Tuple[str, str, str],
                           angle: str,
                           mode: str,
                           type_plot: str):
        """Функция возвращает изополя.
        Если изополя отсутствуют в буфере, запускается отрисовка.
        """
        self.logger.info(f'Запрос {type_plot} альфа = {alpha} размер = {" ".join(model_size)} '
                         f'угол = {angle.rjust(2, "0")} режим = {mode.ljust(4, " ")} из буфера')

        fig = None

        model_scale, scale_factors = get_model_and_scale_factors(*model_size, alpha)

        model_scale_n = model_scale

        if model_scale[0] == model_scale[1]:
            angle_border = 45
            type_base = 'square'

        else:
            angle_border = 90
            type_base = 'rectangle'

        if model_scale[1] in ['2', '3']:
            model_scale_n = model_scale[1] + model_scale[0] + model_scale[2]
            angle = str((int(angle) - 90)) if int(angle) - 90 > 0 else str(360 + int(angle) - 90)

        if int(angle) > angle_border:
            permutation_view = get_view_permutation_data(type_base, int(angle))  # вид последовательности данных
            base_angle = get_base_angle(int(angle), permutation_view, type_base)
            sequence_permutation = get_sequence_permutation_data(type_base, permutation_view, int(angle))

            pressure_coefficients = self.clipboard_obj.get_pressure_coefficients(alpha, model_scale, str(base_angle))
            coordinates = self.clipboard_obj.get_coordinates(alpha, model_scale_n)

            pressure_coefficients = changer_sequence_coefficients(pressure_coefficients, permutation_view,
                                                                  model_scale_n, sequence_permutation)

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
        else:

            id_fig = f'{type_plot}_{mode}_{"_".join(model_size)}'

            if not self.clipboard_obj.clipboard_dict[alpha][model_scale][angle].get(id_fig):
                self.logger.info(f'Отрисовка {type_plot} альфа = {alpha} размер = {" ".join(model_size)} '
                                 f'угол = {angle.rjust(2, "0")} режим = {mode.ljust(4, " ")}')

                pressure_coefficients = self.clipboard_obj.get_pressure_coefficients(alpha, model_scale_n, angle)
                coordinates = self.clipboard_obj.get_coordinates(alpha, model_scale_n)

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

            self.logger.info(f'Запрос {type_plot} альфа = {alpha} размер = {" ".join(model_size)} '
                             f'угол = {angle.rjust(2, "0")} режим = {mode.ljust(4, " ")} из буфера успешно выполнен')

        return fig

    def get_plot_summary_spectres(self,
                                  alpha: str,
                                  model_size: Tuple[str, str, str],
                                  angle: str,
                                  mode: str,
                                  scale: str,
                                  type_plot: str):
        """Функция возвращает графики суммарных спектров.
        Если графики суммарных спектров отсутствуют в буфере, запускается отрисовка.
        """
        message = f'{" ".join(list(model_size))} {alpha} {int(angle):02} {mode}'
        self.logger.info(f'Отрисовка суммарных спектров {message}')

        mode = mode.replace(' ', '_')
        model_scale, _ = get_model_and_scale_factors(*model_size, alpha)
        id_fig = f'{type_plot}_{mode}_{scale}_{"_".join(model_size)}'
        data = dict()

        if model_scale[0] == model_scale[1]:
            angle_border = 45
            type_base = 'square'
        else:
            angle_border = 90
            type_base = 'rectangle'

        if int(angle) > angle_border:
            permutation_view = get_view_permutation_data(type_base, int(angle))  # вид последовательности данных
            base_angle = get_base_angle(int(angle), permutation_view)

            if 'Cx' in mode or 'Cy' in mode:
                cx, cy = self.clipboard_obj.get_cx_cy(alpha, model_scale, str(base_angle))
                if 'Cx' in mode:
                    data['Cx'] = cx
                if 'Cy' in mode:
                    data['Cy'] = cy

            if 'CMz' in mode:
                data['CMz'] = self.clipboard_obj.get_cmz(alpha, model_scale, str(base_angle))

            fig = Plot.welch_graphs(data, model_size, alpha, str(base_angle))

        else:
            self.logger.info(f'Запрос суммарных спектров {message} из буфера')
            if not self.clipboard_obj.clipboard_dict[alpha][model_scale][angle].get(id_fig):

                if 'Cx' in mode or 'Cy' in mode:
                    cx, cy = self.clipboard_obj.get_cx_cy(alpha, model_scale, angle)
                    if 'Cx' in mode:
                        data['Cx'] = cx
                    if 'Cy' in mode:
                        data['Cy'] = cy

                if 'CMz' in mode:
                    data['CMz'] = self.clipboard_obj.get_cmz(alpha, model_scale, angle)

                fig = Plot.welch_graphs(data, model_size, alpha, angle)
                self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][id_fig] = fig
            fig = self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][id_fig]

            self.logger.info(f'Запрос суммарных спектров {message} из буфера успешно выполнен')

        return fig

    def get_plot_summary_coefficients(self,
                                      alpha: str,
                                      model_size: Tuple[str, str, str],
                                      angle: str,
                                      mode: str, ):
        """Функция возвращает графики суммарных коэффициентов.
        Если графики суммарных коэффициентов отсутствуют в буфере, запускается отрисовка.
        """

        model_scale, _ = get_model_and_scale_factors(*model_size, alpha)
        type_plot = 'summary_coefficients'
        id_fig = f'{type_plot}_{mode}_{"_".join(model_size)}'
        data = dict()
        if model_scale[0] == model_scale[1]:
            angle_border = 45
            type_base = 'square'
        else:
            angle_border = 90
            type_base = 'rectangle'

        if int(angle) > angle_border:
            permutation_view = get_view_permutation_data(type_base, int(angle))  # вид последовательности данных
            base_angle = get_base_angle(int(angle), permutation_view)

            if 'Cx' in mode or 'Cy' in mode:
                cx, cy = self.clipboard_obj.get_cx_cy(alpha, model_scale, str(base_angle))
                if 'Cx' in mode:
                    data['Cx'] = cx
                if 'Cy' in mode:
                    data['Cy'] = cy

            if 'CMz' in mode:
                data['CMz'] = self.clipboard_obj.get_cmz(alpha, model_scale, str(base_angle))

            fig = Plot.summary_coefficients(data, model_scale, alpha, str(base_angle))

        else:
            self.logger.info(f'Запрос суммарных коэффициентов размеры = {" ".join(list(model_size))} '
                             f'альфа = {alpha} угол = {angle.rjust(2, "0")}  режим = {mode.ljust(4, " ")} из буфера')
            if not self.clipboard_obj.clipboard_dict[alpha][model_scale][angle].get(id_fig):
                self.logger.info(f'Отрисовка суммарных коэффициентов '
                                 f'{" ".join(list(model_size))} {alpha} {int(angle):02} {mode.ljust(4, " ")}')
                if 'Cx' in mode or 'Cy' in mode:
                    cx, cy = self.clipboard_obj.get_cx_cy(alpha, model_scale, angle)
                    if 'Cx' in mode:
                        data['Cx'] = cx
                    if 'Cy' in mode:
                        data['Cy'] = cy

                if 'CMz' in mode:
                    data['CMz'] = self.clipboard_obj.get_cmz(alpha, model_scale, angle)

                fig = Plot.summary_coefficients(data, model_scale, alpha, angle)
                self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][id_fig] = fig

            fig = self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][id_fig]

            self.logger.info(f'Запрос суммарных коэффициентов размеры = {" ".join(list(model_size))} '
                             f'альфа = {alpha} угол = {int(angle):02}  '
                             f'режим = {mode.ljust(4, " ")} из буфера успешно выполнен')

        return fig

    def get_plot_summary_coefficients_polar(self,
                                            alpha: str,
                                            model_scale: str,
                                            model_size: Tuple[str, str, str],
                                            angle_border: int,
                                            mode: str):
        """Функция возвращает графики суммарных коэффициентов в полярной системе координат.
        Если графики суммарных коэффициентов в полярной системе координат отсутствуют в буфере, запускается отрисовка.
        """
        self.logger.info(f'Запрос суммарных коэффициентов в '
                         f'полярной системе координат размеры = {" ".join(list(model_size))} '
                         f'альфа = {alpha} режим = {mode.ljust(4, " ")} из буфера')

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

        plot_name = f'summary_coefficients_Cx_Cy_CMz_polar_{"_".join(model_size)}_{mode}'
        if not self.clipboard_obj.clipboard_dict[alpha][model_scale]['model_attributes'].get(plot_name):
            args_cmz = [(alpha, model_scale, str(angle)) for angle in range(0, angle_border, 5)]
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

        self.logger.info(f'Запрос суммарных коэффициентов в полярной системе координат размеры = '
                         f'{" ".join(list(model_size))} альфа = {alpha} режим = {mode.ljust(4, " ")} из буфера успешно выполнен')

        return fig

    def get_envelopes(self,
                      alpha: str,
                      model_size: Tuple[str, str, str],
                      angle: str,
                      model_scale: str = ''):
        """Функция возвращает графики огибающих.
        Если огибающие отсутствуют в буфере, запускается отрисовка.
        """
        if model_scale == '':
            model_scale, _ = get_model_and_scale_factors(*model_size, alpha)
        self.logger.info(f'Отрисовка огибающих {" ".join(list(model_scale))} {alpha} {int(angle):02}')

        if model_scale[0] == model_scale[1]:
            angle_border = 45
            type_base = 'square'
        else:
            angle_border = 90
            type_base = 'rectangle'

        if int(angle) > angle_border:
            permutation_view = get_view_permutation_data(type_base, int(angle))  # вид последовательности данных
            base_angle = get_base_angle(int(angle), permutation_view)

            pressure_coefficients = self.clipboard_obj.get_pressure_coefficients(alpha, model_scale, str(base_angle))
            figs = Plot.envelopes(pressure_coefficients, alpha, model_scale, angle)

        else:
            self.logger.info(f'Запрос огибающих параметры = {" ".join(list(model_scale))} '
                             f'альфа = {alpha} угол = {angle.rjust(2, "0")} из буфера')

            if not self.clipboard_obj.clipboard_dict[alpha][model_scale][angle].get('envelopes'):

                pressure_coefficients = self.clipboard_obj.get_pressure_coefficients(alpha, model_scale, angle)
                figs = Plot.envelopes(pressure_coefficients, alpha, model_scale, angle)

                for ind, val in enumerate(figs):
                    self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][f'envelopes_{ind}'] = val
                self.clipboard_obj.clipboard_dict[alpha][model_scale][angle]['envelopes'] = True
                self.logger.info(f'Отрисовка огибающих {" ".join(list(model_scale))} {alpha} {int(angle):02} завершена')

            figs = []
            i = 0
            while True:
                if not self.clipboard_obj.clipboard_dict[alpha][model_scale][angle].get(f'envelopes_{i}'):
                    break
                else:
                    figs.append(self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][f'envelopes_{i}'])
                    i += 1

            self.logger.info(f'Запрос огибающих параметры = {" ".join(list(model_scale))} '
                             f'альфа = {alpha} угол = {angle.rjust(2, "0")} из буфера успешно выполнен')

        return figs

    def draw_envelopes(self,
                       alpha: str,
                       model_scale: str,
                       angle_border: int,
                       path_report: str):
        """Функция запускает отрисовку всех огибающих выбранной модели.
        Отрисовка происходит в многопоточном режиме.
        """
        self.logger.info(f'Отрисовка огибающих параметры = {" ".join(model_scale)} альфа = {alpha}')

        args_envelopes = [(alpha, model_scale, str(angle), path_report) for angle in range(0, angle_border, 5)]

        with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
            executor.map(lambda i: self.envelopes_thread(*i), args_envelopes)

        self.logger.info(f'Отрисовка огибающих параметры = {" ".join(model_scale)} альфа = {alpha} завершена')

    def envelopes_thread(self,
                         alpha: str,
                         model_scale: str,
                         angle: str,
                         path_report: str):
        """Функция для запуска потока генерации графиков огибающих модели и их сохранения."""
        path_folder = f'{path_report}\\Огибающие\\Огибающие {model_scale} {alpha} {int(angle):02}'

        if not os.path.exists(path_folder):
            os.mkdir(path_folder)

        figs = self.get_envelopes(alpha, ('1', '1', '1'), angle, model_scale)

        for i in range(len(figs)):
            file_name = f'Огибающие {model_scale} {alpha} {int(angle):02} {i * 100} - {(i + 1) * 100}.png'
            figs[i].savefig(f'{path_folder}\\{file_name}')

    def draw_welch_graphs(self,
                          alpha: str,
                          model_scale: str,
                          model_size: Tuple[str, str, str],
                          angle_border: int,
                          path_report: str):
        """Функция запускает отрисовку всех граффиков спектральной плотности мощности для выбранной модели.
        Отрисовка происходит в многопоточном режиме.
        """
        self.logger.info(f'Отрисовка спектров суммарных коэффициентов '
                         f'параметры = {" ".join(model_scale)} альфа = {alpha}')

        args_welch_graphs = [(alpha, model_scale, str(angle), model_size, path_report)
                             for angle in range(0, angle_border, 5)]

        with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
            executor.map(lambda i: self.welch_graphs_thread(*i), args_welch_graphs)

        self.logger.info(f'Отрисовка спектров суммарных коэффициентов '
                         f'параметры = {" ".join(model_scale)} альфа = {alpha} завершена')

    def welch_graphs_thread(self,
                            alpha: str,
                            model_scale: str,
                            angle: str,
                            model_size: Tuple[str, str, str],
                            path_report: str):
        """Функция для запуска потока генерации графиков спектральной плотности мощности и их сохранения."""

        path_folder = f'{path_report}\\Спектральная плотность мощности\\Логарифмическая шкала'
        file_name = f'Спектральная плотность мощности {model_scale} {alpha} {int(angle):02}.png'

        fig = self.get_plot_summary_spectres(alpha, model_size, angle, 'Cx Cy CMz', 'log', 'summary_spectres')
        fig.savefig(f'{path_folder}\\{file_name}')

    def draw_isofields(self,
                       alpha: str,
                       model_size: Tuple[str, str, str],
                       angle_border: int,
                       path_report: str):
        """Функция запускает отрисовку всех изополей выбранной модели.
        Отрисовка происходит в многопоточном режиме.
        """
        self.logger.info(f'Отрисовка изополей размеры = {" ".join(model_size)} альфа = {alpha}')

        mods = ('max',
                'mean',
                'min',
                'std',
                )

        args_isofields = [(alpha, model_size, str(angle), mode, path_report)
                          for angle in range(0, angle_border, 5) for mode in mods]

        with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
            executor.map(lambda i: self.isofields_thread(*i), args_isofields)

        self.logger.info(f'Отрисовка изополей размеры = {" ".join(model_size)} альфа = {alpha} завершена')

    def isofields_thread(self,
                         alpha: str,
                         model_size: Tuple[str, str, str],
                         angle: str,
                         mode: str,
                         path_report: str):
        """Функция для запуска потока генерации изополей и их сохранения."""

        file_name = f'Изополя {" ".join(model_size)} {alpha} {int(angle):02} {mode}.png'

        fig = self.get_plot_isofields(alpha, model_size, str(angle), mode, 'integral_isofields')

        fig.savefig(
            f'{path_report}\\Изополя ветровых нагрузок и воздействий\\Непрерывные\\{mode.upper()}\\{file_name}')

        fig = self.get_plot_isofields(alpha, model_size, str(angle), mode, 'discrete_isofields')

        fig.savefig(
            f'{path_report}\\Изополя ветровых нагрузок и воздействий\\Дискретные\\{mode.upper()}\\{file_name}')

    def draw_model(self,
                   alpha: str,
                   model_size: Tuple[str, str, str],
                   model_scale: str,
                   path_report: str, ):
        """Функция запускает отрисовку  выбранной модели.
        - развертка модели
        - модель в полярной системе координат
        - модель в трехмерном виде
        """
        self.logger.info(f'Отрисовка модели размеры = {" ".join(model_size)} альфа = {alpha}')

        self.logger.info('Генерация развертки модели')
        coordinates = self.clipboard_obj.get_coordinates(alpha, model_scale)
        coordinates = converter_coordinates_to_real(*coordinates, model_size, model_scale)
        fig = Plot.model_pic(model_size, model_scale, coordinates)
        fig.savefig(f'{path_report}\\Модель\\Развертка модели.png')
        self.logger.info('Генерация развертки модели завершена')

        self.logger.info('Генерация модели в полярной системе координат')
        fig = Plot.model_polar(model_size)
        fig.savefig(f'{path_report}\\Модель\\Модель в полярной системе координат.png')
        self.logger.info('Генерация модели в полярной системе координат завершена')

        self.logger.info('Генерация модели в трехмерном виде')
        fig = Plot.model_cube(model_size)
        fig.savefig(f'{path_report}\\Модель\\Модель 3D')
        self.logger.info('Генерация модели в трехмерном виде завершена')

        self.logger.info(f'Отрисовка модели размеры = {" ".join(model_size)} альфа = {alpha} завершена')

    def draw_summary_coefficients(self,
                                  alpha: str,
                                  model_size: Tuple[str, str, str],
                                  angle_border: int,
                                  path_report: str):
        """Функция запускает отрисовку всех графиков суммарных коэффициентов для выбранной модели.
        Отрисовка происходит в многопоточном режиме.
        """
        self.logger.info(f'Отрисовка суммарных коэффициентов размеры = {" ".join(model_size)} альфа = {alpha}')

        args_summary_coefficients = [(alpha, model_size, str(angle), path_report)
                                     for angle in range(0, angle_border, 5)]

        with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
            executor.map(lambda i: self.summary_coefficients_thread(*i), args_summary_coefficients)

        self.logger.info(f'Отрисовка суммарных коэффициентов '
                         f'размеры = {" ".join(model_size)} альфа = {alpha} завершена')

    def summary_coefficients_thread(self,
                                    alpha: str,
                                    model_size: Tuple[str, str, str],
                                    angle: str,
                                    path_report: str):
        """Функция для запуска потока генерации графиков суммарных аэродинамических коэффициентов модели
         и их сохранения."""

        file_name = f'Суммарные аэродинамические коэффициенты Cx_Cy_CMz ' \
                    f'{" ".join(model_size)} {alpha} {int(angle):02}.png'

        fig = self.get_plot_summary_coefficients(alpha, model_size, angle, 'Cx_Cy_CMz')

        fig.savefig(f'{path_report}\\Суммарные аэродинамические коэффициенты\\'
                    f'Декартовая система координат\\{file_name}')

    def draw_summary_coefficients_polar(self,
                                        alpha: str,
                                        model_scale: str,
                                        model_size: Tuple[str, str, str],
                                        angle_border: int,
                                        path_report: str):
        """Функция запускает отрисовку всех графиков суммарных коэффициентов в полярной системе для выбранной модели.
        Отрисовка происходит в многопоточном режиме.
        """
        self.logger.info(f'Отрисовка суммарных аэродинамических коэффициентов в полярной системе координат '
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

        args_summary_polar = [(alpha, model_scale, model_size, angle_border, mode) for mode in mods.keys()]

        with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
            polar_plots = list(executor.map(lambda i: self.get_plot_summary_coefficients_polar(*i), args_summary_polar))

        for mode, plot in zip(mods.keys(), polar_plots):
            file_name = f'Суммарные аэродинамические коэффициенты Cx Cy CMz ' \
                        f'{" ".join(model_size)} {mode} в полярной системе координат.png'

            plot.savefig(f'{path_report}\\Суммарные аэродинамические коэффициенты\\Полярная система координат\\'
                         f'{file_name}')

        self.logger.info(f'Отрисовка суммарных аэродинамических коэффициентов в полярной системе координат '
                         f'размеры = {" ".join(model_size)} альфа = {alpha} завершена')

    def get_sensor_statistics(self,
                              alpha: str,
                              model_scale: str,
                              angle_border: int,
                              model_size: Tuple[str, str, str],
                              coordinates
                              ):
        sensor_statistics = []
        accuracy_values = 2

        breadth_real, depth_real, height_real = float(model_size[0]), float(model_size[1]), float(model_size[2])

        face_number = self.clipboard_obj.get_face_number(alpha, model_scale)

        x, z = coordinates

        sensors_on_model = len(x)
        x_new, y_new = converter_coordinates(x, breadth_real, depth_real, face_number, sensors_on_model)

        for angle in range(0, angle_border, 5):
            statistics_of_angle = []

            pressure_coefficients = self.clipboard_obj.get_pressure_coefficients(alpha, model_scale, str(angle))
            pressure_coefficients_t = pressure_coefficients.T

            mean_list = np.mean(pressure_coefficients, axis=0).round(accuracy_values)
            std_list = np.std(pressure_coefficients, axis=0).round(accuracy_values)
            max_list = np.max(pressure_coefficients, axis=0).round(accuracy_values)
            min_list = np.min(pressure_coefficients, axis=0).round(accuracy_values)

            rms_m_list = [rms(i).round(accuracy_values) for i in pressure_coefficients_t]
            rach_list = [rach(i).round(accuracy_values) for i in pressure_coefficients_t]
            obec_p_list = [obes_p(i).round(accuracy_values) for i in pressure_coefficients_t]
            obec_m_list = [obes_m(i).round(accuracy_values) for i in pressure_coefficients_t]

            for i in range(sensors_on_model):
                row = [i + 1,
                       x_new[i],
                       y_new[i],
                       z[i].round(3),
                       mean_list[i],
                       rms_m_list[i],
                       std_list[i],
                       max_list[i],
                       min_list[i],
                       rach_list[i],
                       obec_p_list[i],
                       obec_m_list[i],
                       ]
                statistics_of_angle.append(row)
            sensor_statistics.append(statistics_of_angle)

        return sensor_statistics

    def get_summary_coefficients_statistics(self,
                                            angle_border: int,
                                            alpha: str,
                                            model_name: str,
                                            ):
        accuracy_values = 3
        statistics = []

        for angle in range(0, angle_border, 5):
            list_cmz_cx_cy = []
            cmz = self.clipboard_obj.get_cmz(alpha, model_name, str(angle))
            cx, cy = self.clipboard_obj.get_cx_cy(alpha, model_name, str(angle))
            for data, name in zip((cx, cy, cmz), ('Cx', 'Cy', 'CMz')):
                list_cmz_cx_cy.append([
                    name,
                    np.mean(data).round(accuracy_values),
                    rms(data).round(accuracy_values),
                    np.std(data).round(accuracy_values),
                    np.max(data).round(accuracy_values),
                    np.min(data).round(accuracy_values),
                    rach(data).round(accuracy_values),
                    obes_p(data).round(accuracy_values),
                    obes_m(data).round(accuracy_values)
                ])

            statistics.append(list_cmz_cx_cy)

        return statistics

    @staticmethod
    def run_proc_report(clipboard_dict, alpha, model_size):
        new_core = Core(clipboard_dict)
        new_core.report(alpha, model_size)

    @staticmethod
    def check_alive_proc(proc, button):
        while proc.is_alive():
            time.sleep(3.993)

        button.disabled = False

    def report_process(self, alpha: str, model_size: Tuple[str, str, str], button):
        proc = Process(target=Core.run_proc_report, args=(self.clipboard_obj.clipboard_dict, alpha, model_size))
        proc.start()

        executor = ThreadPoolExecutor(max_workers=1)
        executor.map(lambda i: Core.check_alive_proc(*i), ((proc, button),))

    def report(self, alpha: str, model_size: Tuple[str, str, str]):
        """Создание отчёта для выбранной конфигурации.
        Отчёт включает:
            - отображение модели
            - изополя
            - огибающие
            - суммарные аэродинамические коэффициенты
            - суммарные аэродинамические коэффициенты в полярной системе координат
            - спектральная плотность мощности суммарных аэродинамических коэффициентов
            - характеристика по датчикам
        """
        self.logger.info(f'Начало формирования отчета размеры = {" ".join(model_size)} альфа = {alpha}')

        folder = os.getcwd()

        model_scale, scale_factors = get_model_and_scale_factors(*model_size, alpha)
        breadth, depth, height = model_size
        name_report = f'Отчет ширина {breadth} глубина {depth} высота {height} альфа {alpha}'
        path_report = f'{folder}\\{name_report}'
        generate_directory_for_report(path_report)

        if model_scale[0] == model_scale[1]:
            angle_border = 50
        else:
            angle_border = 95

        x, z = self.clipboard_obj.get_coordinates(alpha, model_scale)

        # Отображение модели
        self.draw_model(alpha, model_size, model_scale, path_report)

        # Изополя
        self.draw_isofields(alpha, model_size, angle_border, path_report)

        # Огибающие
        self.draw_envelopes(alpha, model_scale, angle_border, path_report)

        # Суммарные аэродинамические коэффициенты в декартовой системе координат
        self.draw_summary_coefficients(alpha, model_size, angle_border, path_report)

        # Суммарные аэродинамические коэффициенты в полярной системе координат
        self.draw_summary_coefficients_polar(alpha, model_scale, model_size, angle_border, path_report)

        # Спектральная плотность мощности суммарных аэродинамических коэффициентов
        self.draw_welch_graphs(alpha, model_scale, model_size, angle_border, path_report)

        # Работа с word файлом
        doc = Document()
        style = doc.styles['Normal']
        style.font.size = Pt(14)
        style.font.name = 'Times New Roman'
        section = doc.sections[0]
        section.left_margin = Mm(30)
        section.right_margin = Mm(15)
        section.top_margin = Mm(20)
        section.bottom_margin = Mm(20)

        counter_plots = 1  # Счетчик графиков для нумерации
        counter_tables = 1  # Счетчик таблиц для нумерации

        title = doc.add_heading().add_run(f'Отчет по зданию {breadth}x{depth}x{height}')
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        title.font.size = Pt(24)
        title.bold = True

        for i in ('Параметры ветрового воздействия:',
                  'Ветровой район: None',
                  'Тип местности: None'
                  ):
            doc.add_paragraph().add_run(i)
        p = doc.add_paragraph()

        run = p.add_run()
        run.add_picture(f'{path_report}\\Модель\\Модель 3D.png', width=Mm(82.5))
        run.add_picture(f'{path_report}\\Модель\\Модель в полярной системе координат.png', width=Mm(82.5))
        doc.add_paragraph().add_run(
            f'Рисунок {counter_plots}. Геометрические размеры и система координат направления ветровых потоков')
        counter_plots += 1

        # 1. Геометрические размеры здания

        doc.add_heading().add_run('1. Геометрические размеры здания').font.size = Pt(20)
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

        header_model = (
            'Геометрический размер',
            'Значение, м'
        )
        table_model = doc.add_table(rows=1, cols=len(header_model))
        table_model.style = 'Table Grid'
        hdr_cells = table_model.rows[0].cells
        for i in range(len(header_model)):
            hdr_cells[i].add_paragraph().add_run(header_model[i])
        for i, j in zip((breadth, depth, height), ('Ширина:', 'Глубина:', 'Высота:')):
            row_cells = table_model.add_row().cells
            row_cells[0].add_paragraph().add_run(j)
            row_cells[1].add_paragraph().add_run(str(i))
        doc.add_picture(f'{path_report}\\Модель\\Развертка модели.png', width=Mm(165))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph().add_run(f'Рисунок {counter_plots}. Система датчиков мониторинга')
        counter_plots += 1
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_page_break()

        # 2. Статистика по датчиках. Максимумы и огибающие

        doc.add_heading().add_run('2. Статистика по датчиках. Максимумы и огибающие').font.size = Pt(20)
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

        for angle in range(0, angle_border, 5):
            envelopes = glob.glob(
                f'{path_report}\\Огибающие\\Огибающие {model_scale} {alpha} {angle:02}\\Огибающие *.png')
            for i in envelopes:
                doc.add_picture(i, height=Mm(80))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                doc.add_paragraph().add_run(
                    f'Рисунок {counter_plots}. Огибающие ветрового давления для здания '
                    f'{breadth}x{depth}x{height} угол {angle:02}º '
                    f'датчики {i[i[:i.rfind("-") - 1].rfind(" ") + 1:i.rfind(".")]}')
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                counter_plots += 1
        doc.add_page_break()

        # 3. Изополя ветровых нагрузок и воздействий

        doc.add_heading().add_run('3. Изополя ветровых нагрузок и воздействий').font.size = Pt(20)
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

        # 3.1 Непрерывные изополя

        doc.add_heading(level=2).add_run('3.1 Непрерывные изополя').font.size = Pt(16)
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

        mods = ('MAX', 'MEAN', 'MIN', 'STD')
        for mode in mods:
            for angle in range(0, angle_border, 5):
                isofields = f'{path_report}\\Изополя ветровых нагрузок и воздействий\\Непрерывные\\{mode}\\' \
                            f'Изополя {breadth} {depth} {height} {alpha} {angle:02} {mode}.png'

                doc.add_picture(isofields)
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                doc.add_paragraph().add_run(
                    f'Рисунок {counter_plots}. Непрерывные изополя {mode} '
                    f'для здания {breadth}x{depth}x{height} угол {angle:02}º')
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                counter_plots += 1

        # 3.2 Дискретные изополя

        doc.add_heading(level=2).add_run('3.2 Дискретные изополя').font.size = Pt(16)
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

        for mode in mods:
            for angle in range(0, angle_border, 5):
                isofields = f'{path_report}\\Изополя ветровых нагрузок и воздействий\\Дискретные\\{mode}\\' \
                            f'Изополя {breadth} {depth} {height} {alpha} {angle:02} {mode}.png'

                doc.add_picture(isofields)
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                doc.add_paragraph().add_run(
                    f'Рисунок {counter_plots}. Непрерывные изополя {mode} '
                    f'для здания {breadth}x{depth}x{height} угол {angle:02}º')
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                counter_plots += 1
        doc.add_page_break()

        # 4. Статистика по датчикам в табличном виде

        coordinates = converter_coordinates_to_real(x, z, model_size, model_scale)
        sensor_statistics = self.get_sensor_statistics(alpha,
                                                       model_scale,
                                                       angle_border,
                                                       model_size,
                                                       coordinates,
                                                       )

        doc.add_heading().add_run('4. Статистика по датчикам в табличном виде ').font.size = Pt(20)
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        header_sensors = (
            'ДАТЧИК',
            'X(мм)',
            'Y(мм)',
            'Z(мм)',
            'MEAN',
            'RMS',
            'STD',
            'MAX',
            'MIN',
            'РАСЧЕТНОЕ',
            'ОБЕСП+',
            'ОБЕСП-'
        )
        for angle in range(0, angle_border, 5):
            doc.add_paragraph().add_run(
                f'\nТаблица {counter_tables}. Аэродинамический коэффициент в датчиках для '
                f'здания {breadth}x{depth}x{height} угол {angle:02}º')
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            counter_tables += 1
            table_sensors = doc.add_table(rows=1, cols=len(header_sensors))
            table_sensors.style = 'Table Grid'
            hdr_cells = table_sensors.rows[0].cells
            for i in range(len(header_sensors)):
                hdr_cells[i].add_paragraph().add_run(header_sensors[i]).font.size = Pt(8)

            for rec in sensor_statistics[angle // 5]:
                row_cells = table_sensors.add_row().cells
                for i in range(len(rec)):
                    row_cells[i].add_paragraph().add_run(str(rec[i])).font.size = Pt(12)
        doc.add_page_break()

        del sensor_statistics

        # 5. Суммарные значения аэродинамических коэффициентов

        summary_coefficients_statistics = self.get_summary_coefficients_statistics(angle_border,
                                                                                   alpha,
                                                                                   model_scale,
                                                                                   )

        doc.add_heading().add_run('5. Суммарные значения аэродинамических коэффициентов').font.size = Pt(20)
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        header_sum = (
            'СИЛА',
            'MEAN',
            'RMS',
            'STD',
            'MAX',
            'MIN',
            'РАСЧЕТНОЕ',
            'ОБЕСП+',
            'ОБЕСП-'
        )
        for angle in range(0, angle_border, 5):
            doc.add_picture(
                f'{path_report}\\Суммарные аэродинамические коэффициенты\\Декартовая система координат\\'
                f'Суммарные аэродинамические коэффициенты Cx_Cy_CMz {" ".join(model_size)} {alpha} {angle:02}.png')
            doc.add_paragraph().add_run(
                f'Рисунок {counter_plots}. Суммарные аэродинамические коэффициенты '
                f'для здания {breadth}x{depth}x{height} угол {angle:02}º')
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            counter_plots += 1
            doc.add_paragraph().add_run(
                f'Таблица {counter_tables}. Суммарные аэродинамические коэффициенты '
                f'для здания {breadth}x{depth}x{height} угол {angle:02}º')
            counter_tables += 1
            table_sum = doc.add_table(rows=1, cols=len(header_sum))
            table_sum.style = 'Table Grid'
            hdr_cells = table_sum.rows[0].cells
            for i in range(len(header_sum)):
                hdr_cells[i].add_paragraph().add_run(header_sum[i]).font.size = Pt(8)

            for rec in summary_coefficients_statistics[angle // 5]:
                row_cells = table_sum.add_row().cells
                for i in range(len(rec)):
                    row_cells[i].add_paragraph().add_run(str(rec[i])).font.size = Pt(12)

            doc.add_page_break()

        for mode in header_sum[1:]:
            doc.add_picture(f'{path_report}\\Суммарные аэродинамические коэффициенты\\Полярная система координат\\'
                            f'Суммарные аэродинамические коэффициенты Cx Cy CMz {breadth} {depth} {height} {mode} '
                            f'в полярной системе координат.png')

            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            doc.add_paragraph().add_run(
                f'Рисунок {counter_plots}. Суммарные аэродинамические коэффициенты в полярной системе координат {mode}'
                f' для здания {breadth}x{depth}x{height}')
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            counter_plots += 1
            doc.add_page_break()

        del summary_coefficients_statistics

        # 6. Спектры cуммарных значений аэродинамических коэффициентов

        doc.add_heading().add_run('6. Спектры cуммарных значений аэродинамических коэффициентов').font.size = Pt(20)
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        for angle in range(0, angle_border, 5):
            doc.add_picture(f'{path_report}\\Спектральная плотность мощности\\Логарифмическая шкала\\'
                            f'Спектральная плотность мощности {model_scale} {alpha} {angle:02}.png')
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            doc.add_paragraph().add_run(
                f'Рисунок {counter_plots}. Спектр cуммарных значений аэродинамических коэффициентов '
                f'для здания {breadth}x{depth}x{height} угол {angle:02} º')
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            counter_plots += 1
        doc.add_page_break()

        doc.save(f'{path_report}\\Отчет ширина {breadth} глубина {depth} высота {height} альфа {alpha}.docx')
        # os.startfile(f'{path_report}\\Отчет ширина {breadth} глубина {depth} высота {height} альфа {alpha}.docx')
        self.logger.info(f'Формирование отчета размеры = {" ".join(model_size)} альфа = {alpha} завершено')


if __name__ == '__main__':
    c = Core()
    c.report('4', ('10', '10', '12'))
    # run_proc(c.clipboard_obj.clipboard_dict, '4', ('1', '1', '1'))
    # fig = c.get_plot_summary_coefficients('6', ('0.1', '0.1', '0.1'), '0', 'Cx Cy CMz')
    # utils.utils.open_fig(fig)
    # t1 = time.time()
    # c.report_process('6', ('1', '4', '8'))
    # c.get_plot_isofields('6', ('1', '1', '1'),'0','mean','integral_isofields')
    # c.save_clipboard()
    # c.report_process('6', ('1', '1', '1'))
    # print(time.time() - t1)
    # print('=========================================')
    # c.report('6', ('2', '2', '2')) alpha: str,
    #                            model_size: Tuple[str, str, str],
    #                            angle: str,
    #                            mode: str,
    #                            type_plot: str):
    # print(os.listdir('D:\\'))
