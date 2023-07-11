import os
import time
import glob
import pickle
from typing import Tuple, List
from multiprocessing import Process, Manager, managers
from concurrent.futures import ThreadPoolExecutor

import toml
import numpy as np
from docx import Document
from docx.shared import Pt, Mm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Local imports
from plot.plot import Plot
from clipboard.clipboard import Clipboard
from utils.utils import (get_logger,
                         get_model_and_scale_factors,
                         rms,
                         rach,
                         obes_m,
                         obes_p,
                         converter_coordinates_to_real,
                         converter_coordinates,
                         generate_directory_for_report,
                         get_view_permutation_data,
                         get_sequence_permutation_data,
                         changer_sequence_coefficients,
                         get_base_angle,
                         to_dict,
                         to_multiprocessing_dict,
                         id_to_name,
                         )


class Core:
    config = toml.load('config.toml')

    _count_threads = config['core']['count_threads']  # количество запускаемых потоков при создании отчета

    def __init__(self, ex_clipboard = None):
        """Создание объекта для работы с буфером"""
        self.logger = get_logger('Core')
        self.logger.info('Создание ядра')
        self._manager = None
        self.clipboard_obj = Clipboard(ex_clipboard=ex_clipboard)
        self.logger.info('Ядро успешно создано')

    def save_clipboard(self):
        with open(f'{os.getcwd()}\\clipboard\\clipboard_binary', "wb") as clipboard_binary:
            dict_without_manager = to_dict(self.clipboard_obj.clipboard_dict)
            data_binary = pickle.dumps(dict_without_manager)
            clipboard_binary.write(data_binary)

    def get_plot_isofields(self,
                           alpha: str,
                           model_size: Tuple[str, str, str],
                           angle: str,
                           mode: str,
                           pressure_plot_parameters = None):
        """Функция возвращает изополя"""
        if pressure_plot_parameters:
            type_plot = 'isofields_pressure'
        else:
            type_plot = 'isofields_coefficients'

        fig = self.clipboard_obj.get_isofields(alpha, model_size, angle, mode, type_plot, pressure_plot_parameters)

        return fig

    def get_plot_summary_spectres(self,
                                  alpha: str,
                                  model_size: Tuple[str, str, str],
                                  angle: str,
                                  mode: str,
                                  scale: str = None,
                                  type_plot: str = None):

        fig = self.clipboard_obj.get_summary_spectres(alpha, model_size, angle, mode, scale, type_plot)

        return fig

    def get_plot_summary_coefficients(self,
                                      alpha: str,
                                      model_size: Tuple[str, str, str],
                                      angle: str,
                                      mode: str, ):

        fig = self.clipboard_obj.get_summary_coefficients(alpha, model_size, angle, mode)

        return fig

    def draw_welch_graphs(self,
                          alpha: str,
                          model_scale: str,
                          model_size: Tuple[str, str, str],
                          angle_border: int,
                          path_report: str,
                          mods: List[str]):
        """Функция запускает отрисовку всех граффиков спектральной плотности мощности для выбранной модели.
        Отрисовка происходит в многопоточном режиме.
        """
        self.logger.info(f'Отрисовка спектров суммарных коэффициентов '
                         f'параметры = {" ".join(model_scale)} альфа = {alpha}')
        for mode in mods:
            args_welch_graphs = [(alpha, model_scale, str(angle), model_size, path_report, mode)
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
                            path_report: str,
                            mode: str):
        """Функция для запуска потока генерации графиков спектральной плотности мощности и их сохранения."""
        mode_fig = ' '.join(mode.lower().title().split('_'))
        path_folder = f'{path_report}\\Спектральная плотность мощности\\Логарифмическая шкала'
        file_name = f'Спектральная плотность мощности {model_scale} {alpha} {mode_fig} {int(angle):02}.png'

        fig = self.get_plot_summary_spectres(alpha, model_size, angle, mode, 'log', 'summary_spectres')
        fig.savefig(f'{path_folder}\\{file_name}')
        plt.close(fig)

    def get_plot_pressure_tap_locations(self, model_size, alpha):
        return self.clipboard_obj.get_plot_pressure_tap_locations(model_size, alpha)

    def get_plot_model_3d(self, model_size):
        return self.clipboard_obj.get_plot_model_3d(model_size)

    def get_model_polar(self, model_size):
        return self.clipboard_obj.get_model_polar(model_size)

    def get_pseudocolor_coefficients(self, alpha, model_size, angle, mode):
        return self.clipboard_obj.get_pseudocolor_coefficients(alpha, model_size, angle, mode)

    def draw_summary_coefficients(self,
                                  alpha: str,
                                  model_size: Tuple[str, str, str],
                                  angle_border: int,
                                  path_report: str,
                                  mods: List[str], ):

        """Функция запускает отрисовку всех графиков суммарных коэффициентов для выбранной модели.
        Отрисовка происходит в многопоточном режиме.
        """

        for mode in mods:
            args_summary_coefficients = [(alpha, model_size, str(angle), path_report, mode)
                                         for angle in range(0, angle_border, 5)]

            with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
                executor.map(lambda i: self.summary_coefficients_thread(*i), args_summary_coefficients)

    def summary_coefficients_thread(self,
                                    alpha: str,
                                    model_size: Tuple[str, str, str],
                                    angle: str,
                                    path_report: str,
                                    mode: str):
        """Функция для запуска потока генерации графиков суммарных аэродинамических коэффициентов модели
         и их сохранения."""
        mode_fig = ' '.join(mode.lower().title().split('_'))
        file_name = f'Суммарные аэродинамические коэффициенты {mode_fig} ' \
                    f'{" ".join(model_size)} {alpha} {int(angle):02}.png'

        path_sum = f'{path_report}\\Суммарные аэродинамические коэффициенты\\Декартовая система координат\\' \
                   f'{int(angle):02}'
        if not os.path.isdir(path_sum):
            os.mkdir(path_sum)

        fig = self.get_plot_summary_coefficients(alpha, model_size, angle, mode)

        fig.savefig(f'{path_sum}\\{file_name}')
        plt.close(fig)

    def isofields_coefficients_thread(self,
                                      alpha: str,
                                      model_size: Tuple[str, str, str],
                                      angle: str,
                                      mode: str,
                                      path_report: str):
        """Функция для запуска потока генерации изополей и их сохранения."""

        file_name = f'Изополя {" ".join(model_size)} {alpha} {int(angle):02} {mode}.png'

        fig = self.get_plot_isofields(alpha, model_size, str(angle), mode)

        fig.savefig(
            f'{path_report}\\Изополя ветровых нагрузок и воздействий\\Коэффициенты\\{mode}\\{file_name}')

        plt.close(fig)

    def draw_isofields_coefficients(self,
                                    alpha: str,
                                    model_size: Tuple[str, str, str],
                                    angle_border: int,
                                    path_report: str,
                                    mods: List):
        """Функция запускает отрисовку всех изополей выбранной модели.
        Отрисовка происходит в многопоточном режиме.
        """
        self.logger.info(f'Отрисовка изополей размеры = {" ".join(model_size)} альфа = {alpha}')

        for mode in mods:
            args = [(alpha, model_size, str(angle), mode, path_report) for angle in
                    range(0, angle_border, 5)]

            with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
                executor.map(lambda i: self.isofields_coefficients_thread(*i), args)

        self.logger.info(f'Отрисовка изополей размеры = {" ".join(model_size)} альфа = {alpha} завершена')

    def isofields_pressure_thread(self,
                                  alpha: str,
                                  model_size: Tuple[str, str, str],
                                  angle: str,
                                  mode: str,
                                  path_report: str,
                                  pressure_plot_parameters):
        file_name = f'Изополя давления {" ".join(model_size)} {alpha} {int(angle):02} {mode}.png'

        fig = self.get_plot_isofields(alpha, model_size, str(angle), mode, pressure_plot_parameters)

        fig.savefig(
            f'{path_report}\\Изополя ветровых нагрузок и воздействий\\Давление\\{mode}\\{file_name}')
        plt.close(fig)

    def draw_isofields_pressure(self, alpha, model_size, angle_border, path_report, mods, pressure_plot_parameters):
        for mode in mods:
            args = [(alpha, model_size, str(angle), mode, path_report, pressure_plot_parameters) for angle in
                    range(0, angle_border, 5)]

            with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
                executor.map(lambda i: self.isofields_pressure_thread(*i), args)

    def draw_pseudocolor_coefficients(self, alpha, model_size, angle_border, path_report, mods):
        for mode in mods:
            args = [(alpha, model_size, str(angle), mode, path_report) for angle in
                    range(0, angle_border, 5)]

            with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
                executor.map(lambda i: self.pseudocolor_coefficients_thread(*i), args)

    def pseudocolor_coefficients_thread(self, alpha, model_size, angle, mode, path_report):
        file_name = f'Мозаика коэффициентов {" ".join(model_size)} {alpha} {int(angle):02} {mode}.png'

        fig = self.get_pseudocolor_coefficients(alpha, model_size, angle, mode)

        fig.savefig(
            f'{path_report}\\Мозаика коэффициентов\\{mode}\\{file_name}')
        plt.close(fig)

    def draw_summary_coefficients_polar(self,
                                        alpha: str,
                                        model_scale: str,
                                        model_size: Tuple[str, str, str],
                                        angle_border: int,
                                        path_report: str,
                                        mods):
        """Функция запускает отрисовку всех графиков суммарных коэффициентов в полярной системе для выбранной модели.
        Отрисовка происходит в многопоточном режиме.
        """
        self.logger.info(f'Отрисовка суммарных аэродинамических коэффициентов в полярной системе координат '
                         f'размеры = {" ".join(model_size)} альфа = {alpha}')

        args = [(alpha, model_scale, model_size, angle_border, mode) for mode in mods]

        with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
            polar_plots = list(executor.map(lambda i: self.get_plot_summary_coefficients_polar(*i), args))

        for mode, fig in zip(mods, polar_plots):
            file_name = f'Суммарные аэродинамические коэффициенты Cx Cy CMz ' \
                        f'{" ".join(model_size)} {id_to_name[mode]} в полярной системе координат.png'

            fig.savefig(f'{path_report}\\Суммарные аэродинамические коэффициенты\\Полярная система координат\\'
                         f'{file_name}')
            plt.close(fig)

        self.logger.info(f'Отрисовка суммарных аэродинамических коэффициентов в полярной системе координат '
                         f'размеры = {" ".join(model_size)} альфа = {alpha} завершена')

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
            'mean': np.mean,
            'rms': rms,
            'std': np.std,
            'max': np.max,
            'min': np.min,
            'rach': rach,
            'obesP': obes_m,
            'obesM': obes_p
        }

        id_fig = f'summary_coefficients_Cx_Cy_CMz_polar_{"_".join(model_size)}_{mode}'

        if not self.clipboard_obj.clipboard_dict[alpha][model_scale]['const_stuff'].get(id_fig):
            self.clipboard_obj.get_coordinates(alpha, model_scale)

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

            x_scale, y_scale = Plot.scaling_data(list_norm_cx, list_norm_cy, angle_border=angle_border)
            cmz_scale = Plot.scaling_data(list_norm_cmz, angle_border=angle_border)

            data = {
                'Cx': x_scale,
                'Cy': y_scale,
                'CMz': cmz_scale,
            }

            fig = Plot.polar_plot(data, mode, model_size, alpha)
            self.clipboard_obj.clipboard_dict[alpha][model_scale]['const_stuff'][id_fig] = fig

        fig = self.clipboard_obj.clipboard_dict[alpha][model_scale]['const_stuff'].get(id_fig)

        self.logger.info(f'Запрос суммарных коэффициентов в полярной системе координат размеры = '
                         f'{" ".join(list(model_size))} альфа = {alpha} режим = {mode.ljust(4, " ")} из буфера успешно выполнен')

        return fig

    def get_envelopes(self,
                      alpha: str,
                      model_scale: str,
                      angle: str,
                      mods = ['mean', 'rms', 'std', 'max', 'min']):
        """Функция возвращает графики огибающих.
        Если огибающие отсутствуют в буфере, запускается отрисовка.
        """
        self.logger.info(f'Запрос огибающих параметры = {" ".join(list(model_scale))} '
                         f'альфа = {alpha} угол = {angle.rjust(2, "0")} из буфера')

        if not self.clipboard_obj.clipboard_dict[alpha][model_scale][angle].get('envelopes'):
            self.logger.info(f'Отрисовка огибающих {" ".join(list(model_scale))} {alpha} {int(angle):02}')

            pressure_coefficients = self.clipboard_obj.get_pressure_coefficients(alpha, model_scale, angle)
            figs = Plot.envelopes(pressure_coefficients, alpha, model_scale, angle, mods)

            for ind, val in enumerate(figs):
                self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][f'envelopes_{ind}_{"_".join(mods)}'] = val
            self.clipboard_obj.clipboard_dict[alpha][model_scale][angle]['envelopes'] = True
            self.logger.info(f'Отрисовка огибающих {" ".join(list(model_scale))} {alpha} {int(angle):02} завершена')

        figs = []
        i = 0
        while True:
            if not self.clipboard_obj.clipboard_dict[alpha][model_scale][angle].get(f'envelopes_{i}_{"_".join(mods)}'):
                break
            else:
                figs.append(
                    self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][f'envelopes_{i}_{"_".join(mods)}'])
                i += 1

        self.logger.info(f'Запрос огибающих параметры = {" ".join(list(model_scale))} '
                         f'альфа = {alpha} угол = {angle.rjust(2, "0")} из буфера успешно выполнен')

        return figs

    def draw_envelopes(self,
                       alpha: str,
                       model_scale: str,
                       angle_border: int,
                       path_report: str,
                       mods):
        """Функция запускает отрисовку всех огибающих выбранной модели.
        Отрисовка происходит в многопоточном режиме.
        """
        self.logger.info(f'Отрисовка огибающих параметры = {" ".join(model_scale)} альфа = {alpha}')

        args = [(alpha, model_scale, str(angle), mods, path_report) for angle in
                range(0, angle_border, 5)]

        with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
            executor.map(lambda i: self.envelopes_thread(*i), args)

        self.logger.info(f'Отрисовка огибающих параметры = {" ".join(model_scale)} альфа = {alpha} завершена')

    def envelopes_thread(self,
                         alpha: str,
                         model_scale: str,
                         angle: str,
                         mods,
                         path_report: str):
        """Функция для запуска потока генерации графиков огибающих модели и их сохранения."""
        figs = self.get_envelopes(alpha, model_scale, angle, mods)
        path_envelopes = f'{path_report}\\Огибающие\\{model_scale} {alpha} {angle}'

        if not os.path.isdir(path_envelopes):
            os.mkdir(path_envelopes)

        for i in range(len(figs)):
            file_name = f'Огибающие {model_scale} {alpha} {int(angle):02} ' \
                        f'{i * 100} - {(i + 1) * 100} {" ".join(mods)}.png'
            figs[i].savefig(f'{path_envelopes}\\{file_name}')
            plt.close(figs[i])

    @staticmethod
    def start_report(clipboard_dict, alpha, model_size, pressure_plot_parameters, content):
        new_core = Core(clipboard_dict)
        new_core.report(alpha, model_size, pressure_plot_parameters, content)

    @staticmethod
    def check_alive_proc(proc, button, spinner):
        while proc.is_alive():
            time.sleep(3.993)

        button.disabled = False
        spinner.active = False

    def wrapper_for_clipboard(self):
        if not self._manager:
            self._manager = Manager()

        self.clipboard_obj.clipboard_dict = to_multiprocessing_dict(self.clipboard_obj.clipboard_dict, self._manager)

    def preparation_for_report(self, alpha: str,
                               model_size: Tuple[str, str, str],
                               pressure_plot_parameters,
                               content,
                               button,
                               spinner):
        # if isinstance(self.clipboard_obj.clipboard_dict, dict):
        #     self.wrapper_for_clipboard()

        #self.report(alpha, model_size, pressure_plot_parameters, content)

        args = (self.clipboard_obj.clipboard_dict, alpha, model_size, pressure_plot_parameters, content)
        proc = Process(target=Core.start_report, args=args)
        proc.start()

        executor = ThreadPoolExecutor(max_workers=1)
        executor.map(lambda i: Core.check_alive_proc(*i), ((proc, button, spinner),))

    def draw_plot_model_3d(self, model_size, path_report):
        fig = self.get_plot_model_3d(model_size)
        fig.savefig(f'{path_report}\\Модель\\Модель трехмерная.png')
        plt.close(fig)

    def draw_model_polar(self, model_size, path_report):
        fig = self.get_model_polar(model_size)
        fig.savefig(f'{path_report}\\Модель\\Модель в полярной системе.png')
        plt.close(fig)

    def draw_plot_pressure_tap_locations(self, model_size, alpha, path_report):
        fig = self.get_plot_pressure_tap_locations(model_size, alpha)
        fig.savefig(f'{path_report}\\Модель\\Развертка модели.png')
        plt.close(fig)

    def get_summary_coefficients_statistics(self,
                                            angle: int,
                                            alpha: str,
                                            model_name: str,
                                            mods: List[str], ):
        accuracy_values = 3
        statistics = []

        data = dict()

        cmz = self.clipboard_obj.get_cmz(alpha, model_name, str(angle))
        data['cmz'] = cmz

        cx, cy = self.clipboard_obj.get_cx_cy(alpha, model_name, str(angle))
        data['cx'] = cx
        data['cy'] = cy

        functions = {
            'max': lambda d: np.max(d, axis=0).round(accuracy_values),
            'mean': lambda d: np.mean(d, axis=0).round(accuracy_values),
            'min': lambda d: np.min(d, axis=0).round(accuracy_values),
            'std': lambda d: np.std(d, axis=0).round(accuracy_values),
            'rms': lambda d: rms(d).round(accuracy_values),
            'rach': lambda d: rach(d).round(accuracy_values),
            'obesP': lambda d: obes_p(d).round(accuracy_values),
            'obesM': lambda d: obes_m(d).round(accuracy_values)}

        for name in data.keys():
            statistics_name = []
            statistics_name.append(name.capitalize())

            for mode in mods:
                statistics_name.append(functions[mode](data[name]))

            statistics.append(statistics_name)

        return statistics

    def get_sensor_statistics(self,
                              alpha: str,
                              model_scale: str,
                              angle: int,
                              model_size: Tuple[str, str, str],
                              mods: List[str]):

        accuracy_values = 2

        breadth_real, depth_real, height_real = float(model_size[0]), float(model_size[1]), float(model_size[2])

        face_number = self.clipboard_obj.get_face_number(alpha, model_scale)

        x, z = self.clipboard_obj.get_coordinates(alpha, model_scale)

        sensors_on_model = len(x)
        x_new, y_new = converter_coordinates(x, breadth_real, depth_real, face_number, sensors_on_model)

        pressure_coefficients = self.clipboard_obj.get_pressure_coefficients(alpha, model_scale, str(angle))
        pressure_coefficients_t = pressure_coefficients.T

        functions = {'x': lambda: x_new,
                     'y': lambda: y_new,
                     'z': lambda: z,
                     'max': lambda: np.max(pressure_coefficients, axis=0).round(accuracy_values),
                     'mean': lambda: np.mean(pressure_coefficients, axis=0).round(accuracy_values),
                     'min': lambda: np.min(pressure_coefficients, axis=0).round(accuracy_values),
                     'std': lambda: np.std(pressure_coefficients, axis=0).round(accuracy_values),
                     'rms': lambda: [rms(i).round(accuracy_values) for i in pressure_coefficients_t],
                     'rach': lambda: [rach(i).round(accuracy_values) for i in pressure_coefficients_t],
                     'obesP': lambda: [obes_p(i).round(accuracy_values) for i in pressure_coefficients_t],
                     'obesM': lambda: [obes_m(i).round(accuracy_values) for i in pressure_coefficients_t]}

        statistics_of_angle = [[i for i in range(1, sensors_on_model + 1)]]

        for k in mods:
            statistics_of_angle.append(functions[k]())

        return np.array(statistics_of_angle).T

    def report(self, alpha: str, model_size: Tuple[str, str, str], pressure_plot_parameters, content):
        t1 = time.time()

        self.logger.info(f'Начало формирования отчета размеры = {" ".join(model_size)} альфа = {alpha}')

        current_path = os.getcwd()

        model_scale, scale_factors = get_model_and_scale_factors(*model_size, alpha)
        breadth, depth, height = model_size
        name_report = f'Отчет ширина {breadth} глубина {depth} высота {height} альфа {alpha}'
        path_report = f'{current_path}\\Отчеты\\{name_report}'
        generate_directory_for_report(current_path, name_report)

        if model_scale[0] == model_scale[1]:
            angle_border = 50
        else:
            angle_border = 95

        # Только эти графики используют координаты
        if any((content['isofieldsCoefficients'][0],
                content['pressureTapLocations'][0],
                content['pseudocolorCoefficients'][0],
                content['isofieldsPressure'][0],
                content['summaryCoefficients'][0],
                content['statisticsSensors'][0],
                content['statisticsSummaryCoefficients'][0],
                content['summarySpectres'][0])):
            self.clipboard_obj.get_coordinates(alpha, model_scale)
            args = [(alpha, model_scale, angle) for angle in range(angle_border)]
            with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
                executor.map(lambda i: self.get_pressure_coefficients(*i), args)

        # Отрисовка графиков
        self.logger.info(f'Отрисовка графиков')
        # isofieldsPressure
        if content['isofieldsPressure'][0]:
            mods = [k for k in content['isofieldsPressure'][1].keys() if content['isofieldsPressure'][1][k]]
            self.draw_isofields_pressure(alpha, model_size, angle_border, path_report, mods, pressure_plot_parameters)
        # isofieldsCoefficients
        if content['isofieldsCoefficients'][0]:
            mods = [k for k in content['isofieldsCoefficients'][1].keys() if content['isofieldsCoefficients'][1][k]]
            self.draw_isofields_coefficients(alpha, model_size, angle_border, path_report, mods)

        # pseudocolorCoefficients
        if content['pseudocolorCoefficients'][0]:
            mods = [k for k in content['pseudocolorCoefficients'][1].keys() if content['pseudocolorCoefficients'][1][k]]
            self.draw_pseudocolor_coefficients(alpha, model_size, angle_border, path_report, mods)
        # envelopes
        if content['envelopes'][0]:
            mods = [k for k in content['envelopes'][1].keys() if content['envelopes'][1][k]]
            self.draw_envelopes(alpha, model_scale, angle_border, path_report, mods)
        # polarSummaryCoefficients
        if content['polarSummaryCoefficients'][0]:
            mods_sum_coef_polar = [k for k in content['polarSummaryCoefficients'][1].keys() if
                                   content['polarSummaryCoefficients'][1][k]]
            self.draw_summary_coefficients_polar(alpha, model_scale, model_size, angle_border, path_report,
                                                 mods_sum_coef_polar)
        # summaryCoefficients
        if content['summaryCoefficients'][0]:
            mods_sum_coef = [k for k in content['summaryCoefficients'][1].keys() if
                             content['summaryCoefficients'][1][k]]
            self.draw_summary_coefficients(alpha, model_size, angle_border, path_report, mods_sum_coef)
        # summarySpectres
        if content['summarySpectres'][0]:
            mods_sum_spec = [k for k in content['summarySpectres'][1].keys() if content['summarySpectres'][1][k]]
            self.draw_welch_graphs(alpha, model_scale, model_size, angle_border, path_report, mods_sum_spec)
        # pressureTapLocations
        if content['pressureTapLocations'][0]:
            self.draw_plot_pressure_tap_locations(model_size, alpha, path_report)

        self.draw_plot_model_3d(model_size, path_report)
        self.draw_model_polar(model_size, path_report)

        self.logger.info(f'Отрисовка графиков все')

        # Получение статистики
        self.logger.info(f'Получение статистики')
        # statisticsSensors
        # statistics_sensors = [
        #     угол 0
        #     [
        #         [1 ...],
        #         [2 ...],
        #         [3 ...],
        #         ...
        #     ],
        #     угол 5
        #     [
        #         [1 ...],
        #         [2 ...],
        #         [3 ...],
        #         ...
        #     ],
        # ]
        if content['statisticsSensors'][0]:
            mods_stat_sens = [k for k in content['statisticsSensors'][1].keys() if content['statisticsSensors'][1][k]]
            if mods_stat_sens:
                args = [(alpha, model_scale, angle, model_size, mods_stat_sens) for angle in range(0, angle_border, 5)]
                with ThreadPoolExecutor(max_workers=self._count_threads) as executor:
                    statistics_sensors = list(executor.map(lambda i: self.get_sensor_statistics(*i), args))

        # statisticsSummaryCoefficients
        # statistics_summary_coefficients = [
        #     угол 0
        #     [
        #         [cx ...],
        #         [cy ...],
        #         [cmz ...],
        #     ],
        #     угол 5
        #     [
        #         [cx ...],
        #         [cy ...],
        #         [cmz ...],
        #     ],
        # ]
        if content['statisticsSummaryCoefficients'][0]:
            mods_stat_sum_coeff = [k for k in content['statisticsSummaryCoefficients'][1].keys()
                                   if content['statisticsSummaryCoefficients'][1][k]]

            args = [(angle, alpha, model_scale, mods_stat_sum_coeff) for angle in range(0, angle_border, 5)]
            with ThreadPoolExecutor(max_workers=self._count_threads) as executor:
                statistics_summary_coefficients = list(
                    executor.map(lambda i: self.get_summary_coefficients_statistics(*i), args))

        self.logger.info(f'Получение статистики все')

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
        # ширина A4 210 мм высота 297 мм
        fig_width = Mm(165)

        # Шрифт заголовков разного уровня
        head_lvl1 = Pt(20)
        head_lvl2 = Pt(16)

        counter_plots = 1  # Счетчик графиков для нумерации
        counter_tables = 1  # Счетчик таблиц для нумерации
        counter_head = 1  # Счетчик заголовков

        title = doc.add_heading()
        run = title.add_run(f'Отчет по зданию {breadth}x{depth}x{height}')
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run.font.size = Pt(24)
        run.bold = True

        for i in ('Параметры ветрового воздействия:',
                  f'Ветровой район: {pressure_plot_parameters["wind_region"]}',
                  f'Тип местности: {pressure_plot_parameters["type_area"]}'
                  ):
            doc.add_paragraph().add_run(i)

        p = doc.add_paragraph()
        run = p.add_run()
        run.add_picture(f'{path_report}\\Модель\\Модель трехмерная.png', width=fig_width / 2)
        run.add_picture(f'{path_report}\\Модель\\Модель в полярной системе.png', width=fig_width / 2)
        run.add_text(f'Рисунок {counter_plots}. '
                     f'Геометрические размеры и система координат направления ветровых потоков')
        counter_plots += 1

        self.logger.info('1. Геометрические размеры здания')

        head = doc.add_heading()
        run = head.add_run(f'{counter_head}. Геометрические размеры здания')
        head.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run.font.size = head_lvl1

        counter_head += 1

        header_model = (
            'Геометрический размер',
            'Значение, м'
        )

        table_model = doc.add_table(rows=1, cols=len(header_model))
        table_model.style = 'Table Grid'

        head_cells = table_model.rows[0].cells
        for i, name in enumerate(header_model):
            p = head_cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cell = p.add_run(name)
            cell.bold = True

        for i, j in zip((breadth, depth, height), ('Ширина:', 'Глубина:', 'Высота:')):
            row_cells = table_model.add_row().cells
            row_cells[0].text = j
            row_cells[1].text = str(i)

        if content['pressureTapLocations'][0]:
            p = doc.add_paragraph()
            run = p.add_run()
            run.add_picture(f'{path_report}\\Модель\\Развертка модели.png', width=fig_width)
            run.add_text(f'Рисунок {counter_plots}. Система датчиков мониторинга')
            counter_plots += 1
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER

        doc.add_page_break()

        self.logger.info(f'{counter_head}. Статистика по датчиках. Максимумы и огибающие')
        if content['statisticsSensors'][0]:
            head = doc.add_heading()
            run = head.add_run(f'{counter_head}. Статистика по датчиках. Максимумы и огибающие')
            head.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run.font.size = head_lvl1

            counter_head += 1

            for angle in range(0, angle_border, 5):
                envelopes = glob.glob(f'{path_report}\\Огибающие\\{model_scale} {alpha} {angle}\\'
                                      f'Огибающие * {angle:02}*.png')
                for i in envelopes:
                    p = doc.add_paragraph()
                    run = p.add_run()
                    run.add_picture(i, width=fig_width)
                    run.add_text(
                        f'Рисунок {counter_plots}. Огибающие ветрового давления для здания '
                        f'{breadth}x{depth}x{height} угол {angle:02}º '
                        f'датчики {i[i[:i.rfind("-") - 1].rfind(" ") + 1:i.rfind(".")]}')
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    counter_plots += 1
            doc.add_page_break()

        self.logger.info(f'{counter_head}. Изополя ветровых нагрузок и воздействий')
        mods_isofields = ('max', 'mean', 'min', 'std')
        if content['isofieldsPressure'][0] or content['isofieldsCoefficients'][0]:
            head = doc.add_heading()
            run = head.add_run(f'{counter_head}. Изополя ветровых нагрузок и воздействий')
            run.font.size = head_lvl1
            head.alignment = WD_ALIGN_PARAGRAPH.CENTER

            counter_head += 1

        self.logger.info('3.1 Коэффициенты изополя')
        counter_head_lvl2 = 0.1
        if content['isofieldsCoefficients'][0]:
            head = doc.add_heading(level=2)
            run = head.add_run(f'{counter_head+counter_head_lvl2} Коэффициенты изополя')
            head.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run.font.size = head_lvl2

            counter_head_lvl2 += 0.1

            for mode in mods_isofields:
                for angle in range(0, angle_border, 5):
                    isofields = f'{path_report}\\Изополя ветровых нагрузок и воздействий\\Коэффициенты\\{mode}\\' \
                                f'Изополя {" ".join(model_size)} {alpha} {angle:02} {mode}.png'

                    p = doc.add_paragraph()
                    run = p.add_run()
                    run.add_picture(isofields)
                    run.add_text(
                        f'Рисунок {counter_plots}. Изополя коэффициентов {mode} '
                        f'для здания {breadth}x{depth}x{height} угол {angle:02}º')
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    counter_plots += 1

        self.logger.info('3.2 Изополя давления')
        if content['isofieldsPressure'][0]:
            head = doc.add_heading(level=2)
            run = head.add_run(f'{counter_head+counter_head_lvl2}. Изополя давления')
            head.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run.font.size = head_lvl2

            counter_head_lvl2 += 0.1


            for mode in mods_isofields:
                for angle in range(0, angle_border, 5):
                    isofields = f'{path_report}\\Изополя ветровых нагрузок и воздействий\\Давление\\{mode}\\' \
                                f'Изополя давления {" ".join(model_size)} {alpha} {angle:02} {mode}.png'

                    p = doc.add_paragraph()
                    run = p.add_run()
                    run.add_picture(isofields)
                    run.add_text(
                        f'Рисунок {counter_plots}. Изополя давления {mode} '
                        f'для здания {breadth}x{depth}x{height} угол {angle:02}º')
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    counter_plots += 1
            doc.add_page_break()

        self.logger.info('4. Статистика по датчикам в табличном виде')

        if content['statisticsSensors'][0]:
            head = doc.add_heading()
            run = head.add_run(f'{counter_head}. Статистика по датчикам в табличном виде ')
            run.font.size = head_lvl1
            head.alignment = WD_ALIGN_PARAGRAPH.CENTER

            counter_head += 1

            header_sensors = ['Датчик']
            header_sensors.extend([id_to_name[mode] for mode in mods_stat_sens])

            for angle in range(0, angle_border, 5):
                self.logger.info(f'4. Статистика по датчикам в табличном виде угол {angle}')
                p = doc.add_paragraph()
                p.add_run(
                    f'\nТаблица {counter_tables}. Аэродинамический коэффициент в датчиках для '
                    f'здания {breadth}x{depth}x{height} угол {angle:02}º')
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                counter_tables += 1
                table_sensors = doc.add_table(rows=1, cols=len(header_sensors))
                table_sensors.style = 'Table Grid'
                head_cells = table_sensors.rows[0].cells
                for i, name in enumerate(header_sensors):
                    p = head_cells[i].paragraphs[0]
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

                    cell = p.add_run(name)
                    cell.bold = True
                    cell.font.size = Pt(8)

                for row in statistics_sensors[angle // 5]:
                    cells = table_sensors.add_row().cells
                    for i, value in enumerate(row):
                        cells[i].text = str(value)

                        cells[i].paragraphs[0].runs[0].font.size = Pt(12)

            doc.add_page_break()

            del statistics_sensors

        self.logger.info(f'{counter_head}. Суммарные значения аэродинамических коэффициентов')
        if content['summaryCoefficients'][0]:
            head = doc.add_heading()
            run = head.add_run(f'{counter_head}. Графики суммарных аэродинамических коэффициентов')
            run.font.size = head_lvl1
            head.alignment = WD_ALIGN_PARAGRAPH.CENTER

            counter_head += 1

            for angle in range(0, angle_border, 5):
                for mode in mods_sum_coef:
                    mode = ' '.join(mode.lower().title().split('_'))
                    self.logger.info(f'5. Суммарные значения аэродинамических коэффициентов угол {angle}')
                    p = doc.add_paragraph()
                    run = p.add_run()
                    run.add_picture(
                        f'{path_report}\\Суммарные аэродинамические коэффициенты\\Декартовая система координат\\{angle:02}\\'
                        f'Суммарные аэродинамические коэффициенты {mode} {" ".join(model_size)} {alpha} {angle:02}.png')
                    run.add_text(
                        f'Рисунок {counter_plots}. Суммарные аэродинамические коэффициенты '
                        f'для здания {breadth}x{depth}x{height} угол {angle:02}º')
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    counter_plots += 1

        if content['statisticsSummaryCoefficients'][0]:
            head = doc.add_heading()
            run = head.add_run(f'{counter_head}. Статистика суммарных аэродинамических коэффициентов')
            run.font.size = head_lvl1
            head.alignment = WD_ALIGN_PARAGRAPH.CENTER

            counter_head += 1

            header_sum = ['Сила']
            header_sum.extend([id_to_name[mode] for mode in mods_stat_sum_coeff])

            for angle in range(0, angle_border, 5):
                doc.add_paragraph().add_run(
                    f'Таблица {counter_tables}. Суммарные аэродинамические коэффициенты '
                    f'для здания {breadth}x{depth}x{height} угол {angle:02}º')
                counter_tables += 1

                table = doc.add_table(rows=1, cols=len(header_sum))
                table.style = 'Table Grid'
                head_cells = table.rows[0].cells
                for i, name in enumerate(header_sum):
                    p = head_cells[i].paragraphs[0]
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

                    cell = p.add_run(name)
                    cell.bold = True
                    cell.font.size = Pt(8)

                for row in statistics_summary_coefficients[angle // 5]:
                    cells = table.add_row().cells
                    for i, value in enumerate(row):
                        cells[i].text = str(value)

                        cells[i].paragraphs[0].runs[0].font.size = Pt(12)

            del statistics_summary_coefficients
            doc.add_page_break()

        if content['polarSummaryCoefficients'][0]:
            head = doc.add_heading()
            run = head.add_run(f'{counter_head}. Cуммарные аэродинамических коэффициенты в полярной системе координат')
            run.font.size = head_lvl1
            head.alignment = WD_ALIGN_PARAGRAPH.CENTER

            counter_head += 1

            for mode in mods_sum_coef_polar:
                p = doc.add_paragraph()
                run = p.add_run()
                run.add_picture(f'{path_report}\\Суммарные аэродинамические коэффициенты\\Полярная система координат\\'
                                f'Суммарные аэродинамические коэффициенты Cx Cy CMz {breadth} {depth} {height} '
                                f'{id_to_name[mode]} в полярной системе координат.png')

                run.add_text(
                    f'Рисунок {counter_plots}. Суммарные аэродинамические коэффициенты в полярной системе координат '
                    f'{id_to_name[mode]} для здания {breadth}x{depth}x{height}')
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                counter_plots += 1
                doc.add_page_break()

        self.logger.info('6. Спектры cуммарных значений аэродинамических коэффициентов')

        if content['summarySpectres'][0]:
            head = doc.add_heading()
            run = head.add_run(f'{counter_head}. Спектры cуммарных значений аэродинамических коэффициентов')
            run.font.size = head_lvl1
            head.alignment = WD_ALIGN_PARAGRAPH.CENTER

            counter_head += 1

            for angle in range(0, angle_border, 5):
                self.logger.info(f'8. Спектры cуммарных значений аэродинамических коэффициентов угол {angle}')
                for mode in mods_sum_spec:
                    mode = ' '.join(mode.lower().title().split('_'))
                    p = doc.add_paragraph()
                    run = p.add_run()
                    run.add_picture(f'{path_report}\\Спектральная плотность мощности\\Логарифмическая шкала\\'
                                    f'Спектральная плотность мощности {model_scale} {alpha} {mode} {angle:02}.png')
                    run.add_text(
                        f'Рисунок {counter_plots}. Спектр cуммарных значений аэродинамических коэффициентов '
                        f'для здания {breadth}x{depth}x{height} {mode} угол {angle:02} º')
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    counter_plots += 1
            doc.add_page_break()

        doc.save(f'{path_report}\\Отчет ширина {breadth} глубина {depth} высота {height} альфа {alpha}.docx')
        os.startfile(f'{path_report}\\Отчет ширина {breadth} глубина {depth} высота {height} альфа {alpha}.docx')

        print(time.time() - t1)


if __name__ == '__main__':
    c = Core()
    c.report('6', ('3', '1', '5'))
    # run_proc(c.clipboard_obj.clipboard_dict, '4', ('1', '1', '1'))
    # fig = c.get_plot_summary_coefficients('6', ('0.1', '0.1', '0.1'), '0', 'Cx Cy CMz')
    # utils.utils.open_fig(fig)
    # t1 = time.time()
    # c.preparation_for_report('6', ('1', '4', '8'))
    # c.get_plot_isofields('6', ('1', '1', '1'),'0','mean','isofields_coefficients')
    # c.save_clipboard()
    # c.preparation_for_report('6', ('1', '1', '1'))
    # print(time.time() - t1)
    # print('=========================================')
    # c.report('6', ('2', '2', '2')) alpha: str,
    #                            model_size: Tuple[str, str, str],
    #                            angle: str,
    #                            mode: str,
    #                            type_plot: str):
    # print(os.listdir('D:\\'))
